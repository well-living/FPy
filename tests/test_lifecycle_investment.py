"""
Unit tests for LifecycleInvestmentSimulator

This module provides comprehensive unit tests for the LifecycleInvestmentSimulator class,
covering initialization, parameter validation, rate processing, simulation execution,
and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Adjust imports based on actual module structure
from fpyjp.core.balance_simulator import LifecycleInvestmentSimulator
from fpyjp.core.interest_factor import InterestFactor


class TestLifecycleInvestmentSimulatorInit:
    """Test initialization and parameter validation."""
    
    def test_period_based_initialization_success(self):
        """Test successful initialization with period-based specification."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=20,
            hold_periods=10,
            decumulation_periods=15,
        )
        
        assert simulator.contribution_amount == 10000
        assert simulator.accumulation_periods == 20
        assert simulator.hold_periods == 10
        assert simulator.decumulation_periods == 15
        assert simulator.accumulation_rate == 0.05
        assert simulator.hold_rate == 0.05
        assert simulator.decumulation_rate == 0.05
    
    def test_timepoint_based_initialization_success(self):
        """Test successful initialization with time-point-based specification."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_end_period=20,
            decumulation_start_period=30,
            simulation_end_period=45,
        )
        
        assert simulator.contribution_amount == 10000
        assert simulator.accumulation_periods == 20
        assert simulator.hold_periods == 10
        assert simulator.decumulation_periods == 15
    
    def test_both_specifications_error(self):
        """Test error when both period-based and time-point-based specs are provided."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                rate=0.05,
                # Both specifications provided - this should trigger the error
                accumulation_periods=20,
                hold_periods=10,
                decumulation_periods=15,
                accumulation_end_period=20,
                decumulation_start_period=30,
                simulation_end_period=45,
            )
    
    def test_no_specification_error(self):
        """Test error when neither specification method is provided."""
        with pytest.raises(ValueError, match="Must specify either all period-based parameters or all time-point-based"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                rate=0.05,
            )
    
    def test_incomplete_period_specification_error(self):
        """Test error when period-based specification is incomplete."""
        with pytest.raises(ValueError, match="Must specify either all period-based parameters or all time-point-based"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                rate=0.05,
                accumulation_periods=20,
                hold_periods=10,
                # missing decumulation_periods
            )
    
    def test_incomplete_timepoint_specification_error(self):
        """Test error when time-point-based specification is incomplete."""
        with pytest.raises(ValueError, match="Must specify either all period-based parameters or all time-point-based"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                rate=0.05,
                accumulation_end_period=20,
                decumulation_start_period=30,
                # missing simulation_end_period
            )
    
    def test_negative_periods_error(self):
        """Test error when periods are negative or zero."""
        with pytest.raises(ValueError, match="accumulation_periods and decumulation_periods must be positive"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                rate=0.05,
                accumulation_periods=0,
                hold_periods=10,
                decumulation_periods=15,
            )
    
    def test_invalid_timepoint_order_error(self):
        """Test error when time points are not in ascending order."""
        with pytest.raises(ValueError, match="Time points must be in order"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                rate=0.05,
                accumulation_end_period=30,  # Wrong order
                decumulation_start_period=20,
                simulation_end_period=45,
            )
    
    def test_no_rate_specification_error(self):
        """Test error when no rate is specified."""
        with pytest.raises(ValueError, match="At least one rate specification must be provided"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                accumulation_periods=20,
                hold_periods=10,
                decumulation_periods=15,
            )


class TestRateProcessing:
    """Test rate processing logic."""
    
    def test_single_rate_for_all_phases(self):
        """Test using single rate for all phases."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=20,
            hold_periods=10,
            decumulation_periods=15,
        )
        
        assert simulator.accumulation_rate == 0.05
        assert simulator.hold_rate == 0.05
        assert simulator.decumulation_rate == 0.05
    
    def test_individual_phase_rates_priority(self):
        """Test that individual phase rates take priority over general rate."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            rate_during_accumulation=0.07,
            rate_during_hold=0.04,
            rate_during_decumulation=0.03,
            accumulation_periods=20,
            hold_periods=10,
            decumulation_periods=15,
        )
        
        assert simulator.accumulation_rate == 0.07
        assert simulator.hold_rate == 0.04
        assert simulator.decumulation_rate == 0.03
    
    def test_mixed_rate_specification(self):
        """Test mixed rate specification with fallback."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            rate_during_accumulation=0.08,
            rate_during_decumulation=0.02,
            # rate_during_hold not specified, should use fallback
            accumulation_periods=20,
            hold_periods=10,
            decumulation_periods=15,
        )
        
        assert simulator.accumulation_rate == 0.08
        assert simulator.hold_rate == 0.05  # Fallback
        assert simulator.decumulation_rate == 0.02
    
    def test_rate_list_distribution(self):
        """Test rate list distribution across phases."""
        rate_list = [0.08, 0.06, 0.04, 0.03, 0.02]
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=rate_list,
            accumulation_periods=2,
            hold_periods=1,
            decumulation_periods=2,
        )
        
        assert simulator.accumulation_rate == [0.08, 0.06]
        assert simulator.hold_rate == [0.04]
        assert simulator.decumulation_rate == [0.03, 0.02]
    
    def test_rate_list_length_mismatch_error(self):
        """Test error when rate list length doesn't match total periods."""
        rate_list = [0.08, 0.06, 0.04]  # 3 rates for 5 periods
        with pytest.raises(ValueError, match="Rate list length .* must equal total periods"):
            LifecycleInvestmentSimulator(
                contribution_amount=10000,
                rate=rate_list,
                accumulation_periods=2,
                hold_periods=1,
                decumulation_periods=2,  # Total: 5 periods
            )


class TestSimulationExecution:
    """Test simulation execution and results."""
    
    def test_simulate_integration_basic(self):
        """Test basic integration simulation without mocks."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=2,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        result = simulator.simulate()
        
        # Basic structure checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert result.index.name == 'time_period'
        
        # Check that accumulation shows contributions
        accumulation_rows = result.iloc[:2]
        assert all(accumulation_rows['cash_outflow'] == 10000)
        
        # Check that hold period has no cash flows
        hold_row = result.iloc[2]
        assert hold_row['cash_outflow'] == 0
        assert hold_row['capital_cash_inflow_before_tax'] == 0
        
        # Check that decumulation shows withdrawals
        decumulation_row = result.iloc[3]
        assert decumulation_row['capital_cash_inflow_before_tax'] > 0
        assert decumulation_row['cash_outflow'] == 0
    
    def test_individual_phase_dataframes_accessible(self):
        """Test that individual phase DataFrames are accessible after simulation."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=2,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        simulator.simulate()
        
        # Check individual phase DataFrames exist and have correct lengths
        assert isinstance(simulator.accumulation_df, pd.DataFrame)
        assert len(simulator.accumulation_df) == 2
        
        assert isinstance(simulator.hold_df, pd.DataFrame)
        assert len(simulator.hold_df) == 1
        
        assert isinstance(simulator.decumulation_df, pd.DataFrame)
        assert len(simulator.decumulation_df) == 1
        
        assert isinstance(simulator.combined_df, pd.DataFrame)
        assert len(simulator.combined_df) == 4


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_contribution_amount(self):
        """Test with zero contribution amount."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=0,
            rate=0.05,
            accumulation_periods=2,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        result = simulator.simulate()
        
        # Should not crash, but accumulation should show no asset growth from contributions
        accumulation_rows = result.iloc[:2]
        assert all(accumulation_rows['cash_outflow'] == 0)
    
    def test_zero_rate(self):
        """Test with zero interest rate."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.0,
            accumulation_periods=2,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        result = simulator.simulate()
        
        # Should not crash and prices should remain constant
        assert all(result['rate'] == 0.0)
        # First period price should be 1.0, subsequent prices should remain 1.0
        assert result.iloc[0]['price'] == 1.0
        assert result.iloc[1]['price'] == 1.0
    
    def test_single_period_phases(self):
        """Test with single period for each phase."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=1,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        result = simulator.simulate()
        
        assert len(result) == 3
        assert isinstance(result, pd.DataFrame)
    
    def test_high_contribution_amount(self):
        """Test with high contribution amount."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=100_000,  # Reduced from 1_000_000
            rate=0.05,
            accumulation_periods=2,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        result = simulator.simulate()
        
        # Should handle large numbers without overflow
        assert all(result['cash_outflow'].iloc[:2] == 100_000)
        assert result['al_balance'].iloc[1] > 100_000  # Should grow with interest


class TestInterestFactorIntegration:
    """Test integration with InterestFactor for decumulation calculations."""
    
    @patch('fpyjp.core.balance_simulator.InterestFactor')
    def test_capital_recovery_calculation(self, mock_interest_factor):
        """Test that InterestFactor is used correctly for capital recovery."""
        # Setup mock
        mock_factor_instance = Mock()
        mock_factor_instance.calculate_capital_recovery.return_value = 15000
        mock_interest_factor.return_value = mock_factor_instance
        
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=1,
            hold_periods=1,
            decumulation_periods=2,
        )
        
        # Run simulation (which triggers decumulation calculation)
        result = simulator.simulate()
        
        # Verify InterestFactor was called correctly
        assert mock_interest_factor.called
        call_args = mock_interest_factor.call_args
        
        # Check that the correct parameters were passed
        assert call_args[1]['rate'] == 0.05  # decumulation rate
        assert call_args[1]['time_period'] == 2  # decumulation periods
        assert call_args[1]['amount'] > 0  # asset value from hold phase
        
        # Check that calculate_capital_recovery was called
        mock_factor_instance.calculate_capital_recovery.assert_called_once()


class TestDataFrameStructure:
    """Test DataFrame structure and content correctness."""
    
    def test_combined_dataframe_index_continuity(self):
        """Test that combined DataFrame has continuous index."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=3,
            hold_periods=2,
            decumulation_periods=2,
        )
        
        result = simulator.simulate()
        
        # Check index continuity
        expected_index = list(range(7))  # 3 + 2 + 2
        assert list(result.index) == expected_index
        assert result.index.name == 'time_period'
    
    def test_required_columns_present(self):
        """Test that all required columns are present in result."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=1,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        result = simulator.simulate()
        
        required_columns = [
            'price', 'pre_cash_balance', 'pre_al_unit', 'pre_al_balance',
            'cash_outflow', 'capital_cash_inflow_before_tax', 'cash_balance',
            'al_unit', 'al_balance', 'al_book_balance', 'rate'
        ]
        
        for col in required_columns:
            assert col in result.columns, f"Column {col} missing from result"
    
    def test_cash_flow_consistency(self):
        """Test cash flow consistency across phases."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=2,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        result = simulator.simulate()
        
        # Accumulation phase should have outflows
        accumulation_rows = result.iloc[:2]
        assert all(accumulation_rows['cash_outflow'] == 10000)
        assert all(accumulation_rows['capital_cash_inflow_before_tax'] == 0)
        
        # Hold phase should have no flows
        hold_row = result.iloc[2]
        assert hold_row['cash_outflow'] == 0
        assert hold_row['capital_cash_inflow_before_tax'] == 0
        
        # Decumulation phase should have inflows
        decumulation_row = result.iloc[3]
        assert decumulation_row['cash_outflow'] == 0
        assert decumulation_row['capital_cash_inflow_before_tax'] > 0


class TestParameterDefaults:
    """Test parameter defaults and optional settings."""
    
    def test_default_initial_settings(self):
        """Test default values for initial settings."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=1,
            hold_periods=1,
            decumulation_periods=1,
        )
        
        assert simulator.initial_cash_balance == 0.0
        assert simulator.initial_price == 1.0
        assert simulator.cash_inflow_per_unit == 0.0
        assert simulator.income_gain_tax_rate == 0.0
        assert simulator.capital_gain_tax_rate == 0.0
    
    def test_custom_initial_settings(self):
        """Test custom values for initial settings."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=1,
            hold_periods=1,
            decumulation_periods=1,
            initial_cash_balance=5000,
            initial_price=2.0,
            cash_inflow_per_unit=0.03,
            income_gain_tax_rate=0.15,
            capital_gain_tax_rate=0.20,
        )
        
        assert simulator.initial_cash_balance == 5000
        assert simulator.initial_price == 2.0
        assert simulator.cash_inflow_per_unit == 0.03
        assert simulator.income_gain_tax_rate == 0.15
        assert simulator.capital_gain_tax_rate == 0.20


# hold_periods=0のテストケースを追加
class TestHoldPeriodsZero:
    """Test cases for hold_periods=0."""
    
    def test_hold_periods_zero_period_based(self):
        """Test hold_periods=0 with period-based specification."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=2,
            hold_periods=0,  # Zero hold periods
            decumulation_periods=2,
        )
        
        result = simulator.simulate()
        
        # Should have 2 + 0 + 2 = 4 periods
        assert len(result) == 4
        assert isinstance(result, pd.DataFrame)
        
        # Check that indices are continuous (0, 1, 2, 3)
        assert list(result.index) == [0, 1, 2, 3]
        
        # Accumulation phase: periods 0-1
        accumulation_rows = result.iloc[:2]
        assert all(accumulation_rows['cash_outflow'] == 10000)
        
        # Decumulation phase: periods 2-3 (no hold phase)
        decumulation_rows = result.iloc[2:]
        assert all(decumulation_rows['capital_cash_inflow_before_tax'] > 0)
        assert all(decumulation_rows['cash_outflow'] == 0)
    
    def test_hold_periods_zero_timepoint_based(self):
        """Test hold_periods=0 with time-point-based specification."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_end_period=2,
            decumulation_start_period=2,  # Same as accumulation_end → hold_periods=0
            simulation_end_period=4,
        )
        
        # Should calculate periods correctly
        assert simulator.accumulation_periods == 2
        assert simulator.hold_periods == 0
        assert simulator.decumulation_periods == 2
        
        result = simulator.simulate()
        assert len(result) == 4
    
    def test_hold_df_empty_when_hold_periods_zero(self):
        """Test that hold_df is empty when hold_periods=0."""
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=0.05,
            accumulation_periods=1,
            hold_periods=0,
            decumulation_periods=1,
        )
        
        simulator.simulate()
        
        # hold_df should be empty but have correct structure
        assert isinstance(simulator.hold_df, pd.DataFrame)
        assert len(simulator.hold_df) == 0
        
        # But should have correct columns
        expected_columns = [
            'price', 'pre_cash_balance', 'pre_al_unit', 'pre_al_balance',
            'pre_al_book_balance', 'pre_unrealized_gl', 'cash_inflow_per_unit',
            'income_cash_inflow_before_tax', 'income_gain_tax_rate',
            'income_gain_tax', 'income_cash_inflow', 'unit_outflow',
            'capital_cash_inflow_before_tax', 'capital_gain_tax_rate',
            'capital_gain_tax', 'capital_cash_inflow', 'cash_inflow', 'unit_inflow',
            'cash_outflow', 'cash_flow', 'unit_flow', 'cash_balance', 'al_unit',
            'al_balance', 'al_book_balance', 'unrealized_gl', 'rate'
        ]
        
        for col in expected_columns:
            assert col in simulator.hold_df.columns
    
    def test_rate_distribution_with_hold_periods_zero(self):
        """Test rate list distribution when hold_periods=0."""
        rate_list = [0.08, 0.06, 0.04, 0.02]  # 4 rates for 2+0+2 periods
        simulator = LifecycleInvestmentSimulator(
            contribution_amount=10000,
            rate=rate_list,
            accumulation_periods=2,
            hold_periods=0,
            decumulation_periods=2,
        )
        
        assert simulator.accumulation_rate == [0.08, 0.06]
        assert simulator.hold_rate == []  # Empty list for zero hold periods
        assert simulator.decumulation_rate == [0.04, 0.02]

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])