"""
Unit tests for AssetLiabilitySimulator

Tests cover initialization, schema extraction, simulation logic,
tax calculations, and edge cases including zero division protection.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Assuming these imports would be available in the actual project
from fpyjp.schemas.balance import AssetLiabilitySchema
from fpyjp.core.balance_simulator import AssetLiabilitySimulator, TAX_RATE


class TestAssetLiabilitySimulatorInitialization:
    """Test cases for AssetLiabilitySimulator initialization"""
    
    def test_init_with_al_schema(self):
        """Test initialization with AssetLiabilitySchema"""
        # Create mock schema
        schema = Mock(spec=AssetLiabilitySchema)
        schema.balance = 1000.0
        schema.book_balance = 950.0
        schema.unit = 100.0
        schema.price = 10.0
        schema.cashinflow_per_unit = 0.5
        schema.rate = 0.05
        
        simulator = AssetLiabilitySimulator(
            al_schema=schema,
            initial_cash_balance=500.0
        )
        
        assert simulator.al_schema == schema
        assert simulator.initial_cash_balance == 500.0
        assert simulator.capital_cash_inflow_before_tax == [0]
        assert simulator.cash_outflow == [0]
        assert simulator.income_gain_tax_rate == [TAX_RATE]
        assert simulator.capital_gain_tax_rate == [TAX_RATE]
    
    def test_init_with_individual_parameters(self):
        """Test initialization with individual parameters"""
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=5000.0,
            initial_al_book_balance=4800.0,
            initial_price=50.0,
            rate=0.03,
            cash_inflow_per_unit=2.0,
            capital_cash_inflow_before_tax=100.0,
            cash_outflow=200.0,
            income_gain_tax_rate=0.25,
            capital_gain_tax_rate=0.15
        )
        
        assert simulator.initial_cash_balance == 1000.0
        assert simulator.capital_cash_inflow_before_tax == [100.0]
        assert simulator.cash_outflow == [200.0]
        assert simulator.income_gain_tax_rate == [0.25]
        assert simulator.capital_gain_tax_rate == [0.15]
        
        # Check that al_schema was created properly
        assert simulator.al_schema.balance == 5000.0
        assert simulator.al_schema.book_balance == 4800.0
        assert simulator.al_schema.price == 50.0
        assert simulator.al_schema.unit == 100.0  # 5000/50
    
    def test_init_missing_required_parameters(self):
        """Test initialization fails with missing required parameters"""
        
        # Test case 1: Missing initial_al_balance (only initial_price provided)
        with pytest.raises(ValueError, match="initial_al_balance is required when al_schema is not provided"):
            AssetLiabilitySimulator(initial_price=10.0)
        
        # Test case 2: Missing initial_price (only initial_al_balance provided)
        with pytest.raises(ValueError, match="initial_price is required when al_schema is not provided"):
            AssetLiabilitySimulator(initial_al_balance=1000.0)
        
        # Test case 3: Missing both required parameters (completely empty initialization)
        with pytest.raises(ValueError, match="initial_price is required when al_schema is not provided"):
            AssetLiabilitySimulator()
        
        # Test case 4: Missing initial_al_balance with other parameters
        with pytest.raises(ValueError, match="initial_al_balance is required when al_schema is not provided"):
            AssetLiabilitySimulator(
                initial_price=50.0,
                rate=0.02,
                cash_inflow_per_unit=1.0
            )
        
        # Test case 5: Missing initial_price with other parameters
        with pytest.raises(ValueError, match="initial_price is required when al_schema is not provided"):
            AssetLiabilitySimulator(
                initial_al_balance=5000.0,
                rate=0.02,
                cash_inflow_per_unit=1.0
            )
        
        # Test case 6: Verify that providing both required parameters works
        simulator = AssetLiabilitySimulator(
            initial_price=10.0,
            initial_al_balance=1000.0
        )
        assert simulator.al_schema.price == 10.0
        assert simulator.al_schema.balance == 1000.0
        assert simulator.al_schema.unit == 100.0  # 1000/10
        
        # Test case 7: Verify that providing al_schema bypasses the requirement
        schema = Mock(spec=AssetLiabilitySchema)
        schema.balance = 2000.0
        schema.book_balance = 1900.0
        schema.unit = 200.0
        schema.price = 10.0
        schema.cashinflow_per_unit = 0.5
        schema.rate = 0.05
        
        simulator_with_schema = AssetLiabilitySimulator(al_schema=schema)
        assert simulator_with_schema.al_schema == schema
        
    def test_init_with_list_parameters(self):
        """Test initialization with list parameters"""
        simulator = AssetLiabilitySimulator(
            initial_price=10.0,
            initial_al_balance=1000.0,
            capital_cash_inflow_before_tax=[100, 200, 300],
            cash_outflow=[50, 75, 100],
            income_gain_tax_rate=[0.20, 0.25, 0.30]
        )
        
        assert simulator.capital_cash_inflow_before_tax == [100, 200, 300]
        assert simulator.cash_outflow == [50, 75, 100]
        assert simulator.income_gain_tax_rate == [0.20, 0.25, 0.30]


class TestExtractSchemaValues:
    """Test cases for _extract_schema_values method"""
    
    def test_extract_schema_values_basic(self):
        """Test basic schema value extraction"""
        schema = Mock(spec=AssetLiabilitySchema)
        schema.balance = 2000.0
        schema.book_balance = 1900.0
        schema.unit = 200.0
        schema.price = 10.0
        schema.cashinflow_per_unit = 1.0
        schema.rate = 0.04
        
        simulator = AssetLiabilitySimulator(al_schema=schema)
        simulator._extract_schema_values()
        
        assert simulator.initial_al_balance == 2000.0
        assert simulator.initial_al_book_balance == 1900.0
        assert simulator.initial_al_unit == 200.0
        assert simulator.initial_price == 10.0
        assert simulator.cash_inflow_per_unit == [1.0]
        assert simulator.rate == [0.04]
    
    def test_extract_schema_values_list_price(self):
        """Test schema value extraction with list price"""
        schema = Mock(spec=AssetLiabilitySchema)
        schema.balance = 1000.0
        schema.book_balance = None  # Test None book_balance
        schema.unit = 100.0
        schema.price = [10.0, 11.0, 12.0]  # List price
        schema.cashinflow_per_unit = [0.5, 1.0]
        schema.rate = [0.02, 0.03, 0.04]
        
        simulator = AssetLiabilitySimulator(al_schema=schema)
        simulator._extract_schema_values()
        
        assert simulator.initial_price == 10.0  # First value
        assert simulator.initial_al_book_balance == 1000.0  # Falls back to balance
        assert simulator.cash_inflow_per_unit == [0.5, 1.0]
        assert simulator.rate == [0.02, 0.03, 0.04]
    
    def test_extract_schema_values_none_values(self):
        """Test schema value extraction with None values"""
        schema = Mock(spec=AssetLiabilitySchema)
        schema.balance = 1500.0
        schema.book_balance = 1400.0
        schema.unit = 150.0
        schema.price = 10.0
        schema.cashinflow_per_unit = None
        schema.rate = None
        
        simulator = AssetLiabilitySimulator(al_schema=schema)
        simulator._extract_schema_values()
        
        assert simulator.cash_inflow_per_unit == [0.0]
        assert simulator.rate == [0.0]


class TestSimulateMethod:
    """Test cases for simulate method"""
    
    def create_basic_simulator(self):
        """Helper method to create a basic simulator"""
        return AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=5000.0,
            initial_al_book_balance=4500.0,
            initial_price=50.0,
            rate=0.02,
            cash_inflow_per_unit=1.0,
            capital_cash_inflow_before_tax=0.0,
            cash_outflow=0.0,
            income_gain_tax_rate=0.20,
            capital_gain_tax_rate=0.15
        )
    
    def test_simulate_basic_single_period(self):
        """Test basic simulation for single period"""
        simulator = self.create_basic_simulator()
        result = simulator.simulate(1)
        
        assert len(result) == 1
        assert result.index.name == 'time_period'
        
        # Check initial values
        assert result.loc[0, 'price'] == 50.0
        assert result.loc[0, 'pre_cash_balance'] == 1000.0
        assert result.loc[0, 'pre_al_unit'] == 100.0  # 5000/50
        assert result.loc[0, 'pre_al_balance'] == 5000.0
        assert result.loc[0, 'pre_al_book_balance'] == 4500.0
        assert result.loc[0, 'pre_unrealized_gl'] == 500.0  # 5000-4500
        
        # Check income calculations
        assert result.loc[0, 'income_cash_inflow_before_tax'] == 100.0  # 100*1.0
        assert result.loc[0, 'income_gain_tax'] == 20.0  # 100*0.20
        assert result.loc[0, 'income_cash_inflow'] == 80.0  # 100-20
        
        # Check final balances
        assert result.loc[0, 'cash_balance'] == 1080.0  # 1000+80
        assert result.loc[0, 'al_unit'] == 100.0
        assert result.loc[0, 'al_balance'] == 5000.0
        assert result.loc[0, 'al_book_balance'] == 4500.0
    
    def test_simulate_multiple_periods(self):
        """Test simulation for multiple periods"""
        simulator = self.create_basic_simulator()
        result = simulator.simulate(3)
        
        assert len(result) == 3
        
        # Check price growth over periods
        assert result.loc[0, 'price'] == 50.0
        assert result.loc[1, 'price'] == pytest.approx(51.0)  # 50*1.02
        assert result.loc[2, 'price'] == pytest.approx(52.02)  # 51*1.02
        
        # Period 0 - Initial state
        assert result.loc[0, 'pre_cash_balance'] == 1000.0  # Initial cash balance
        assert result.loc[0, 'pre_al_unit'] == 100.0  # 5000/50
        assert result.loc[0, 'pre_al_balance'] == 5000.0  # Initial AL balance
        assert result.loc[0, 'pre_al_book_balance'] == 4500.0  # Initial book balance
        assert result.loc[0, 'pre_unrealized_gl'] == 500.0  # 5000-4500
        
        # Period 0 - Income calculations
        assert result.loc[0, 'cash_inflow_per_unit'] == 1.0
        assert result.loc[0, 'income_cash_inflow_before_tax'] == 100.0  # 100*1.0
        assert result.loc[0, 'income_gain_tax'] == 20.0  # 100*0.20
        assert result.loc[0, 'income_cash_inflow'] == 80.0  # 100-20
        
        # Period 0 - Capital transactions (none in this case)
        assert result.loc[0, 'capital_cash_inflow_before_tax'] == 0.0
        assert result.loc[0, 'unit_outflow'] == 0.0
        assert result.loc[0, 'capital_gain_tax'] == 0.0
        assert result.loc[0, 'capital_cash_inflow'] == 0.0
        assert result.loc[0, 'cash_outflow'] == 0.0
        assert result.loc[0, 'unit_inflow'] == 0.0
        
        # Period 0 - Final calculations
        assert result.loc[0, 'cash_flow'] == 80.0  # income_cash_inflow + capital_cash_inflow - cash_outflow
        assert result.loc[0, 'cash_balance'] == 1080.0  # 1000 + 80
        assert result.loc[0, 'al_unit'] == 100.0  # No unit changes
        assert result.loc[0, 'al_balance'] == pytest.approx(5000.0)  # 100 * 50 (updated price)
        assert result.loc[0, 'al_book_balance'] == 4500.0  # No book balance changes
        assert result.loc[0, 'unrealized_gl'] == pytest.approx(500.0)  # 5000-4500
        
        # Period 1 - Starting with previous period's ending values
        assert result.loc[1, 'pre_cash_balance'] == 1080.0  # Previous period's cash_balance
        assert result.loc[1, 'pre_al_unit'] == 100.0  # Same units
        assert result.loc[1, 'pre_al_balance'] == pytest.approx(5100.0)  # 100 * 51
        assert result.loc[1, 'pre_al_book_balance'] == 4500.0  # Book balance unchanged
        assert result.loc[1, 'pre_unrealized_gl'] == pytest.approx(600.0)  # 5100-4500
        
        # Period 1 - Income calculations
        assert result.loc[1, 'cash_inflow_per_unit'] == 1.0
        assert result.loc[1, 'income_cash_inflow_before_tax'] == 100.0  # 100*1.0
        assert result.loc[1, 'income_gain_tax'] == 20.0  # 100*0.20
        assert result.loc[1, 'income_cash_inflow'] == 80.0  # 100-20
        
        # Period 1 - Capital transactions (none in this case)
        assert result.loc[1, 'capital_cash_inflow_before_tax'] == 0.0
        assert result.loc[1, 'unit_outflow'] == 0.0
        assert result.loc[1, 'capital_gain_tax'] == 0.0
        assert result.loc[1, 'capital_cash_inflow'] == 0.0
        assert result.loc[1, 'cash_outflow'] == 0.0
        assert result.loc[1, 'unit_inflow'] == 0.0
        
        # Period 1 - Final calculations
        assert result.loc[1, 'cash_flow'] == 80.0  # Same as period 0
        assert result.loc[1, 'cash_balance'] == 1160.0  # 1080 + 80
        assert result.loc[1, 'al_unit'] == 100.0  # No unit changes
        assert result.loc[1, 'al_balance'] == pytest.approx(5100.0)  # 100 * 51
        assert result.loc[1, 'al_book_balance'] == 4500.0  # No book balance changes
        assert result.loc[1, 'unrealized_gl'] == pytest.approx(600.0)  # 5100-4500
        
        # Period 2 - Starting with previous period's ending values
        assert result.loc[2, 'pre_cash_balance'] == 1160.0  # Previous period's cash_balance
        assert result.loc[2, 'pre_al_unit'] == 100.0  # Same units
        assert result.loc[2, 'pre_al_balance'] == pytest.approx(5202.0)  # 100 * 52.02
        assert result.loc[2, 'pre_al_book_balance'] == 4500.0  # Book balance unchanged
        assert result.loc[2, 'pre_unrealized_gl'] == pytest.approx(702.0)  # 5202-4500
        
        # Period 2 - Income calculations
        assert result.loc[2, 'cash_inflow_per_unit'] == 1.0
        assert result.loc[2, 'income_cash_inflow_before_tax'] == 100.0  # 100*1.0
        assert result.loc[2, 'income_gain_tax'] == 20.0  # 100*0.20
        assert result.loc[2, 'income_cash_inflow'] == 80.0  # 100-20
        
        # Period 2 - Capital transactions (none in this case)
        assert result.loc[2, 'capital_cash_inflow_before_tax'] == 0.0
        assert result.loc[2, 'unit_outflow'] == 0.0
        assert result.loc[2, 'capital_gain_tax'] == 0.0
        assert result.loc[2, 'capital_cash_inflow'] == 0.0
        assert result.loc[2, 'cash_outflow'] == 0.0
        assert result.loc[2, 'unit_inflow'] == 0.0
        
        # Period 2 - Final calculations
        assert result.loc[2, 'cash_flow'] == 80.0  # Same as previous periods
        assert result.loc[2, 'cash_balance'] == 1240.0  # 1160 + 80
        assert result.loc[2, 'al_unit'] == 100.0  # No unit changes
        assert result.loc[2, 'al_balance'] == pytest.approx(5202.0)  # 100 * 52.02
        assert result.loc[2, 'al_book_balance'] == 4500.0  # No book balance changes
        assert result.loc[2, 'unrealized_gl'] == pytest.approx(702.0)  # 5202-4500
        
        # Summary checks across all periods
        # Cash accumulation should be consistent
        cash_balances = [1080.0, 1160.0, 1240.0]
        for i, expected_cash in enumerate(cash_balances):
            assert result.loc[i, 'cash_balance'] == pytest.approx(expected_cash)
        
        # Cash flow should be consistent across periods
        for i in range(3):
            assert result.loc[i, 'cash_flow'] == 80.0
        
        # AL units should remain constant (no buying/selling)
        for i in range(3):
            assert result.loc[i, 'al_unit'] == 100.0
            assert result.loc[i, 'pre_al_unit'] == 100.0
        
        # Book balance should remain constant (no new purchases)
        for i in range(3):
            assert result.loc[i, 'al_book_balance'] == 4500.0
            assert result.loc[i, 'pre_al_book_balance'] == 4500.0
    
    def test_simulate_with_capital_outflow(self):
        """Test simulation with capital cash inflow (unit sales)"""
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=5000.0,
            initial_al_book_balance=4000.0,
            initial_price=50.0,
            rate=0.0,
            cash_inflow_per_unit=0.0,
            capital_cash_inflow_before_tax=1000.0,  # Sell units
            cash_outflow=0.0,
            income_gain_tax_rate=0.20,
            capital_gain_tax_rate=0.15
        )
        
        result = simulator.simulate(1)
        
        # Check unit outflow
        assert result.loc[0, 'unit_outflow'] == 20.0  # 1000/50
        
        # Check capital gain tax calculation
        # avg_book_price = 4000/100 = 40
        # gain_per_unit = 50-40 = 10
        # total_gain = 10*20 = 200
        # tax = 200*0.15 = 30
        assert result.loc[0, 'capital_gain_tax'] == 30.0
        assert result.loc[0, 'capital_cash_inflow'] == 970.0  # 1000-30
        
        # Check unit reduction
        assert result.loc[0, 'al_unit'] == 80.0  # 100-20
    
    def test_simulate_with_cash_outflow(self):
        """Test simulation with cash outflow (unit purchases)"""
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=2000.0,
            initial_al_balance=5000.0,
            initial_al_book_balance=4000.0,
            initial_price=50.0,
            rate=0.0,
            cash_inflow_per_unit=0.0,
            capital_cash_inflow_before_tax=0.0,
            cash_outflow=500.0,  # Buy units
            income_gain_tax_rate=0.20,
            capital_gain_tax_rate=0.15
        )
        
        result = simulator.simulate(1)
        
        # Check unit inflow
        assert result.loc[0, 'unit_inflow'] == 10.0  # 500/50
        assert result.loc[0, 'al_unit'] == 110.0  # 100+10
        assert result.loc[0, 'al_book_balance'] == 4500.0  # 4000+500
        assert result.loc[0, 'cash_balance'] == 1500.0  # 2000-500
    
    def test_simulate_zero_division_protection(self):
        """Test zero division protection in various scenarios"""
        # Test with zero price
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=0.0,
            initial_al_book_balance=0.0,
            initial_price=0.001,  # Very small price
            rate=0.0,
            cash_inflow_per_unit=0.0,
            capital_cash_inflow_before_tax=100.0,
            cash_outflow=100.0,
            income_gain_tax_rate=0.20,
            capital_gain_tax_rate=0.15
        )
        
        result = simulator.simulate(1)
        
        # Should not raise division by zero errors
        assert not pd.isna(result.loc[0, 'unit_outflow'])
        assert not pd.isna(result.loc[0, 'unit_inflow'])
        assert not pd.isna(result.loc[0, 'capital_gain_tax'])
    
    def test_simulate_negative_income_tax(self):
        """Test that negative income doesn't result in negative tax"""
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=5000.0,
            initial_al_book_balance=4000.0,
            initial_price=50.0,
            rate=0.0,
            cash_inflow_per_unit=-2.0,  # Negative cash inflow
            capital_cash_inflow_before_tax=0.0,
            cash_outflow=0.0,
            income_gain_tax_rate=0.20,
            capital_gain_tax_rate=0.15
        )
        
        result = simulator.simulate(1)
        
        # Income tax should be 0 for negative income
        assert result.loc[0, 'income_gain_tax'] == 0
        assert result.loc[0, 'income_cash_inflow'] == -200.0  # -200-0
    
    def test_simulate_with_list_parameters(self):
        """Test simulation with varying parameters across periods"""
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=5000.0,
            initial_al_book_balance=4000.0,
            initial_price=50.0,
            rate=[0.02, 0.03, 0.01],
            cash_inflow_per_unit=[1.0, 1.5, 0.5],
            capital_cash_inflow_before_tax=[0, 500, 0],
            cash_outflow=[0, 0, 300],
            income_gain_tax_rate=[0.20, 0.25, 0.15],
            capital_gain_tax_rate=[0.15, 0.20, 0.10]
        )
        
        result = simulator.simulate(3)
        
        # Check that different rates are applied in different periods
        assert result.loc[0, 'rate'] == 0.02
        assert result.loc[1, 'rate'] == 0.03
        assert result.loc[2, 'rate'] == 0.01
        
        assert result.loc[0, 'cash_inflow_per_unit'] == 1.0
        assert result.loc[1, 'cash_inflow_per_unit'] == 1.5
        assert result.loc[2, 'cash_inflow_per_unit'] == 0.5
        
        assert result.loc[0, 'income_gain_tax_rate'] == 0.20
        assert result.loc[1, 'income_gain_tax_rate'] == 0.25
        assert result.loc[2, 'income_gain_tax_rate'] == 0.15
    
    def test_simulate_capital_loss_no_tax(self):
        """Test that capital losses don't result in negative tax"""
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=5000.0,
            initial_al_book_balance=6000.0,  # Book value higher than market
            initial_price=50.0,
            rate=0.0,
            cash_inflow_per_unit=0.0,
            capital_cash_inflow_before_tax=1000.0,  # Sell at loss
            cash_outflow=0.0,
            income_gain_tax_rate=0.20,
            capital_gain_tax_rate=0.15
        )
        
        result = simulator.simulate(1)
        
        # Should have no capital gains tax for capital loss
        assert result.loc[0, 'capital_gain_tax'] == 0.0
        assert result.loc[0, 'capital_cash_inflow'] == 1000.0
    
    @patch('fpyjp.core.balance_simulator.safe_divide')
    @patch('fpyjp.core.balance_simulator.get_padded_value_at_period')
    @patch('fpyjp.core.balance_simulator.ensure_list')
    def test_simulate_uses_utility_functions(self, mock_ensure_list, mock_get_padded, mock_safe_divide):
        """Test that simulation uses utility functions properly"""
        # Setup mocks to return expected values
        mock_ensure_list.side_effect = lambda x: [x] if not isinstance(x, list) else x
        mock_get_padded.side_effect = lambda lst, period, pad_mode="zero": lst[0] if len(lst) > 0 else 0
        mock_safe_divide.side_effect = lambda x, y, default=0.0: x/y if y != 0 else default
        
        # Create simulator - this will call ensure_list during initialization
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1000.0,
            initial_al_balance=5000.0,
            initial_price=50.0,
            rate=0.02,
            cash_inflow_per_unit=1.0,
            capital_cash_inflow_before_tax=100.0,
            cash_outflow=200.0
        )
        
        # Verify ensure_list was called during initialization
        assert mock_ensure_list.called, "ensure_list should be called during initialization"
        
        # Reset call counts after initialization to test simulate() separately
        mock_ensure_list.reset_mock()
        mock_get_padded.reset_mock() 
        mock_safe_divide.reset_mock()
        
        # Run simulation - this will call utility functions
        result = simulator.simulate(1)
        
        # Verify _extract_schema_values() calls ensure_list during simulate()
        # since it's called at the beginning of simulate()
        assert mock_ensure_list.called, "ensure_list should be called during simulate() via _extract_schema_values()"
        
        # _extract_schema_values() calls ensure_list for cashinflow_per_unit and rate
        assert mock_ensure_list.call_count >= 2, f"Expected at least 2 calls to ensure_list, got {mock_ensure_list.call_count}"
        
        # get_padded_value_at_period should be called multiple times (6 times per period)
        # cash_inflow_per_unit, capital_cash_inflow_before_tax, cash_outflow, rate, income_gain_tax_rate, capital_gain_tax_rate
        assert mock_get_padded.called, "get_padded_value_at_period should be called during simulate()"
        assert mock_get_padded.call_count == 6, f"Expected 6 calls to get_padded_value_at_period, got {mock_get_padded.call_count}"
        
        # safe_divide should be called multiple times for various calculations
        assert mock_safe_divide.called, "safe_divide should be called during simulate()"
        assert mock_safe_divide.call_count >= 3, f"Expected at least 3 calls to safe_divide, got {mock_safe_divide.call_count}"


class TestSimulateEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_simulate_zero_periods(self):
        """Test simulation with zero periods"""
        simulator = AssetLiabilitySimulator(
            initial_price=10.0,
            initial_al_balance=1000.0
        )
        
        result = simulator.simulate(0)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_simulate_large_number_of_periods(self):
        """Test simulation with large number of periods"""
        simulator = AssetLiabilitySimulator(
            initial_price=10.0,
            initial_al_balance=1000.0,
            rate=0.01
        )
        
        result = simulator.simulate(1000)
        assert len(result) == 1000
        # Check that simulation completes without errors
        assert not result.isnull().any().any()
    
    def test_simulate_extreme_values(self):
        """Test simulation with extreme parameter values"""
        simulator = AssetLiabilitySimulator(
            initial_cash_balance=1e6,
            initial_al_balance=1e9,
            initial_al_book_balance=1e8,
            initial_price=1e3,
            rate=0.5,  # 50% growth rate
            cash_inflow_per_unit=100.0,
            capital_cash_inflow_before_tax=1e6,
            cash_outflow=1e6,
            income_gain_tax_rate=0.50,
            capital_gain_tax_rate=0.40
        )
        
        result = simulator.simulate(5)
        
        # Check that extreme values don't break the simulation
        assert len(result) == 5
        assert not result.isnull().any().any()
        assert all(result['price'] > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])