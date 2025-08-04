"""
Unit tests for AssetLiabilitySchema using pytest.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from fpyjp.schemas.balance import AssetLiabilitySchema  # Import your module


class TestAssetLiabilityBasicFunctionality:
    """Test basic functionality and value calculations."""
    
    def test_calculate_balance_from_price_and_unit(self):
        """Test calculating balance when price and unit are provided."""
        asset = AssetLiabilitySchema(price=100.0, unit=5.0)
        assert asset.balance == 500.0
        assert asset.book_balance == 500.0  # Should default to balance
    
    def test_calculate_price_from_balance_and_unit(self):
        """Test calculating price when balance and unit are provided."""
        asset = AssetLiabilitySchema(balance=1000.0, unit=10.0)
        assert asset.price == 100.0
        assert asset.book_balance == 1000.0
    
    def test_calculate_unit_from_price_and_balance(self):
        """Test calculating unit when price and balance are provided."""
        asset = AssetLiabilitySchema(price=50.0, balance=300.0)
        assert asset.unit == 6.0
        assert asset.book_balance == 300.0
    
    def test_all_three_values_provided_valid(self):
        """Test when all three values are provided and consistent."""
        asset = AssetLiabilitySchema(price=20.0, unit=15.0, balance=300.0)
        assert asset.price == 20.0
        assert asset.unit == 15.0
        assert asset.balance == 300.0
        assert asset.book_balance == 300.0
    
    def test_book_balance_explicit_value(self):
        """Test when book_balance is explicitly provided."""
        asset = AssetLiabilitySchema(
            price=100.0, 
            unit=5.0, 
            book_balance=450.0
        )
        assert asset.balance == 500.0
        assert asset.book_balance == 450.0
    
    def test_default_values(self):
        """Test default values for optional fields."""
        asset = AssetLiabilitySchema(price=100.0, unit=5.0)
        assert asset.cashinflow_per_unit == 0.0
        assert asset.rate == 0.0


class TestAssetLiabilityValidationErrors:
    """Test validation error cases."""
    
    def test_insufficient_parameters(self):
        """Test error when less than 2 of price, unit, balance are provided."""
        with pytest.raises(ValidationError, match="At least two of price, unit, and balance must be provided"):
            AssetLiabilitySchema(price=100.0)
        
        with pytest.raises(ValidationError, match="At least two of price, unit, and balance must be provided"):
            AssetLiabilitySchema(unit=5.0)
        
        with pytest.raises(ValidationError, match="At least two of price, unit, and balance must be provided"):
            AssetLiabilitySchema(balance=500.0)
        
        with pytest.raises(ValidationError, match="At least two of price, unit, and balance must be provided"):
            AssetLiabilitySchema()
    
    def test_inconsistent_values(self):
        """Test error when all three values are provided but inconsistent."""
        with pytest.raises(ValidationError, match="price \\* unit must equal balance"):
            AssetLiabilitySchema(price=100.0, unit=5.0, balance=600.0)  # Should be 500
    
    def test_negative_unit(self):
        """Test error when unit is negative."""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            AssetLiabilitySchema(price=100.0, unit=-5.0)
    
    def test_rate_less_than_or_equal_to_minus_one(self):
        """Test error when rate is <= -1."""
        with pytest.raises(ValidationError, match="rate must be greater than -1"):
            AssetLiabilitySchema(price=100.0, unit=5.0, rate=-1.0)
        
        with pytest.raises(ValidationError, match="rate must be greater than -1"):
            AssetLiabilitySchema(price=100.0, unit=5.0, rate=-1.5)
    
    def test_rate_list_with_invalid_values(self):
        """Test error when rate list contains values <= -1."""
        with pytest.raises(ValidationError, match="rate\\[1\\] must be greater than -1"):
            AssetLiabilitySchema(price=100.0, unit=5.0, rate=[0.1, -1.0, 0.05])
    
    def test_division_by_zero_cases(self):
        """Test error cases involving division by zero."""
        with pytest.raises(ValidationError, match="Cannot calculate price when unit is 0"):
            AssetLiabilitySchema(balance=500.0, unit=0.0)
        
        with pytest.raises(ValidationError, match="Cannot calculate unit when price is 0"):
            AssetLiabilitySchema(price=0.0, balance=500.0)


class TestAssetLiabilityListProcessing:
    """Test complex list processing for price and rate."""
    
    def test_single_element_list_normalization(self):
        """Test that single-element lists are normalized to scalars."""
        asset = AssetLiabilitySchema(price=[100.0], unit=5.0, rate=[0.1])
        assert asset.price == 100.0  # Should be scalar, not list
        assert asset.rate == 0.1     # Should be scalar, not list
    
    def test_empty_list_error(self):
        """Test error when empty lists are provided."""
        with pytest.raises(ValidationError, match="list cannot be empty"):
            AssetLiabilitySchema(price=[], unit=5.0)
        
        with pytest.raises(ValidationError, match="list cannot be empty"):
            AssetLiabilitySchema(price=100.0, unit=5.0, rate=[])
    
    def test_generate_price_from_rate_list(self):
        """Test generating price list from scalar price and rate list."""
        asset = AssetLiabilitySchema(
            price=100.0, 
            unit=5.0, 
            rate=[0.1, 0.05, -0.02]
        )
        expected_prices = [100.0, 110.0, 115.5, 113.19]
        assert len(asset.price) == 4
        for i, expected in enumerate(expected_prices):
            assert abs(asset.price[i] - expected) < 1e-10
    
    def test_generate_rate_from_price_list(self):
        """Test calculating rate list from price list."""
        asset = AssetLiabilitySchema(
            price=[100.0, 110.0, 121.0], 
            unit=5.0
        )
        expected_rates = [0.1, 0.1]  # 10% growth each period
        assert len(asset.rate) == 2
        for i, expected in enumerate(expected_rates):
            assert abs(asset.rate[i] - expected) < 1e-10
    
    def test_consistent_price_and_rate_lists(self):
        """Test when both price and rate lists are provided and consistent."""
        price_list = [100.0, 105.0, 110.25]
        rate_list = [0.05, 0.05]  # 5% growth each period
        
        asset = AssetLiabilitySchema(
            price=price_list,
            unit=5.0,
            rate=rate_list
        )
        assert asset.price == price_list
        assert asset.rate == rate_list
    
    def test_inconsistent_price_and_rate_lists(self):
        """Test error when price and rate lists are inconsistent."""
        with pytest.raises(ValidationError, match="Price growth relationship violated"):
            AssetLiabilitySchema(
                price=[100.0, 110.0, 130.0],  # [100.0, 110.0, 121.0]
                unit=5.0,
                rate=[0.1, 0.1]  # Both 10%
            )
    
    def test_scalar_rate_vs_calculated_rate_consistency(self):
        """Test consistency between scalar rate and calculated rate from prices."""
        # Consistent case
        asset = AssetLiabilitySchema(
            price=[100.0, 110.0],
            unit=5.0,
            rate=0.1  # 10% matches the price growth
        )
        assert asset.rate == [0.1]  # Should be converted to list
        
        # Inconsistent case
        with pytest.raises(ValidationError, match="Existing rate.*does not match calculated rate"):
            AssetLiabilitySchema(
                price=[100.0, 110.0],
                unit=5.0,
                rate=0.05  # 5% doesn't match the 10% price growth
            )
    
    def test_price_list_too_short_for_rate_calculation(self):
        """Test error when price list is too short to calculate rates."""
        with pytest.raises(ValidationError, match="price list must have at least 2 elements"):
            AssetLiabilitySchema(
                price=[100.0],  # Only one element
                unit=5.0,
                rate=0.1
            )
    
    def test_zero_price_in_rate_calculation(self):
        """Test error when zero price prevents rate calculation."""
        with pytest.raises(ValidationError, match="Cannot calculate rate when price\\[0\\] is 0"):
            AssetLiabilitySchema(
                price=[0.0, 110.0],
                unit=5.0
            )


class TestAssetLiabilityEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_values(self):
        """Test with very small but positive values."""
        asset = AssetLiabilitySchema(price=1e-6, unit=1e6)
        assert abs(asset.balance - 1.0) < 1e-10
    
    def test_very_large_values(self):
        """Test with very large values."""
        asset = AssetLiabilitySchema(price=1e6, unit=1e6)
        assert asset.balance == 1e12
    
    def test_precision_tolerance(self):
        """Test that small floating-point errors are tolerated."""
        # This should not raise an error due to floating-point precision
        asset = AssetLiabilitySchema(
            price=1.0/3.0, 
            unit=3.0, 
            balance=1.0000000000001  # Slightly off due to floating-point
        )
        assert abs(asset.balance - 1.0000000000001) < 1e-10
    
    def test_rate_boundary_values(self):
        """Test rate values near the boundary (-1)."""
        # Just above -1 should work
        asset = AssetLiabilitySchema(price=100.0, unit=5.0, rate=-0.9999)
        assert asset.rate == -0.9999
        
        # Very close to -1 but still valid
        asset = AssetLiabilitySchema(price=100.0, unit=5.0, rate=-0.999999999)
        assert asset.rate == -0.999999999
    
    def test_zero_cashinflow_per_unit(self):
        """Test that cashinflow_per_unit defaults to 0.0."""
        asset = AssetLiabilitySchema(price=100.0, unit=5.0)
        assert asset.cashinflow_per_unit == 0.0
    
    def test_cashinflow_per_unit_list(self):
        """Test cashinflow_per_unit as a list."""
        asset = AssetLiabilitySchema(
            price=100.0, 
            unit=5.0, 
            cashinflow_per_unit=[10.0, 12.0, 8.0]
        )
        assert asset.cashinflow_per_unit == [10.0, 12.0, 8.0]


class TestAssetLiabilityNumpyIntegration:
    """Test numpy array integration and vectorized operations."""
    
    def test_numpy_array_validation(self):
        """Test that numpy arrays work correctly in validation."""
        price_array = np.array([100.0, 105.0, 110.25, 115.7625])
        rate_array = np.array([0.05, 0.05, 0.05])
        
        asset = AssetLiabilitySchema(
            price=price_array.tolist(),
            unit=5.0,
            rate=rate_array.tolist()
        )
        
        # Verify the relationship holds
        assert len(asset.price) == 4
        assert len(asset.rate) == 3
    
    def test_large_list_performance(self):
        """Test performance with larger lists (basic smoke test)."""
        # Generate a longer sequence
        initial_price = 100.0
        rates = [0.01] * 100  # 1% growth for 100 periods
        
        asset = AssetLiabilitySchema(
            price=initial_price,
            unit=5.0,
            rate=rates
        )
        
        # Should have 101 prices (initial + 100 growth periods)
        assert len(asset.price) == 101
        assert len(asset.rate) == 100
        
        # Verify final price is approximately correct
        expected_final = initial_price * (1.01 ** 100)
        assert abs(asset.price[-1] - expected_final) < 1e-8


# Integration tests
class TestAssetLiabilityIntegration:
    """Integration tests combining multiple features."""
    
    def test_complex_scenario_bond_like(self):
        """Test a bond-like scenario with price appreciation and cash flows."""
        asset = AssetLiabilitySchema(
            price=[95.0, 97.0, 100.0],  # Price appreciation to par
            unit=1000.0,  # 1000 bonds
            cashinflow_per_unit=[2.5, 2.5],  # Semi-annual coupon
            book_balance=94000.0  # Purchased at discount
        )
        
        assert asset.balance == 95000.0  # Current market value
        assert asset.book_balance == 94000.0  # Purchase price
        assert len(asset.rate) == 2  # Two rate periods
        
        # Verify rate calculations
        expected_rate_1 = (97.0 / 95.0) - 1
        expected_rate_2 = (100.0 / 97.0) - 1
        
        assert abs(asset.rate[0] - expected_rate_1) < 1e-10
        assert abs(asset.rate[1] - expected_rate_2) < 1e-10
    
    def test_complex_scenario_stock_like(self):
        """Test a stock-like scenario with dividend growth."""
        initial_price = 50.0
        growth_rates = [0.08, 0.06, 0.05]  # Declining growth rates
        
        asset = AssetLiabilitySchema(
            price=initial_price,
            unit=200.0,  # 200 shares
            rate=growth_rates,
            cashinflow_per_unit=1.50  # Annual dividend
        )
        
        assert asset.balance == 10000.0  # 200 * 50
        assert len(asset.price) == 4  # Initial + 3 growth periods
        
        # Verify price progression
        expected_prices = [50.0]
        current_price = 50.0
        for rate in growth_rates:
            current_price *= (1 + rate)
            expected_prices.append(current_price)
        
        for i, expected in enumerate(expected_prices):
            assert abs(asset.price[i] - expected) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])