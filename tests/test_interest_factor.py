import pytest
from pydantic import ValidationError

from interest_factor import InterestFactor


class TestInterestFactorValidation:
    """Test class for InterestFactor validation and instantiation."""
    
    def test_valid_instance_creation(self):
        """Test creating a valid InterestFactor instance with all parameters."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=1000.0)
        assert factor.rate == 0.05
        assert factor.time_period == 10
        assert factor.amount == 1000.0
    
    def test_default_amount_value(self):
        """Test that amount defaults to 1.0 when not provided."""
        factor = InterestFactor(rate=0.05, time_period=10)
        assert factor.amount == 1.0
    
    @pytest.mark.parametrize("invalid_rate", [
        -1.0,    # Exactly -1 (should fail)
        -1.1,    # Less than -1
        -2.0,    # Much less than -1
    ])
    def test_invalid_rate_validation(self, invalid_rate):
        """Test that rates <= -1 raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterestFactor(rate=invalid_rate, time_period=10)
        assert "Rate must be greater than -1" in str(exc_info.value)
    
    @pytest.mark.parametrize("valid_rate", [
        -0.99,   # Just above -1
        -0.5,    # Negative but valid
        0.0,     # Zero rate
        0.05,    # Positive rate
        1.0,     # 100% rate
        2.0,     # 200% rate
    ])
    def test_valid_rate_values(self, valid_rate):
        """Test that valid rates are accepted."""
        factor = InterestFactor(rate=valid_rate, time_period=10)
        assert factor.rate == valid_rate
    
    @pytest.mark.parametrize("invalid_time_period", [
        0,       # Zero periods
        -1,      # Negative periods
        -10,     # Much negative
    ])
    def test_invalid_time_period_validation(self, invalid_time_period):
        """Test that time_period < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            InterestFactor(rate=0.05, time_period=invalid_time_period)
    
    @pytest.mark.parametrize("valid_time_period", [
        1,       # Minimum valid value
        5,       # Normal value
        100,     # Large value
    ])
    def test_valid_time_period_values(self, valid_time_period):
        """Test that valid time periods are accepted."""
        factor = InterestFactor(rate=0.05, time_period=valid_time_period)
        assert factor.time_period == valid_time_period


class TestFutureValueCalculations:
    """Test class for future value related calculations."""
    
    def test_future_value_factor_normal_case(self):
        """Test future value factor calculation with standard 5% rate for 10 periods."""
        factor = InterestFactor(rate=0.05, time_period=10)
        result = factor.future_value_factor()
        expected = (1 + 0.05) ** 10  # ≈ 1.6289
        assert abs(result - expected) < 1e-10
    
    def test_future_value_factor_zero_rate(self):
        """Test future value factor with zero interest rate."""
        factor = InterestFactor(rate=0.0, time_period=10)
        result = factor.future_value_factor()
        assert result == 1.0
    
    def test_future_value_factor_one_period(self):
        """Test future value factor with single time period."""
        factor = InterestFactor(rate=0.05, time_period=1)
        result = factor.future_value_factor()
        assert result == 1.05
    
    def test_calculate_future_value_uses_factor(self):
        """Test that calculate_future_value uses future_value_factor method."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        factor_result = factor.future_value_factor()
        calculate_result = factor.calculate_future_value()
        expected = factor_result * 1000
        assert abs(calculate_result - expected) < 1e-10
    
    def test_calculate_future_value_with_different_amounts(self):
        """Test calculate_future_value with various amount values."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=500)
        result = factor.calculate_future_value()
        factor_value = (1.05) ** 10
        expected = factor_value * 500
        assert abs(result - expected) < 1e-10


class TestPresentValueCalculations:
    """Test class for present value related calculations."""
    
    def test_present_value_factor_normal_case(self):
        """Test present value factor calculation with standard parameters."""
        factor = InterestFactor(rate=0.05, time_period=10)
        result = factor.present_value_factor()
        expected = 1 / ((1 + 0.05) ** 10)  # ≈ 0.6139
        assert abs(result - expected) < 1e-10
    
    def test_present_value_factor_zero_rate(self):
        """Test present value factor with zero interest rate."""
        factor = InterestFactor(rate=0.0, time_period=10)
        result = factor.present_value_factor()
        assert result == 1.0
    
    def test_present_value_reciprocal_relationship(self):
        """Test that present value factor is reciprocal of future value factor."""
        factor = InterestFactor(rate=0.05, time_period=10)
        pv_factor = factor.present_value_factor()
        fv_factor = factor.future_value_factor()
        assert abs(pv_factor * fv_factor - 1.0) < 1e-10
    
    def test_calculate_present_value_uses_factor(self):
        """Test that calculate_present_value uses present_value_factor method."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        factor_result = factor.present_value_factor()
        calculate_result = factor.calculate_present_value()
        expected = factor_result * 1000
        assert abs(calculate_result - expected) < 1e-10


class TestFutureValueOfAnnuityCalculations:
    """Test class for future value of annuity related calculations."""
    
    def test_future_value_of_annuity_factor_normal_case(self):
        """Test future value of annuity factor with standard parameters."""
        factor = InterestFactor(rate=0.05, time_period=10)
        result = factor.future_value_of_annuity_factor()
        expected = ((1.05 ** 10) - 1) / 0.05  # ≈ 12.5779
        assert abs(result - expected) < 1e-10
    
    def test_future_value_of_annuity_factor_zero_rate(self):
        """Test future value of annuity factor with zero interest rate."""
        factor = InterestFactor(rate=0.0, time_period=10)
        result = factor.future_value_of_annuity_factor()
        assert result == 10.0
    
    def test_future_value_of_annuity_factor_single_period(self):
        """Test future value of annuity factor with single time period."""
        factor = InterestFactor(rate=0.05, time_period=1)
        result = factor.future_value_of_annuity_factor()
        assert result == 1.0
    
    def test_calculate_future_value_of_annuity_uses_factor(self):
        """Test that calculate_future_value_of_annuity uses the factor method."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=100)
        factor_result = factor.future_value_of_annuity_factor()
        calculate_result = factor.calculate_future_value_of_annuity()
        expected = factor_result * 100
        assert abs(calculate_result - expected) < 1e-10
    
    def test_future_value_of_annuity_with_different_amounts(self):
        """Test calculate_future_value_of_annuity with various amount values."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=200)
        result = factor.calculate_future_value_of_annuity()
        factor_value = ((1.05 ** 10) - 1) / 0.05
        expected = factor_value * 200
        assert abs(result - expected) < 1e-10


class TestSinkingFundCalculations:
    """Test class for sinking fund related calculations."""
    
    def test_sinking_fund_factor_normal_case(self):
        """Test sinking fund factor with standard parameters."""
        factor = InterestFactor(rate=0.05, time_period=10)
        result = factor.sinking_fund_factor()
        expected = 0.05 / ((1.05 ** 10) - 1)  # ≈ 0.0795
        assert abs(result - expected) < 1e-10
    
    def test_sinking_fund_factor_zero_rate(self):
        """Test sinking fund factor with zero interest rate."""
        factor = InterestFactor(rate=0.0, time_period=10)
        result = factor.sinking_fund_factor()
        assert result == 0.1  # 1/10
    
    def test_sinking_fund_reciprocal_relationship(self):
        """Test that sinking fund factor is reciprocal of future value of annuity factor."""
        factor = InterestFactor(rate=0.05, time_period=10)
        sf_factor = factor.sinking_fund_factor()
        fva_factor = factor.future_value_of_annuity_factor()
        assert abs(sf_factor * fva_factor - 1.0) < 1e-10
    
    def test_calculate_sinking_fund_uses_factor(self):
        """Test that calculate_sinking_fund uses sinking_fund_factor method."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        factor_result = factor.sinking_fund_factor()
        calculate_result = factor.calculate_sinking_fund()
        expected = factor_result * 1000
        assert abs(calculate_result - expected) < 1e-10


class TestCapitalRecoveryCalculations:
    """Test class for capital recovery related calculations."""
    
    def test_capital_recovery_factor_normal_case(self):
        """Test capital recovery factor with standard parameters."""
        factor = InterestFactor(rate=0.05, time_period=10)
        result = factor.capital_recovery_factor()
        expected = (0.05 * (1.05 ** 10)) / ((1.05 ** 10) - 1)  # ≈ 0.1295
        assert abs(result - expected) < 1e-10
    
    def test_capital_recovery_factor_zero_rate(self):
        """Test capital recovery factor with zero interest rate."""
        factor = InterestFactor(rate=0.0, time_period=10)
        result = factor.capital_recovery_factor()
        assert result == 0.1  # 1/10
    
    def test_capital_recovery_reciprocal_relationship(self):
        """Test that capital recovery factor is reciprocal of present value of annuity factor."""
        factor = InterestFactor(rate=0.05, time_period=10)
        cr_factor = factor.capital_recovery_factor()
        pva_factor = factor.present_value_of_annuity_factor()
        assert abs(cr_factor * pva_factor - 1.0) < 1e-10
    
    def test_calculate_capital_recovery_uses_factor(self):
        """Test that calculate_capital_recovery uses capital_recovery_factor method."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        factor_result = factor.capital_recovery_factor()
        calculate_result = factor.calculate_capital_recovery()
        expected = factor_result * 1000
        assert abs(calculate_result - expected) < 1e-10


class TestPresentValueOfAnnuityCalculations:
    """Test class for present value of annuity related calculations."""
    
    def test_present_value_of_annuity_factor_normal_case(self):
        """Test present value of annuity factor with standard parameters."""
        factor = InterestFactor(rate=0.05, time_period=10)
        result = factor.present_value_of_annuity_factor()
        expected = ((1.05 ** 10) - 1) / (0.05 * (1.05 ** 10))  # ≈ 7.7217
        assert abs(result - expected) < 1e-10
    
    def test_present_value_of_annuity_factor_zero_rate(self):
        """Test present value of annuity factor with zero interest rate."""
        factor = InterestFactor(rate=0.0, time_period=10)
        result = factor.present_value_of_annuity_factor()
        assert result == 10.0
    
    def test_present_value_of_annuity_factor_single_period(self):
        """Test present value of annuity factor with single time period."""
        factor = InterestFactor(rate=0.05, time_period=1)
        result = factor.present_value_of_annuity_factor()
        assert abs(result - (1 / 1.05)) < 1e-10
    
    def test_calculate_present_value_of_annuity_uses_factor(self):
        """Test that calculate_present_value_of_annuity uses the factor method."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=100)
        factor_result = factor.present_value_of_annuity_factor()
        calculate_result = factor.calculate_present_value_of_annuity()
        expected = factor_result * 100
        assert abs(calculate_result - expected) < 1e-10
    
    def test_present_value_of_annuity_with_different_amounts(self):
        """Test calculate_present_value_of_annuity with various amount values."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=150)
        result = factor.calculate_present_value_of_annuity()
        factor_value = ((1.05 ** 10) - 1) / (0.05 * (1.05 ** 10))
        expected = factor_value * 150
        assert abs(result - expected) < 1e-10


class TestEdgeCases:
    """Test class for edge cases and boundary conditions."""
    
    @pytest.mark.parametrize("rate,time_period", [
        (0.0, 1),      # Zero rate, single period
        (0.0, 100),    # Zero rate, many periods
        (-0.99, 1),    # Near minimum rate
        (10.0, 1),     # Very high rate
        (0.05, 1),     # Single period
        (0.05, 100),   # Many periods
    ])
    def test_edge_case_combinations(self, rate, time_period):
        """Test various edge case combinations of rate and time_period."""
        factor = InterestFactor(rate=rate, time_period=time_period)
        
        # Test that all methods return finite positive numbers
        assert factor.future_value_factor() > 0
        assert factor.present_value_factor() > 0
        assert factor.future_value_of_annuity_factor() > 0
        assert factor.sinking_fund_factor() > 0
        assert factor.capital_recovery_factor() > 0
        assert factor.present_value_of_annuity_factor() > 0
    
    def test_very_small_positive_rate(self):
        """Test calculations with very small positive interest rate."""
        factor = InterestFactor(rate=1e-10, time_period=10)
        # Should behave almost like zero rate case
        fva_factor = factor.future_value_of_annuity_factor()
        assert abs(fva_factor - 10.0) < 1e-5
    
    def test_large_time_periods(self):
        """Test calculations with large time periods."""
        factor = InterestFactor(rate=0.01, time_period=1000)
        # Should not overflow or underflow
        fv_factor = factor.future_value_factor()
        pv_factor = factor.present_value_factor()
        assert fv_factor > 0
        assert pv_factor > 0
        assert not (fv_factor == float('inf') or pv_factor == 0.0)


class TestMethodIntegration:
    """Test class for integration between different methods."""
    
    def test_all_calculate_methods_use_factors(self):
        """Test that all calculate_ methods properly use their corresponding factor methods."""
        factor = InterestFactor(rate=0.05, time_period=10, amount=500)
        
        # Test each calculate method uses its corresponding factor method
        assert abs(factor.calculate_future_value() - 
                  factor.future_value_factor() * 500) < 1e-10
        
        assert abs(factor.calculate_present_value() - 
                  factor.present_value_factor() * 500) < 1e-10
        
        assert abs(factor.calculate_future_value_of_annuity() - 
                  factor.future_value_of_annuity_factor() * 500) < 1e-10
        
        assert abs(factor.calculate_sinking_fund() - 
                  factor.sinking_fund_factor() * 500) < 1e-10
        
        assert abs(factor.calculate_capital_recovery() - 
                  factor.capital_recovery_factor() * 500) < 1e-10
        
        assert abs(factor.calculate_present_value_of_annuity() - 
                  factor.present_value_of_annuity_factor() * 500) < 1e-10
    
    def test_mathematical_relationships(self):
        """Test mathematical relationships between different factors."""
        factor = InterestFactor(rate=0.05, time_period=10)
        
        # Future value and present value are reciprocals
        fv_factor = factor.future_value_factor()
        pv_factor = factor.present_value_factor()
        assert abs(fv_factor * pv_factor - 1.0) < 1e-10
        
        # Sinking fund and future value of annuity are reciprocals
        sf_factor = factor.sinking_fund_factor()
        fva_factor = factor.future_value_of_annuity_factor()
        assert abs(sf_factor * fva_factor - 1.0) < 1e-10
        
        # Capital recovery and present value of annuity are reciprocals
        cr_factor = factor.capital_recovery_factor()
        pva_factor = factor.present_value_of_annuity_factor()
        assert abs(cr_factor * pva_factor - 1.0) < 1e-10


class TestRealWorldScenarios:
    """Test class for real-world financial scenarios."""
    
    def test_mortgage_calculation_scenario(self):
        """Test scenario: 30-year mortgage at 5% annual interest."""
        # Monthly interest rate and periods
        monthly_rate = 0.05 / 12
        months = 30 * 12
        loan_amount = 300000
        
        factor = InterestFactor(rate=monthly_rate, time_period=months, amount=loan_amount)
        monthly_payment = factor.calculate_capital_recovery()
        
        # Monthly payment should be around $1610 for this scenario
        assert 1600 <= monthly_payment <= 1620
    
    def test_retirement_savings_scenario(self):
        """Test scenario: Saving for retirement over 30 years."""
        annual_rate = 0.07
        years = 30
        annual_contribution = 10000
        
        factor = InterestFactor(rate=annual_rate, time_period=years, amount=annual_contribution)
        future_value = factor.calculate_future_value_of_annuity()
        
        # Should accumulate to around $944k
        assert 900000 <= future_value <= 1000000
    
    def test_investment_growth_scenario(self):
        """Test scenario: Lump sum investment growth."""
        rate = 0.08
        years = 20
        initial_investment = 50000
        
        factor = InterestFactor(rate=rate, time_period=years, amount=initial_investment)
        future_value = factor.calculate_future_value()
        
        # Should grow to around $233k
        assert 230000 <= future_value <= 240000


# Fixture for commonly used InterestFactor instance
@pytest.fixture
def standard_factor():
    """Fixture providing a standard InterestFactor instance for testing."""
    return InterestFactor(rate=0.05, time_period=10, amount=1000)


class TestWithFixtures:
    """Test class demonstrating use of pytest fixtures."""
    
    def test_with_standard_factor(self, standard_factor):
        """Test using the standard factor fixture."""
        assert standard_factor.rate == 0.05
        assert standard_factor.time_period == 10
        assert standard_factor.amount == 1000
        
        # Test that calculations work with the fixture
        fv = standard_factor.calculate_future_value()
        assert fv > 1000  # Should be greater than initial amount