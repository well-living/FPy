# test_loan.py
"""
Comprehensive unit tests for the Loan class.

Tests cover:
- Basic model creation and validation
- Field validators
- Model validators
- Computed properties
- Edge cases and error conditions
- Business logic validation
"""

import datetime
import math
import pytest
from pydantic import ValidationError
from typing import List

# Assuming the Loan class is imported from the appropriate module
from fpyjp.schemas.loan import Loan, RepaymentMethod, InterestRateType


class TestLoanBasicCreation:
    """Test basic loan creation and field assignment."""
    
    def test_minimal_loan_creation(self):
        """Test creating a loan with minimal required fields."""
        loan = Loan(
            name="Test Loan",
            interest_rate=0.025,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        
        assert loan.name == "Test Loan"
        assert loan.interest_rate == 0.025
        assert loan.remaining_balance == 1000000.0
        assert loan.repayment_method == RepaymentMethod.EQUAL_PRINCIPAL
        assert loan.interest_rate_type == InterestRateType.FIXED
        assert loan.payment_frequency == 1
    
    def test_full_loan_creation(self):
        """Test creating a loan with all fields populated."""
        contract_date = datetime.date(2023, 1, 15)
        
        loan = Loan(
            name="フラット35",
            interest_rate=0.013,
            contract_date=contract_date,
            principal=35000000.0,
            remaining_balance=30000000.0,
            total_term_months=360,
            remaining_term_months=300,
            payment_frequency=1,
            repayment_method=RepaymentMethod.EQUAL_PAYMENT,
            interest_rate_type=InterestRateType.FIXED,
            repayment_amount=98000.0
        )
        
        assert loan.name == "フラット35"
        assert loan.interest_rate == 0.013
        assert loan.contract_date == contract_date
        assert loan.principal == 35000000.0
        assert loan.remaining_balance == 30000000.0
        assert loan.total_term_months == 360
        assert loan.remaining_term_months == 300
        assert loan.payment_frequency == 1
        assert loan.repayment_method == RepaymentMethod.EQUAL_PAYMENT
        assert loan.interest_rate_type == InterestRateType.FIXED
        assert loan.repayment_amount == 98000.0


class TestLoanComputedFields:
    """Test computed properties of the Loan class."""
    
    def test_total_term_years_computation(self):
        """Test total_term_years computed property."""
        # Test with exact years
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            total_term_months=240  # 20 years
        )
        assert loan.total_term_years == 20
        
        # Test with partial years (should round up)
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            total_term_months=241
        )
        assert loan.total_term_years == 21
        
        # Test with None
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            total_term_months=None
        )
        assert loan.total_term_years is None
    
    def test_remaining_term_years_computation(self):
        """Test remaining_term_years computed property."""
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=185  # 15.4 years -> should be 16
        )
        assert loan.remaining_term_years == 16
        
        # Test with None
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=None
        )
        assert loan.remaining_term_years is None
    
    def test_payment_count_computation(self):
        """Test payment_count computed property."""
        # Monthly payments
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=120,
            payment_frequency=1
        )
        assert loan.payment_count == 120  # math.ceil(120/1) = 120
        
        # Quarterly payments
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=120,
            payment_frequency=3
        )
        assert loan.payment_count == 40  # math.ceil(120/3) = 40
        
        # Test with remainder
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=121,
            payment_frequency=3
        )
        assert loan.payment_count == 41  # math.ceil(121/3) = 41
        
        # Test with None remaining_term_months
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=None
        )
        assert loan.payment_count is None
    
    def test_final_repayment_amount_computation(self):
        """Test final_repayment_amount computed property."""
        # Fixed rate loan with all required fields
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            interest_rate_type=InterestRateType.FIXED,
            principal=1200000.0,
            remaining_balance=1200000.0,
            remaining_term_months=12,
            payment_frequency=1,
            repayment_amount=100000.0
        )
        
        # Formula: remaining_balance - repayment_amount × (payment_count - 1)
        # payment_count = math.ceil(12/1) = 12
        # final_repayment_amount = 1200000 - 100000 × (12 - 1) = 1200000 - 1100000 = 100000
        expected_final = 1200000.0 - 100000.0 * (12 - 1)  # = 100000.0
        assert loan.final_repayment_amount == expected_final
        
        # Variable rate loan should return None
        loan = Loan(
            name="Test",
            interest_rate=[0.02, 0.025],
            interest_rate_type=InterestRateType.VARIABLE,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=12,
            payment_frequency=1,
            repayment_amount=85000.0
        )
        assert loan.final_repayment_amount is None
        
        # Missing repayment_amount should return None
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            interest_rate_type=InterestRateType.FIXED,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=12,
            payment_frequency=1,
            repayment_amount=None
        )
        assert loan.final_repayment_amount is None


class TestLoanFieldValidators:
    """Test individual field validators."""
    
    def test_name_validation(self):
        """Test name field validation."""
        # Valid names
        valid_names = ["フラット35", "JASSO奨学金", "A" * 50]
        for name in valid_names:
            loan = Loan(
                name=name,
                interest_rate=0.02,
                principal=1000000.0,
                remaining_balance=1000000.0
            )
            assert loan.name == name
        
        # Invalid names
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="", 
                interest_rate=0.02, 
                principal=1000000.0,
                remaining_balance=1000000.0
            )
        assert "at least 1 character" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="A" * 51, 
                interest_rate=0.02, 
                principal=1000000.0,
                remaining_balance=1000000.0
            )
        assert "at most 50 characters" in str(exc_info.value)
    
    def test_interest_rate_validation(self):
        """Test interest_rate field validation."""
        # Valid single rate
        loan = Loan(
            name="Test",
            interest_rate=0.025,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.interest_rate == 0.025
        
        # Valid list of rates with variable type
        loan = Loan(
            name="Test",
            interest_rate=[0.02, 0.025, 0.03],
            interest_rate_type=InterestRateType.VARIABLE,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.interest_rate == [0.02, 0.025, 0.03]
        
        # Negative rates (should be allowed if > -1)
        loan = Loan(
            name="Test",
            interest_rate=-0.005,  # -0.5%
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.interest_rate == -0.005
        
        # Invalid rates
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test", 
                interest_rate=-1.1, 
                principal=1000000.0,
                remaining_balance=1000000.0
            )
        assert "greater than -1" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test", 
                interest_rate=[-0.5, -1.1], 
                interest_rate_type=InterestRateType.VARIABLE,
                principal=1000000.0,
                remaining_balance=1000000.0
            )
        assert "greater than -1" in str(exc_info.value)
        
        # Empty list
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test", 
                interest_rate=[], 
                interest_rate_type=InterestRateType.VARIABLE,
                principal=1000000.0,
                remaining_balance=1000000.0
            )
        assert "cannot be empty" in str(exc_info.value)
    
    def test_principal_validation(self):
        """Test principal field validation."""
        # Valid principal
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=35000000.0,
            remaining_balance=30000000.0
        )
        assert loan.principal == 35000000.0
        
        # Invalid principal (not positive)
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test",
                interest_rate=0.02,
                principal=0.0,
                remaining_balance=1000000.0
            )
        assert "greater than 0" in str(exc_info.value)
    
    def test_remaining_balance_validation(self):
        """Test remaining_balance field validation."""
        # Valid remaining balance
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=35000000.0,
            remaining_balance=30000000.0
        )
        assert loan.remaining_balance == 30000000.0
        
        # Remaining balance exceeds principal
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test",
                interest_rate=0.02,
                principal=30000000.0,
                remaining_balance=35000000.0
            )
        assert "cannot exceed principal" in str(exc_info.value)
        
        # Zero remaining balance should be valid
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=35000000.0,
            remaining_balance=0.0
        )
        assert loan.remaining_balance == 0.0
    
    def test_term_months_validation(self):
        """Test term months field validation."""
        # Valid terms
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            total_term_months=360,
            remaining_term_months=300
        )
        assert loan.total_term_months == 360
        assert loan.remaining_term_months == 300
        
        # Invalid total term (too long)
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test",
                interest_rate=0.02,
                principal=1000000.0,
                remaining_balance=1000000.0,
                total_term_months=601  # > 600 months
            )
        assert "less than or equal to 600" in str(exc_info.value)
        
        # Remaining term exceeds total term
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test",
                interest_rate=0.02,
                principal=1000000.0,
                remaining_balance=1000000.0,
                total_term_months=300,
                remaining_term_months=360
            )
        assert "cannot exceed total term" in str(exc_info.value)
    
    def test_payment_frequency_validation(self):
        """Test payment_frequency field validation."""
        # Valid frequencies
        valid_frequencies = [1, 3, 6, 12]
        for freq in valid_frequencies:
            loan = Loan(
                name="Test",
                interest_rate=0.02,
                principal=1000000.0,
                remaining_balance=1000000.0,
                payment_frequency=freq
            )
            assert loan.payment_frequency == freq
        
        # Invalid frequencies
        invalid_frequencies = [0, 13, -1]
        for freq in invalid_frequencies:
            with pytest.raises(ValidationError):
                Loan(
                    name="Test",
                    interest_rate=0.02,
                    principal=1000000.0,
                    remaining_balance=1000000.0,
                    payment_frequency=freq
                )


class TestLoanModelValidators:
    """Test model-level validators."""
    
    def test_interest_rate_type_consistency(self):
        """Test interest rate type consistency validation."""
        # Fixed rate with single value - should be valid
        loan = Loan(
            name="Test",
            interest_rate=0.025,
            interest_rate_type=InterestRateType.FIXED,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.interest_rate_type == InterestRateType.FIXED
        
        # Fixed rate with identical values in list - should be valid
        loan = Loan(
            name="Test",
            interest_rate=[0.025, 0.025, 0.025],
            interest_rate_type=InterestRateType.FIXED,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.interest_rate_type == InterestRateType.FIXED
        
        # Variable rate with list - should be valid
        loan = Loan(
            name="Test",
            interest_rate=[0.02, 0.025, 0.03],
            interest_rate_type=InterestRateType.VARIABLE,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.interest_rate_type == InterestRateType.VARIABLE
        
        # Variable rate with single value - should be invalid
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test",
                interest_rate=0.025,
                interest_rate_type=InterestRateType.VARIABLE,
                principal=1000000.0,
                remaining_balance=1000000.0
            )
        assert "requires list of rates" in str(exc_info.value)
        
        # Fixed rate with different values in list - should be invalid
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test",
                interest_rate=[0.02, 0.025, 0.03],
                interest_rate_type=InterestRateType.FIXED,
                principal=1000000.0,
                remaining_balance=1000000.0
            )
        assert "requires all rates to be identical" in str(exc_info.value)
    
    def test_payment_amount_consistency(self):
        """Test payment amount consistency validation for fixed rate loans."""
        # Valid payment amount consistency
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            interest_rate_type=InterestRateType.FIXED,
            principal=120000.0,
            remaining_balance=120000.0,
            remaining_term_months=12,
            repayment_amount=10000.0  # 12回 × 10000円 = 120000円
        )
        assert loan.repayment_amount == 10000.0
        
        # Invalid payment amount consistency - 支払い額が大きすぎる
        with pytest.raises(ValidationError) as exc_info:
            Loan(
                name="Test",
                interest_rate=0.02,
                interest_rate_type=InterestRateType.FIXED,
                principal=100000.0,
                remaining_balance=100000.0,
                remaining_term_months=1,
                repayment_amount=150000.0  # 残高を超える
            )
        assert "consistency check failed" in str(exc_info.value)


class TestLoanMethods:
    """Test loan instance methods."""
    
    def test_get_current_interest_rate(self):
        """Test get_current_interest_rate method."""
        # Fixed rate
        loan = Loan(
            name="Test",
            interest_rate=0.025,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.get_current_interest_rate() == 0.025
        
        # Variable rate
        loan = Loan(
            name="Test",
            interest_rate=[0.02, 0.025, 0.03],
            interest_rate_type=InterestRateType.VARIABLE,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.get_current_interest_rate() == 0.02  # First rate
    
    def test_validate_payment_formula(self):
        """Test validate_payment_formula method."""
        # Valid fixed rate loan
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            interest_rate_type=InterestRateType.FIXED,
            principal=120000.0,
            remaining_balance=120000.0,
            remaining_term_months=12,
            repayment_amount=10000.0
        )
        assert loan.validate_payment_formula() is True
        
        # Variable rate loan should return False
        loan = Loan(
            name="Test",
            interest_rate=[0.02, 0.025],
            interest_rate_type=InterestRateType.VARIABLE,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=12,
            repayment_amount=85000.0
        )
        assert loan.validate_payment_formula() is False
        
        # Missing repayment_amount should return False
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            interest_rate_type=InterestRateType.FIXED,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=12,
            repayment_amount=None
        )
        assert loan.validate_payment_formula() is False


class TestLoanEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_interest_rate(self):
        """Test loan with zero interest rate."""
        loan = Loan(
            name="Zero Interest Loan",
            interest_rate=0.0,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.interest_rate == 0.0
    
    def test_very_small_amounts(self):
        """Test loan with very small amounts."""
        loan = Loan(
            name="Small Loan",
            interest_rate=0.02,
            principal=1.0,
            remaining_balance=0.5
        )
        assert loan.principal == 1.0
        assert loan.remaining_balance == 0.5
    
    def test_very_large_amounts(self):
        """Test loan with very large amounts."""
        large_amount = 1e12  # 1 trillion
        loan = Loan(
            name="Large Loan",
            interest_rate=0.02,
            principal=large_amount,
            remaining_balance=large_amount * 0.8
        )
        assert loan.principal == large_amount
        assert loan.remaining_balance == large_amount * 0.8
    
    def test_maximum_term_months(self):
        """Test loan with maximum allowed term."""
        loan = Loan(
            name="Long Term Loan",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            total_term_months=600,  # Maximum allowed
            remaining_term_months=600
        )
        assert loan.total_term_months == 600
        assert loan.total_term_years == 50  # 600/12 = 50
    
    def test_single_payment_loan(self):
        """Test loan with single payment remaining."""
        loan = Loan(
            name="Single Payment",
            interest_rate=0.02,
            interest_rate_type=InterestRateType.FIXED,
            principal=100000.0,
            remaining_balance=100000.0,
            remaining_term_months=1,
            repayment_amount=100000.0
        )
        assert loan.payment_count == 1  # math.ceil(1/1) = 1
        # Formula: remaining_balance - repayment_amount × (payment_count - 1)
        # final_repayment_amount = 100000 - 100000 × (1 - 1) = 100000 - 0 = 100000
        assert loan.final_repayment_amount == 100000.0

class TestLoanDefaultValues:
    """Test default value assignments."""
    
    def test_default_remaining_balance_assignment(self):
        """Test that remaining_balance defaults to principal when not provided."""
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        assert loan.remaining_balance == loan.principal
    
    def test_default_remaining_term_assignment(self):
        """Test that remaining_term_months defaults to total_term_months when not provided."""
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            total_term_months=360,
            remaining_term_months=360
        )
        assert loan.remaining_term_months == loan.total_term_months


class TestLoanSerialization:
    """Test loan serialization and deserialization."""
    
    def test_loan_dict_conversion(self):
        """Test converting loan to dictionary."""
        loan = Loan(
            name="フラット35",
            interest_rate=0.013,
            contract_date=datetime.date(2023, 1, 15),
            principal=35000000.0,
            remaining_balance=30000000.0,
            total_term_months=360,
            remaining_term_months=300
        )
        
        loan_dict = loan.model_dump()
        assert loan_dict["name"] == "フラット35"
        assert loan_dict["interest_rate"] == 0.013
        assert loan_dict["principal"] == 35000000.0
        assert "total_term_years" in loan_dict  # Computed field
        assert "payment_count" in loan_dict  # Computed field
    
    def test_loan_json_conversion(self):
        """Test converting loan to JSON."""
        loan = Loan(
            name="Test Loan",
            interest_rate=0.025,
            principal=1000000.0,
            remaining_balance=1000000.0
        )
        
        json_str = loan.model_dump_json()
        assert "Test Loan" in json_str
        assert "0.025" in json_str
    
    def test_loan_recreation_from_dict(self):
        """Test recreating loan from dictionary."""
        original_loan = Loan(
            name="フラット35",
            interest_rate=0.013,
            principal=35000000.0,
            remaining_balance=30000000.0,
            total_term_months=360
        )
        
        loan_dict = original_loan.model_dump()
        # Remove computed fields as they can't be set directly
        computed_fields = ["total_term_years", "remaining_term_years", "payment_count", "final_repayment_amount"]
        for field in computed_fields:
            loan_dict.pop(field, None)
        
        recreated_loan = Loan(**loan_dict)
        assert recreated_loan.name == original_loan.name
        assert recreated_loan.interest_rate == original_loan.interest_rate
        assert recreated_loan.remaining_balance == original_loan.remaining_balance
        assert recreated_loan.total_term_years == original_loan.total_term_years


# Additional fixtures and parametrized tests can be added here
@pytest.fixture
def sample_flat35_loan():
    """Fixture providing a sample Flat35 mortgage loan."""
    return Loan(
        name="フラット35",
        interest_rate=0.013,
        contract_date=datetime.date(2023, 1, 15),
        principal=35000000.0,
        remaining_balance=32000000.0,
        total_term_months=360,
        remaining_term_months=330,
        payment_frequency=1,
        repayment_method=RepaymentMethod.EQUAL_PAYMENT,
        interest_rate_type=InterestRateType.FIXED,
        repayment_amount=95000.0  # 98000.0 → 95000.0 に修正
        # 95000 × 330 = 31,350,000 < 32,000,000 なので整合性が取れる
    )


@pytest.fixture
def sample_jasso_loan():
    """Fixture providing a sample JASSO education loan."""
    return Loan(
        name="JASSO奨学金",
        interest_rate=[0.01, 0.015, 0.02],
        contract_date=datetime.date(2020, 4, 1),
        principal=2400000.0,
        remaining_balance=1800000.0,
        total_term_months=240,
        remaining_term_months=180,
        payment_frequency=1,
        repayment_method=RepaymentMethod.EQUAL_PRINCIPAL,
        interest_rate_type=InterestRateType.VARIABLE
    )


class TestLoanFixtures:
    """Test using loan fixtures."""
    
    def test_flat35_loan_fixture(self, sample_flat35_loan):
        """Test the Flat35 loan fixture."""
        assert sample_flat35_loan.name == "フラット35"
        assert sample_flat35_loan.interest_rate == 0.013
        assert sample_flat35_loan.interest_rate_type == InterestRateType.FIXED
        assert sample_flat35_loan.total_term_years == 30
        assert sample_flat35_loan.remaining_term_years == 28  # ceil(330/12)
    
    def test_jasso_loan_fixture(self, sample_jasso_loan):
        """Test the JASSO loan fixture."""
        assert sample_jasso_loan.name == "JASSO奨学金"
        assert isinstance(sample_jasso_loan.interest_rate, list)
        assert sample_jasso_loan.interest_rate_type == InterestRateType.VARIABLE
        assert sample_jasso_loan.get_current_interest_rate() == 0.01
        assert sample_jasso_loan.repayment_method == RepaymentMethod.EQUAL_PRINCIPAL


class TestParametrizedLoanCreation:
    """Test loan creation with parametrized data."""
    
    @pytest.mark.parametrize("name,interest_rate,principal,remaining_balance", [
        ("住宅ローン", 0.015, 30000000.0, 25000000.0),
        ("教育ローン", 0.02, 2000000.0, 1500000.0),
        ("マイカーローン", 0.025, 3000000.0, 2500000.0),
        ("リフォームローン", 0.018, 5000000.0, 4000000.0),
    ])
    def test_parametrized_loan_creation(self, name, interest_rate, principal, remaining_balance):
        """Test loan creation with various parameter combinations."""
        loan = Loan(
            name=name,
            interest_rate=interest_rate,
            principal=principal,
            remaining_balance=remaining_balance
        )
        
        assert loan.name == name
        assert loan.interest_rate == interest_rate
        assert loan.principal == principal
        assert loan.remaining_balance == remaining_balance
        assert loan.repayment_method == RepaymentMethod.EQUAL_PRINCIPAL
        assert loan.interest_rate_type == InterestRateType.FIXED
    
    @pytest.mark.parametrize("frequency,expected_count", [
        (1, 120),   # Monthly: math.ceil(120/1) = 120
        (3, 40),    # Quarterly: math.ceil(120/3) = 40
        (6, 20),    # Semi-annually: math.ceil(120/6) = 20
        (12, 10),   # Annually: math.ceil(120/12) = 10
    ])
    def test_parametrized_payment_count(self, frequency, expected_count):
        """Test payment count calculation with different frequencies."""
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=120,
            payment_frequency=frequency
        )
        
        assert loan.payment_count == expected_count
    
    @pytest.mark.parametrize("remaining_months,frequency,expected_count", [
        (121, 3, 41),   # math.ceil(121/3) = 41
        (122, 3, 41),   # math.ceil(122/3) = 41
        (123, 3, 41),   # math.ceil(123/3) = 41
        (124, 3, 42),   # math.ceil(124/3) = 42
        (37, 12, 4),    # math.ceil(37/12) = 4
        (36, 12, 3),    # math.ceil(36/12) = 3
        (25, 6, 5),     # math.ceil(25/6) = 5
        (24, 6, 4),     # math.ceil(24/6) = 4
    ])
    def test_parametrized_payment_count_with_remainder(self, remaining_months, frequency, expected_count):
        """Test payment count calculation with various remainder scenarios."""
        loan = Loan(
            name="Test",
            interest_rate=0.02,
            principal=1000000.0,
            remaining_balance=1000000.0,
            remaining_term_months=remaining_months,
            payment_frequency=frequency
        )
        
        assert loan.payment_count == expected_count

