# fpyjp/schemas/loan.py
"""
Loan Schema Definition

Defines data models for loan using Pydantic V2.

以下のような政府系機関や金融機関の住宅ローンや教育ローンなど様々なローンに対応
・住宅金融支援機構の住宅ローン「フラット35」
  https://www.simulation.jhf.go.jp/flat35/kinri/index.php/rates/top/
・日本学生支援機構の教育ローン(奨学金)
  https://www.jasso.go.jp/shogakukin/about/taiyo/index.html

"""

import datetime
import math
from typing import List, Optional, Union
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field


class RepaymentMethod(str, Enum):
    """Enumeration for repayment method options."""
    EQUAL_PRINCIPAL = "equal_principal"  # 元金均等返済
    EQUAL_PAYMENT = "equal_payment"  # 元利均等返済


class InterestRateType(str, Enum):
    """Enumeration for interest rate type options."""
    FIXED = "fixed"  # 固定金利
    VARIABLE = "variable"  # 変動金利


class Loan(BaseModel):
    """
    Loan data model for various types of loans including mortgage and education loans.
    
    This model supports different loan types such as:
    - Mortgage loans (住宅ローン) like Flat35
    - Education loans (教育ローン) from JASSO
    - Other consumer loans
    
    Attributes
    ----------
    name : str
        Name of the loan product
    interest_rate : Union[float, List[float]]
        Interest rate(s) as decimal (e.g., 0.025 for 2.5%)
    contract_date : datetime.date
        Date when the loan contract was signed
    principal : float
        Original loan amount (元本)
    total_term_months : int
        Total loan term in months
    remaining_term_months : int
        Remaining loan term in months
    payment_frequency : int
        Payment frequency in months (1=monthly, 3=quarterly, 6=semi-annually, 12=annually)
    repayment_method : RepaymentMethod
        Method of repayment (equal payment or equal principal)
    interest_rate_type : InterestRateType
        Type of interest rate (fixed or variable)
    remaining_balance : Optional[float]
        Current remaining balance of the loan
    total_term_years : int
        Total loan term in years (computed from total_term_months, rounded up)
    remaining_term_years : int
        Remaining loan term in years (computed from remaining_term_months, rounded up)
    """
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Name of the loan product",
        examples=["フラット35", "JASSO奨学金", "住宅ローン"]
    )
    
    interest_rate: Union[float, List[float]] = Field(
        ...,
        description="Interest rate as decimal (e.g., 0.025 for 2.5%). List for variable rates.",
        examples=[0.025, [0.02, 0.025, 0.03]]
    )
    
    contract_date: Optional[datetime.date] = Field(
        None,
        description="Date when the loan contract was signed"
    )
    
    principal: Optional[float] = Field(
        None,
        gt=0,
        description="Original loan amount in currency units",
        examples=[35000000, 2400000]
    )
    
    total_term_months: Optional[int] = Field(
        None,
        gt=0,
        le=600,  # Maximum 50 years
        description="Total loan term in months",
        examples=[360, 240, 120]  # 30年, 20年, 10年
    )
    
    remaining_term_months: Optional[int] = Field(
        None,
        gt=0,
        le=600,  # Maximum 50 years
        description="Remaining loan term in months",
        examples=[300, 180, 60]  # 25年, 15年, 5年
    )
    
    payment_frequency: int = Field(
        default=1,
        gt=0,
        le=12,
        description="Payment frequency in months (1=monthly, 3=quarterly, 6=semi-annually, 12=annually)",
        examples=[1, 3, 6, 12]
    )
    
    repayment_method: RepaymentMethod = Field(
        default=RepaymentMethod.EQUAL_PRINCIPAL,  # 元金均等をデフォルトに
        description="Method of loan repayment"
    )
    
    interest_rate_type: InterestRateType = Field(
        default=InterestRateType.FIXED,
        description="Type of interest rate"
    )
    
    remaining_balance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current remaining balance of the loan"
    )
    
    @computed_field
    @property
    def total_term_years(self) -> Optional[int]:
        """
        Total loan term in years (computed from total_term_months, rounded up).
        
        Returns
        -------
        Optional[int]
            Total loan term in years, or None if total_term_months is not set
        """
        if self.total_term_months is None:
            return None
        return math.ceil(self.total_term_months / 12)
    
    @computed_field
    @property
    def remaining_term_years(self) -> Optional[int]:
        """
        Remaining loan term in years (computed from remaining_term_months, rounded up).
        
        Returns
        -------
        Optional[int]
            Remaining loan term in years, or None if remaining_term_months is not set
        """
        if self.remaining_term_months is None:
            return None
        return math.ceil(self.remaining_term_months / 12)
    
    @field_validator('interest_rate')
    @classmethod
    def validate_interest_rate(cls, v):
        """
        Validate interest rate values.
        
        Parameters
        ----------
        v : Union[float, List[float]]
            Interest rate value(s) to validate
            
        Returns
        -------
        Union[float, List[float]]
            Validated interest rate value(s)
            
        Raises
        ------
        ValueError
            If interest rate is not greater than -1 (allowing for negative rates)
        """
        if isinstance(v, list):
            for rate in v:
                if not isinstance(rate, (int, float)) or rate <= -1:
                    raise ValueError("All interest rates must be greater than -1")
            if len(v) == 0:
                raise ValueError("Interest rate list cannot be empty")
        else:
            if not isinstance(v, (int, float)) or v <= -1:
                raise ValueError("Interest rate must be greater than -1")
        return v
    
    @field_validator('remaining_balance')
    @classmethod
    def validate_remaining_balance(cls, v, info):
        """
        Validate remaining balance does not exceed principal.
        
        Parameters
        ----------
        v : Optional[float]
            Remaining balance to validate
        info : ValidationInfo
            Validation context containing other field values
            
        Returns
        -------
        Optional[float]
            Validated remaining balance
            
        Raises
        ------
        ValueError
            If remaining balance exceeds principal amount
        """
        if v is not None and 'principal' in info.data:
            if v > info.data['principal']:
                raise ValueError("Remaining balance cannot exceed principal amount")
        return v
    
    @field_validator('remaining_term_months')
    @classmethod
    def validate_remaining_term_months(cls, v, info):
        """
        Validate remaining term does not exceed total term.
        
        Parameters
        ----------
        v : Optional[int]
            Remaining term months to validate
        info : ValidationInfo
            Validation context containing other field values
            
        Returns
        -------
        Optional[int]
            Validated remaining term months
            
        Raises
        ------
        ValueError
            If remaining term exceeds total term
        """
        if v is not None and 'total_term_months' in info.data:
            if info.data['total_term_months'] is not None and v > info.data['total_term_months']:
                raise ValueError("Remaining term cannot exceed total term")
        return v
    
    @model_validator(mode='after')
    def validate_loan_consistency(self):
        """
        Validate consistency between loan fields.
        
        Returns
        -------
        Loan
            Validated loan instance
            
        Raises
        ------
        ValueError
            If variable rate type has single interest rate value or
            if fixed rate type has multiple interest rate values
        """
        # Validate interest rate type consistency
        if self.interest_rate_type == InterestRateType.VARIABLE:
            if not isinstance(self.interest_rate, list):
                raise ValueError("Variable interest rate type requires list of rates")
        elif self.interest_rate_type == InterestRateType.FIXED:
            if isinstance(self.interest_rate, list):
                raise ValueError("Fixed interest rate type requires single rate value")
        
        # Set default remaining balance to principal if not provided
        if self.remaining_balance is None:
            self.remaining_balance = self.principal
        
        # Set default remaining term to total term if not provided
        if self.remaining_term_months is None:
            self.remaining_term_months = self.total_term_months
            
        return self
    
    def get_current_interest_rate(self) -> float:
        """
        Get the current applicable interest rate.
        
        Returns
        -------
        float
            Current interest rate as decimal
            
        Notes
        -----
        For fixed rates, returns the single rate.
        For variable rates, returns the first rate in the list.
        """
        if isinstance(self.interest_rate, list):
            return self.interest_rate[0]
        return self.interest_rate
    
    def get_annual_payment_count(self) -> int:
        """
        Get number of payments per year based on payment frequency.
        
        Returns
        -------
        int
            Number of payments per year
        """
        return 12 // self.payment_frequency
    
    def calculate_total_payments(self) -> Optional[int]:
        """
        Calculate total number of payments over the total loan term.
        
        Returns
        -------
        Optional[int]
            Total number of payments, or None if total_term_months is not set
        """
        if self.total_term_months is None:
            return None
        return self.total_term_months // self.payment_frequency
    
    def calculate_remaining_payments(self) -> Optional[int]:
        """
        Calculate remaining number of payments.
        
        Returns
        -------
        Optional[int]
            Remaining number of payments, or None if remaining_term_months is not set
        """
        if self.remaining_term_months is None:
            return None
        return self.remaining_term_months // self.payment_frequency
    
    def calculate_payments_made(self) -> Optional[int]:
        """
        Calculate number of payments already made.
        
        Returns
        -------
        Optional[int]
            Number of payments made, or None if either total or remaining term is not set
        """
        total_payments = self.calculate_total_payments()
        remaining_payments = self.calculate_remaining_payments()
        
        if total_payments is None or remaining_payments is None:
            return None
            
        return total_payments - remaining_payments
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime.date: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "examples": [
                {
                    "name": "フラット35",
                    "interest_rate": 0.013,
                    "contract_date": "2024-01-15",
                    "principal": 35000000,
                    "total_term_months": 360,
                    "remaining_term_months": 300,
                    "payment_frequency": 1,
                    "repayment_method": "equal_principal",
                    "interest_rate_type": "fixed",
                    "remaining_balance": 28000000
                },
                {
                    "name": "JASSO奨学金",
                    "interest_rate": 0.000,
                    "contract_date": "2023-04-01",
                    "principal": 2400000,
                    "total_term_months": 240,
                    "remaining_term_months": 180,
                    "payment_frequency": 1,
                    "repayment_method": "equal_principal",
                    "interest_rate_type": "fixed",
                    "remaining_balance": 1800000
                }
            ]
        }