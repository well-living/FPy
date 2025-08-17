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
from typing import List, Optional, Union
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class PaymentFrequency(str, Enum):
    """Enumeration for payment frequency options."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"


class RepaymentMethod(str, Enum):
    """Enumeration for repayment method options."""
    EQUAL_PAYMENT = "equal_payment"  # 元利均等返済
    EQUAL_PRINCIPAL = "equal_principal"  # 元金均等返済


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
    loan_term : int
        Loan term in months
    payment_frequency : PaymentFrequency
        Frequency of payments
    repayment_method : RepaymentMethod
        Method of repayment (equal payment or equal principal)
    interest_rate_type : InterestRateType
        Type of interest rate (fixed or variable)
    remaining_balance : Optional[float]
        Current remaining balance of the loan
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
    
    loan_term: int = Field(
        ...,
        gt=0,
        le=600,  # Maximum 50 years
        description="Loan term in months",
        examples=[360, 240, 120]  # 30年, 20年, 10年
    )
    
    payment_frequency: PaymentFrequency = Field(
        default=PaymentFrequency.MONTHLY,
        description="Frequency of loan payments"
    )
    
    repayment_method: RepaymentMethod = Field(
        ...,
        description="Method of loan repayment"
    )
    
    interest_rate_type: InterestRateType = Field(
        ...,
        description="Type of interest rate"
    )
    
    remaining_balance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current remaining balance of the loan"
    )
    
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
        frequency_map = {
            PaymentFrequency.MONTHLY: 12,
            PaymentFrequency.QUARTERLY: 4,
            PaymentFrequency.SEMI_ANNUALLY: 2,
            PaymentFrequency.ANNUALLY: 1
        }
        return frequency_map[self.payment_frequency]
    
    def calculate_total_payments(self) -> int:
        """
        Calculate total number of payments over the loan term.
        
        Returns
        -------
        int
            Total number of payments
        """
        return (self.loan_term * self.get_annual_payment_count()) // 12
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "examples": [
                {
                    "name": "フラット35",
                    "interest_rate": 0.013,
                    "contract_date": "2024-01-15",
                    "principal": 35000000,
                    "loan_term": 360,
                    "payment_frequency": "monthly",
                    "repayment_method": "equal_payment",
                    "interest_rate_type": "fixed",
                    "remaining_balance": 35000000
                },
                {
                    "name": "JASSO奨学金",
                    "interest_rate": 0.000,
                    "contract_date": "2023-04-01",
                    "principal": 2400000,
                    "loan_term": 240,
                    "payment_frequency": "monthly",
                    "repayment_method": "equal_payment",
                    "interest_rate_type": "fixed",
                    "remaining_balance": 2400000
                }
            ]
        }