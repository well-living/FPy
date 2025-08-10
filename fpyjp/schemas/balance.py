# fpyjp/schemas/balance.py
"""
Asset and Liability Management System Schema Definition

Defines data models for assets and liabilities using Pydantic V2.
"""

from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class AssetLiabilitySchema(BaseModel):
    """
    Asset and Liability model for financial calculations.
    
    This class represents an asset or liability with price, unit, and balance
    information, along with optional book balance, cash inflow per unit, and rate.
    
    Attributes
    ----------
    name : str
        Name of the asset or liability, must be between 1 and 20 characters
    price : Optional[Union[float, List[float]]]
        Price per unit of the asset or liability
    unit : Optional[float]
        Number of units of the asset or liability
    balance : Optional[float]
        Asset or liability balance. (Fair value, market value, etc.)
    book_balance : Optional[float]
        Book value of the asset or liability, defaults to balance if not provided
    cashinflow_per_unit : Optional[Union[float, List[float]]]
        Cash inflow per unit, defaults to 0.0 if not provided
        (Interest from bonds and loans, dividends from stocks, distributions from investment trusts, etc.)
    rate : Optional[Union[float, List[float]]]
        Price growth rate (rate of return), must be greater than -1, defaults to None if not provided

    Notes
    -----
    - `price`, `unit` and `balance` must satisfy the relationship:
        - `price * unit = balance`
    - `price` and `rate` must satisfy the relationship:
        - `price[i] * (1 + rate[i]) = price[i+1] for all i`
    """
    
    name: str = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="Name of the asset or liability"
    )
    price: Optional[Union[float, List[float]]] = Field(
        default=None,
        description="Price per unit of the asset or liability"
    )
    unit: Optional[float] = Field(
        default=None,
        ge=0,
        description="Number of units of the asset or liability (must be non-negative)"
    )
    balance: Optional[float] = Field(
        default=None,
        description="Asset or liability balance (Fair value, market value, etc.)"
    )
    book_balance: Optional[float] = Field(
        default=None,
        description="Book value of the asset or liability, defaults to balance if not provided"
    )
    cashinflow_per_unit: Optional[Union[float, List[float]]] = Field(
        default=0.0,
        description="Cash inflow per unit (interest, dividends, distributions, etc.)"
    )
    rate: Optional[Union[float, List[float]]] = Field(
        default=None,
        description="Price growth rate (rate of return), must be greater than -1"
    )
    
    @field_validator('rate')
    @classmethod
    def validate_rate(cls, v: Optional[Union[float, List[float]]]) -> Optional[Union[float, List[float]]]:
        """
        Validate that rate(s) are greater than -1.
        
        Parameters
        ----------
        v : Optional[Union[float, List[float]]]
            Rate value(s) to validate
            
        Returns
        -------
        Optional[Union[float, List[float]]]
            Validated rate value(s)
            
        Raises
        ------
        ValueError
            If any rate is not greater than -1
        """
        if v is None:
            return None
        
        if isinstance(v, list):
            for i, rate in enumerate(v):
                if rate <= -1:
                    raise ValueError(f'rate[{i}] must be greater than -1, got {rate}')
            return v
        else:
            if v <= -1:
                raise ValueError(f'rate must be greater than -1, got {v}')
            return v
    
    @model_validator(mode='after')
    def validate_price_unit_balance(self):
        """
        Validate that exactly two of price, unit, and balance are provided,
        and calculate the third value. Also set book_balance if not provided.
        Additionally, validate and align price and rate lists.
        
        Returns
        -------
        AssetLiability
            Self with calculated values
            
        """
        provided_values = sum([
            self.price is not None,
            self.unit is not None,
            self.balance is not None
        ])
        
        if provided_values < 2:
            raise ValueError(
                'At least two of price, unit, and balance must be provided'
            )
        
        # Validate and normalize price and rate before processing
        self.price = self._normalize_to_scalar_if_single(self.price)
        self.rate = self._normalize_to_scalar_if_single(self.rate)
        
        # Get the current price (first element if list, otherwise the value itself)
        current_price = self.price[0] if isinstance(self.price, list) else self.price
        
        # Calculate the missing value if any
        if provided_values == 2:
            if self.balance is None:
                self.balance = current_price * self.unit
            elif self.price is None:
                if self.unit == 0:
                    raise ValueError('Cannot calculate price when unit is 0')
                self.price = self.balance / self.unit
            elif self.unit is None:
                if current_price == 0:
                    raise ValueError('Cannot calculate unit when price is 0')
                self.unit = self.balance / current_price
        
        if provided_values == 3:
            # Always verify the relationship holds using current price
            current_price = self.price[0] if isinstance(self.price, list) else self.price
            if abs(current_price * self.unit - self.balance) > 1e-10:
                raise ValueError(
                    f'price * unit must equal balance. '
                    f'Current price: {current_price}, unit: {self.unit}, balance: {self.balance}'
                )
        
        # Set book_balance to balance if not provided
        if self.book_balance is None:
            self.book_balance = self.balance
        
        # Validate and align price and rate
        self._process_price_rate_relationship()
            
        return self
    
    def _process_price_rate_relationship(self):
        """
        Validate and align price and rate based on the following rules:

        - Validate the relationship: price[i] * (1 + rate[i]) = price[i+1]
        
        Raises
        ------
        ValueError
            If price and rate list lengths are inconsistent or
            if the price growth relationship doesn't hold
        """
        price_is_list = isinstance(self.price, list)
        rate_is_list = isinstance(self.rate, list)
        
        if not price_is_list and not rate_is_list:
            return  # No processing needed when both are scalars
        elif rate_is_list and not price_is_list:  # rate is list, price is scalar
            self._generate_price_from_rate()
        elif price_is_list and not rate_is_list:  # price is list, rate is scalar
            if self.rate is None:
                # Calculate rate from price list
                self._generate_rate_from_price()
            else:
                # Validate consistency and generate rate list
                self._generate_rate_from_price()
        elif price_is_list and rate_is_list:  # both are lists
            self._validate_both_lists()

    def _normalize_to_scalar_if_single(self, value):
        """Convert single-element list to scalar, check for empty lists"""
        if not isinstance(value, list):
            return value
        
        if len(value) == 0:
            raise ValueError(f'list cannot be empty')
        elif len(value) == 1:
            return value[0]
        return value

    def _generate_price_from_rate(self):
        """Generate price list from scalar price and list of growth rates"""
        initial_price = self.price
        price_list = [initial_price]
        
        current_price = initial_price
        for rate in self.rate:
            current_price *= (1 + rate)
            price_list.append(current_price)
        
        self.price = price_list

    def _generate_rate_from_price(self):
        """Calculate and generate actual growth rate list from price list"""
        # Note: This method is only called when self.price is a list with 2+ elements
        # due to normalization in _normalize_to_scalar_if_single()
        
        # Calculate actual growth rates from price list
        calculated_rates = []
        for i in range(len(self.price) - 1):
            if self.price[i] == 0:
                raise ValueError(f'Cannot calculate rate when price[{i}] is 0')
            
            # Calculate rate[i] from price[i] * (1 + rate[i]) = price[i+1]
            # rate[i] = (price[i+1] / price[i]) - 1
            actual_rate = (self.price[i + 1] / self.price[i]) - 1
            calculated_rates.append(actual_rate)
        
        # If existing rate is scalar, check consistency with first calculated result
        if self.rate is not None and not isinstance(self.rate, list) and len(calculated_rates) > 0:
            tolerance = 1e-10
            first_calculated_rate = calculated_rates[0]
            
            if abs(self.rate - first_calculated_rate) > tolerance:
                raise ValueError(
                    f'Existing rate ({self.rate}) does not match calculated rate from first two prices '
                    f'({first_calculated_rate}). price[0]={self.price[0]}, price[1]={self.price[1]}'
                )
        
        # Set the calculated growth rate list
        self.rate = calculated_rates

    def _validate_both_lists(self):
        """Vectorized validation when using numpy"""
        
        price_array = np.array(self.price)
        rate_array = np.array(self.rate)
        
        # Vectorized calculations
        expected_prices = price_array[:-1] * (1 + rate_array)
        actual_prices = price_array[1:]
        
        # Check all differences at once
        differences = np.abs(expected_prices - actual_prices)
        tolerance = 1e-10
        
        if np.any(differences > tolerance):
            # Identify error details
            error_indices = np.where(differences > tolerance)[0]
            first_error = error_indices[0]
            raise ValueError(
                f'Price growth relationship violated at index {first_error}: '
                f'expected {expected_prices[first_error]}, got {actual_prices[first_error]}'
            )