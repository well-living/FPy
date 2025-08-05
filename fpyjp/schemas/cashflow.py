# fpyjp.schemas.cashflow.py
from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from fpyjp.utils.list_utils import (
    ensure_list,  
    validate_position_within_length, 
    validate_start_end_consistency, 
    pad_list,
)


class CashflowSchema(BaseModel):
    """
    Schema for cash flow items.
    
    Represents a cash flow item with a name, amount, sign, and optional 
    start and end periods for financial simulations.
    
    Attributes
    ----------
    name : Optional[str]
        Name of the cash flow item, must be between 1 and 20 characters if specified.
    amount : Union[float, List[float]]
        Cash flow amount(s). Can be a single value or a list of values.
        If list has length 1, it will be converted to scalar.
    sign : int
        Direction of cash flow. 1 for inflow, -1 for outflow.
    n_periods : Optional[int]
        Number of periods for the cash flow. Must be positive if specified.
    start : Optional[int]
        Start period for cash flow (0-based index). Must be non-negative if specified.
    end : Optional[int]
        End period for cash flow (0-based index). Must be non-negative if specified.
    step : Optional[int]
        Step size for cash flow periods. Must be positive if specified.
        
    Examples
    --------
    >>> cf = CashflowSchema(
    ...     name="Salary",
    ...     amount=5000.0,
    ...     sign=1,
    ...     start=0,
    ...     end=120,
    ...     step=1
    ... )
    >>> cf.name
    'Salary'
    >>> cf.amount
    5000.0
    
    >>> cf_list = CashflowSchema(
    ...     name="Bonus",
    ...     amount=[10000.0, 12000.0, 15000.0],
    ...     sign=1
    ... )
    >>> cf_list.amount
    [10000.0, 12000.0, 15000.0]
    
    >>> cf_unnamed = CashflowSchema(
    ...     amount=1000.0,
    ...     sign=-1
    ... )
    >>> cf_unnamed.name is None
    True
    """
    
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="Name of the cash flow item"
    )
    
    amount: Union[float, List[float]] = Field(
        default=0.0,
        description="Cash flow amount(s). Single value or list of values."
    )
    
    sign: int = Field(
        default=1,
        description="1 for inflow, -1 for outflow"
    )
    
    n_periods: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of periods for the cash flow"
    )
    
    start: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start period for cash flow (0-based index)"
    )
    
    end: Optional[int] = Field(
        default=None,
        ge=0,
        description="End period for cash flow (0-based index)"
    )
    

    
    @field_validator('amount')
    @classmethod
    def validate_and_convert_amount(cls, v):
        """
        Validate and convert amount field.
        
        Converts single-element lists to scalars and ensures all values
        are finite numbers.
        
        Parameters
        ----------
        v : Union[float, List[float]]
            Input amount value(s)
            
        Returns
        -------
        Union[float, List[float]]
            Validated and possibly converted amount
            
        Raises
        ------
        ValueError
            If amount contains non-finite values or empty list

        Examples
        --------
        >>> # Single number input
        >>> validate_and_convert_amount(100.0)
        100.0

        >>> # Single-element list converted to scalar
        >>> validate_and_convert_amount([500.0])
        500.0

        >>> # Multiple-element list preserved as list
        >>> validate_and_convert_amount([100.0, 200.0, 300.0])
        [100.0, 200.0, 300.0]

        >>> # Mixed int/float list converted to float list
        >>> validate_and_convert_amount([100, 200.5, 300])
        [100.0, 200.5, 300.0]

        >>> # Invalid: empty list raises error
        >>> validate_and_convert_amount([])
        ValueError: Amount list cannot be empty

        >>> # Invalid: non-finite values raise error
        >>> validate_and_convert_amount([100.0, float('inf')])
        ValueError: All amount values must be finite numbers
        """
        if isinstance(v, (int, float)):
            if not np.isfinite(v):
                raise ValueError("Amount must be a finite number")
            return float(v)
        
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Amount list cannot be empty")
            
            # Convert all list elements to float type for consistency
            # Note: ensure_list() converts element types, not list structure
            amount_list = ensure_list(v)
            
            # Check for finite values
            if not all(np.isfinite(val) for val in amount_list):
                raise ValueError("All amount values must be finite numbers")
            
            # Convert single-element list to scalar
            if len(amount_list) == 1:
                return amount_list[0]
            
            return amount_list
        
        raise ValueError("Amount must be a number or list of numbers")
    
    @field_validator('sign')
    @classmethod
    def validate_sign(cls, v):
        """
        Validate sign field.
        
        This validator runs after any Field constraints (ge, le, etc.).
        If Field constraints fail, their error messages take precedence
        and this validator will not be executed.
        
        Parameters
        ----------
        v : int
            Sign value that has already passed Field validation
            
        Returns
        -------
        int
            Validated sign value (must be -1 or 1)
            
        Raises
        ------
        ValueError
            If sign is not -1 (outflow) or 1 (inflow)
            
        Notes
        -----
        Validation order in Pydantic:
        1. Field constraints (ge, le, etc.) - if any exist
        2. Custom field validators (this method)
        
        Examples
        --------
        >>> # Valid inflow
        >>> validate_sign(1)
        1
        
        >>> # Valid outflow  
        >>> validate_sign(-1)
        -1
        
        >>> # Invalid value
        >>> validate_sign(0)
        ValueError: Sign must be -1 (outflow) or 1 (inflow)
        """
        if v not in [-1, 1]:
            raise ValueError("Sign must be -1 (outflow) or 1 (inflow)")
        return v

    def _validate_all_periods(self, start: Optional[int], end: Optional[int], n_periods: Optional[int]) -> None:
        """
        Validate all period-related constraints in one method.
        
        Performs the following validations:
        1. start < n_periods (if both specified)
        2. end < n_periods (if both specified)  
        3. start <= end (if both specified)
        
        Parameters
        ----------
        start : Optional[int]
            Start position (0-based index)
        end : Optional[int]
            End position (0-based index)
        n_periods : Optional[int]
            Total number of periods
            
        Raises
        ------
        ValueError
            If any period constraint is violated
        """
        # Validate positions are within n_periods
        self.validate_position_within_length(start, "start", n_periods)
        self.validate_position_within_length(end, "end", n_periods)
        
        # Validate start <= end consistency
        self.validate_start_end_consistency(start, end)
    
    def _calculate_effective_end(self, amount_list: List[float]) -> Optional[int]:
        """
        Calculate the effective end position for a list of amounts.
        
        Returns the index of the last non-zero value, or None if all values are zero
        (indicating no actual cash flow activity).
        
        Parameters
        ----------
        amount_list : List[float]
            List of amount values
            
        Returns
        -------
        Optional[int]
            Effective end position (0-based index), or None if all values are zero
            
        Examples
        --------
        >>> _calculate_effective_end([100, 200, 0, 0])  # Returns 1
        >>> _calculate_effective_end([100, 200, 300])   # Returns 2
        >>> _calculate_effective_end([0, 0, 0])         # Returns None (no activity)
        >>> _calculate_effective_end([0, 100, 0])       # Returns 1
        """
        # Find the last non-zero value
        for i in range(len(amount_list) - 1, -1, -1):
            if amount_list[i] != 0.0:
                return i
        
        # If all values are zero, return None (no cash flow activity)
        return None

    @model_validator(mode='after')
    def validate_period_consistency(self):
        """
        Process and validate cash flow parameters after model initialization.

        Validation and processing order:
        1. If n_periods is None, calculate it from len(self.amount) if amount is a list
        2. Validate all period constraints (start, end positions and consistency)
        3. Process amount adjustments based on n_periods
        4. Handle end period truncation with zero padding

        Returns
        -------
        CashflowSchema
            Self after validation with adjusted amount

        Raises
        ------
        ValueError
            If start >= n_periods or end >= n_periods (invalid 0-based indexing)
            If start > end when both are specified
            If period fields are inconsistent
        """
        
        n_periods = self.n_periods
        start = self.start
        end = self.end
        
        # 1. Calculate n_periods if None
        if isinstance(self.amount, list) and n_periods is None:
            self.n_periods = len(self.amount)
            n_periods = self.n_periods
        elif not isinstance(self.amount, list) and n_periods is None:
            # amount is scalar, set n_periods to 1
            self.n_periods = 1
            n_periods = 1
        
        # 2. Validate all period constraints
        self._validate_all_periods(start, end, n_periods)
        
        # 3. Handle amount adjustment based on n_periods
        if isinstance(self.amount, list):
            amount_length = len(self.amount)
            
            # n_periods is now guaranteed to be set
            if start is not None:
                # Use pad_list to handle both truncation and padding with start position
                self.amount = pad_list(self.amount, n_periods, start=start)
            else:
                # No start specified - handle truncation/padding from position 0
                if n_periods < amount_length:
                    # Truncate amount list
                    self.amount = self.amount[:n_periods]
                    self.start = 0  # Reset start to 0 if truncating
                elif n_periods > amount_length:
                    # Pad amount list from position 0
                    self.amount = pad_list(self.amount, n_periods, start=0)
                    self.start = 0  # Reset start to 0 if padding
                # If n_periods == amount_length, no change needed
        else:
            # amount is scalar
            if n_periods >= 2:
                # Convert scalar to list
                if start is not None:
                    # Create list with scalar value at start position, zero-pad before and after
                    self.amount = pad_list([self.amount], n_periods, start=start)
                else:
                    # Create list with scalar value repeated from position 0
                    self.amount = [self.amount] * n_periods
                    self.start = 0  # Reset start to 0 if scalar
            # If n_periods == 1, keep as scalar (no change needed)
        
        # 4. Handle end period truncation if specified
        if end is not None and isinstance(self.amount, list):
            current_length = len(self.amount)
            # Zero-fill positions after end (but keep the array length)
            for i in range(end + 1, current_length):
                self.amount[i] = 0.0
        
        # 5. Set effective end position if end is None
        if end is None and isinstance(self.amount, list):
            self.end = self._calculate_effective_end(self.amount)
        elif end is None and not isinstance(self.amount, list):
            # For scalar amounts, end is always 0
            self.end = 0
        
        return self



    def get_periods_count(self) -> Optional[int]:
        """
        Get the total number of periods for this cash flow.
        
        Returns
        -------
        Optional[int]
            Number of periods, or None if cannot be determined
            
        Examples
        --------
        >>> cf = CashflowSchema(start=0, end=10, step=2)
        >>> cf.get_periods_count()
        6
        
        >>> cf = CashflowSchema(n_periods=5)
        >>> cf.get_periods_count()
        5
        
        >>> cf = CashflowSchema(amount=[100, 200, 300])
        >>> cf.get_periods_count()
        3
        """
        return self.n_periods
    
    def get_amount_at_period(self, period: int) -> float:
        """
        Get the cash flow amount for a specific period.
        
        Parameters
        ----------
        period : int
            Period index (0-based)
            
        Returns
        -------
        float
            Cash flow amount for the specified period
            
        Examples
        --------
        >>> cf = CashflowSchema(amount=[100, 200, 300])
        >>> cf.get_amount_at_period(1)
        200.0
        >>> cf.get_amount_at_period(5)
        0.0
        
        >>> cf_scalar = CashflowSchema(amount=500.0, n_periods=3)
        >>> cf_scalar.get_amount_at_period(0)
        500.0
        >>> cf_scalar.get_amount_at_period(2)
        500.0
        >>> cf_scalar.get_amount_at_period(5)
        0.0
        
        >>> cf_single = CashflowSchema(amount=1000.0)
        >>> cf_single.get_amount_at_period(0)
        1000.0
        >>> cf_single.get_amount_at_period(1)
        0.0
        """
        if isinstance(self.amount, list):
            if 0 <= period < len(self.amount):
                return self.amount[period]
            return 0.0
        else:
            # スカラーの場合の処理
            n_periods = self.get_periods_count()
            if n_periods is None or n_periods == 1:
                # n_periodsが指定されていない、または1の場合は期間0のみ
                return self.amount if period == 0 else 0.0
            else:
                # n_periodsが指定されている場合はその期間内で同じ値を返す
                return self.amount if 0 <= period < n_periods else 0.0
    
    def to_padded_array(self, total_periods: int, start_period: Optional[int] = None) -> List[float]:
        """
        Convert cash flow to padded array.
        
        This method creates a padded array representation of the cash flow,
        useful for time-series calculations and simulations.
        
        Parameters
        ----------
        total_periods : int
            Total length of the output array
        start_period : Optional[int]
            Starting position for the cash flow. If None, uses self.start
            
        Returns
        -------
        List[float]
            Padded array of cash flow amounts
            
        Raises
        ------
        ValueError
            If required parameters are missing
            
        Examples
        --------
        >>> cf = CashflowSchema(amount=1000.0, start=2, n_periods=3)
        >>> cf.to_padded_array(10)
        [0.0, 0.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        >>> cf = CashflowSchema(amount=[500.0, 600.0], start=1)
        >>> cf.to_padded_array(5)
        [0.0, 500.0, 600.0, 0.0, 0.0]
        """
        # Determine start position
        start_pos = start_period if start_period is not None else self.start
        if start_pos is None:
            start_pos = 0  # デフォルトは0から開始
        
        # Create array of cash flow amounts
        if isinstance(self.amount, list):
            amount_values = self.amount
        else:
            # スカラーの場合
            if self.n_periods is None:
                # n_periodsが指定されていない場合は1期間とする
                amount_values = [self.amount]
            else:
                amount_values = [self.amount] * self.n_periods
        
        # Create padded array
        result = [0.0] * total_periods
        
        # Fill in the cash flow values at appropriate positions
        for i, value in enumerate(amount_values):
            pos = start_pos + i
            if 0 <= pos < total_periods:
                result[pos] = value
        
        return result