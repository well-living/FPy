from typing import List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# fpyjp.utils.list_utilからの関数をインポート
from fpyjp.utils.list_util import ensure_list


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
    
    step: Optional[int] = Field(
        default=None,
        ge=1,
        description="Step size for cash flow periods"
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
        """
        if isinstance(v, (int, float)):
            if not np.isfinite(v):
                raise ValueError("Amount must be a finite number")
            return float(v)
        
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Amount list cannot be empty")
            
            # Convert to list of floats using utility function
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
        
        Parameters
        ----------
        v : int
            Sign value
            
        Returns
        -------
        int
            Validated sign value
            
        Raises
        ------
        ValueError
            If sign is not 1 or -1
        """
        if v not in [-1, 1]:
            raise ValueError("Sign must be -1 (outflow) or 1 (inflow)")
        return v
    
    @model_validator(mode='after')
    def validate_period_consistency(self):
        """
        Validate consistency between period-related fields and calculate n_periods.
        
        - If n_periods is None, calculate it from len(self.amount) if amount is a list
        - If n_periods is specified, validate it matches len(self.amount) if amount is a list
        - Ensures that start <= end when both are specified
        
        Returns
        -------
        CashflowSchema
            Self after validation
            
        Raises
        ------
        ValueError
            If period fields are inconsistent
        """
        start = self.start
        end = self.end
        n_periods = self.n_periods
        step = self.step
        
        # Check start <= end
        if start is not None and end is not None:
            if start > end:
                raise ValueError("End period must be greater than or equal to start period")
        
        # Handle n_periods calculation and validation
        if isinstance(self.amount, list):
            amount_length = len(self.amount)
            if n_periods is None:
                # Calculate n_periods from amount list length
                self.n_periods = amount_length
            else:
                # Validate n_periods matches amount list length
                if n_periods != amount_length:
                    raise ValueError(
                        f"n_periods ({n_periods}) must match length of amount list ({amount_length})"
                    )
        else:
            # amount is scalar - n_periods remains as specified or None
            pass
        
        # Validate step with start/end
        if step is not None and start is not None and end is not None:
            expected_periods = (end - start) // step + 1
            if self.n_periods is not None and self.n_periods != expected_periods:
                raise ValueError(
                    f"n_periods ({self.n_periods}) is inconsistent with calculated "
                    f"periods from start/end/step ({expected_periods})"
                )
        
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