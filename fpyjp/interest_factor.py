from typing import Union

from pydantic import BaseModel, Field, field_validator


class InterestFactor(BaseModel):
    """
    A class for calculating various interest factors used in financial analysis.
    
    This class provides methods to compute compound interest factors including
    future value, present value, annuity factors, and related calculations.
    
    Attributes
    ----------
    rate : float
        Interest rate per period (must be greater than -1)
    time_period : int
        Number of time periods (must be 1 or greater)
    amount : float, default=1.0
        Principal amount or base value for calculations
    
    Examples
    --------
    >>> factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
    >>> fv = factor.future_value_factor()
    >>> print(f"Future value factor: {fv:.4f}")
    """
    
    rate: float = Field(..., gt=-1.0, description="Interest rate per period (must be > -1)")
    time_period: int = Field(..., ge=1, description="Number of time periods (must be >= 1)")
    amount: float = Field(default=1.0, description="Principal amount or base value")
    
    @field_validator('rate')
    @classmethod
    def validate_rate(cls, v: float) -> float:
        """Validate that rate is greater than -1."""
        if v <= -1.0:
            raise ValueError("Rate must be greater than -1")
        return v
    
    def future_value_factor(self) -> float:
        """
        Calculate the future value factor (F/P).
        
        This factor converts a present value to its equivalent future value
        using compound interest.
        
        Returns
        -------
        float
            Future value factor: (1 + rate)^time_period
            
        Notes
        -----
        The future value factor is calculated as:
        F/P = (1 + i)^n
        
        where i is the interest rate and n is the number of periods.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10)
        >>> fv_factor = factor.future_value_factor()
        >>> print(f"Future value factor: {fv_factor:.4f}")
        """
        return (1 + self.rate) ** self.time_period
    
    def calculate_future_value(self) -> float:
        """
        Calculate the future value (F/P).
        
        This method converts a present value to its equivalent future value
        using compound interest, multiplied by the amount.
        
        Returns
        -------
        float
            Future value: future_value_factor() * amount
            
        Notes
        -----
        The future value is calculated as:
        F = P * (1 + i)^n
        
        where P is the amount, i is the interest rate and n is the number of periods.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        >>> fv = factor.calculate_future_value()
        >>> print(f"Future value: {fv:.2f}")
        """
        return self.future_value_factor() * self.amount
    
    def present_value_factor(self) -> float:
        """
        Calculate the present value factor (P/F).
        
        This factor converts a future value to its equivalent present value
        using compound interest discounting.
        
        Returns
        -------
        float
            Present value factor: 1 / (1 + rate)^time_period
            
        Notes
        -----
        The present value factor is calculated as:
        P/F = 1 / (1 + i)^n
        
        This is the reciprocal of the future value factor.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10)
        >>> pv_factor = factor.present_value_factor()
        >>> print(f"Present value factor: {pv_factor:.4f}")
        """
        return 1 / ((1 + self.rate) ** self.time_period)
    
    def calculate_present_value(self) -> float:
        """
        Calculate the present value (P/F).
        
        This method converts a future value to its equivalent present value
        using compound interest discounting, multiplied by the amount.
        
        Returns
        -------
        float
            Present value: present_value_factor() * amount
            
        Notes
        -----
        The present value is calculated as:
        P = F / (1 + i)^n
        
        This is the reciprocal of the future value calculation.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        >>> pv = factor.calculate_present_value()
        >>> print(f"Present value: {pv:.2f}")
        """
        return self.present_value_factor() * self.amount
    
    def future_value_of_annuity_factor(self) -> float:
        """
        Calculate the future value of annuity factor (F/A).
        
        This factor converts a uniform series of payments to its equivalent
        future value at the end of the series.
        
        Returns
        -------
        float
            Future value of annuity factor: ((1 + rate)^time_period - 1) / rate
            
        Raises
        ------
        ValueError
            If rate is zero (division by zero)
            
        Notes
        -----
        The future value of annuity factor is calculated as:
        F/A = ((1 + i)^n - 1) / i
        
        For zero interest rate, the factor equals the number of periods.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10)
        >>> fva_factor = factor.future_value_of_annuity_factor()
        >>> print(f"Future value of annuity factor: {fva_factor:.4f}")
        """
        if self.rate == 0:
            return float(self.time_period)
        return ((1 + self.rate) ** self.time_period - 1) / self.rate
    
    def calculate_future_value_of_annuity(self) -> float:
        """
        Calculate the future value of annuity (F/A).
        
        This method converts a uniform series of payments to its equivalent
        future value at the end of the series, multiplied by the amount.
        
        Returns
        -------
        float
            Future value of annuity: future_value_of_annuity_factor() * amount
            
        Notes
        -----
        The future value of annuity is calculated as:
        F = A * ((1 + i)^n - 1) / i
        
        For zero interest rate, the factor equals the number of periods.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10, amount=100)
        >>> fva = factor.calculate_future_value_of_annuity()
        >>> print(f"Future value of annuity: {fva:.2f}")
        """
        return self.future_value_of_annuity_factor() * self.amount
    
    def sinking_fund_factor(self) -> float:
        """
        Calculate the sinking fund factor (A/F).
        
        This factor determines the uniform series payment required to
        accumulate to a specified future value.
        
        Returns
        -------
        float
            Sinking fund factor: rate / ((1 + rate)^time_period - 1)
            
        Raises
        ------
        ValueError
            If rate is zero (division by zero)
            
        Notes
        -----
        The sinking fund factor is calculated as:
        A/F = i / ((1 + i)^n - 1)
        
        This is the reciprocal of the future value of annuity factor.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10)
        >>> sf_factor = factor.sinking_fund_factor()
        >>> print(f"Sinking fund factor: {sf_factor:.4f}")
        """
        if self.rate == 0:
            return 1.0 / self.time_period
        return self.rate / ((1 + self.rate) ** self.time_period - 1)
    
    def calculate_sinking_fund(self) -> float:
        """
        Calculate the sinking fund payment (A/F).
        
        This method determines the uniform series payment required to
        accumulate to a specified future value, based on the amount.
        
        Returns
        -------
        float
            Sinking fund payment: sinking_fund_factor() * amount
            
        Notes
        -----
        The sinking fund payment is calculated as:
        A = F * i / ((1 + i)^n - 1)
        
        This is the reciprocal of the future value of annuity factor.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        >>> sf = factor.calculate_sinking_fund()
        >>> print(f"Sinking fund payment: {sf:.2f}")
        """
        return self.sinking_fund_factor() * self.amount
    
    def capital_recovery_factor(self) -> float:
        """
        Calculate the capital recovery factor (A/P).
        
        This factor determines the uniform series payment equivalent to
        a given present value over a specified number of periods.
        
        Returns
        -------
        float
            Capital recovery factor: rate * (1 + rate)^time_period / ((1 + rate)^time_period - 1)
            
        Raises
        ------
        ValueError
            If rate is zero (division by zero)
            
        Notes
        -----
        The capital recovery factor is calculated as:
        A/P = i * (1 + i)^n / ((1 + i)^n - 1)
        
        This is the reciprocal of the present value of annuity factor.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10)
        >>> cr_factor = factor.capital_recovery_factor()
        >>> print(f"Capital recovery factor: {cr_factor:.4f}")
        """
        if self.rate == 0:
            return 1.0 / self.time_period
        return (self.rate * (1 + self.rate) ** self.time_period) / ((1 + self.rate) ** self.time_period - 1)
    
    def calculate_capital_recovery(self) -> float:
        """
        Calculate the capital recovery payment (A/P).
        
        This method determines the uniform series payment equivalent to
        a given present value over a specified number of periods, based on the amount.
        
        Returns
        -------
        float
            Capital recovery payment: capital_recovery_factor() * amount
            
        Notes
        -----
        The capital recovery payment is calculated as:
        A = P * i * (1 + i)^n / ((1 + i)^n - 1)
        
        This is the reciprocal of the present value of annuity factor.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10, amount=1000)
        >>> cr = factor.calculate_capital_recovery()
        >>> print(f"Capital recovery payment: {cr:.2f}")
        """
        return self.capital_recovery_factor() * self.amount
    
    def present_value_of_annuity_factor(self) -> float:
        """
        Calculate the present value of annuity factor (P/A).
        
        This factor converts a uniform series of payments to its equivalent
        present value at the beginning of the series.
        
        Returns
        -------
        float
            Present value of annuity factor: ((1 + rate)^time_period - 1) / (rate * (1 + rate)^time_period)
            
        Raises
        ------
        ValueError
            If rate is zero (division by zero)
            
        Notes
        -----
        The present value of annuity factor is calculated as:
        P/A = ((1 + i)^n - 1) / (i * (1 + i)^n)
        
        For zero interest rate, the factor equals the number of periods.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10)
        >>> pva_factor = factor.present_value_of_annuity_factor()
        >>> print(f"Present value of annuity factor: {pva_factor:.4f}")
        """
        if self.rate == 0:
            return float(self.time_period)
        return ((1 + self.rate) ** self.time_period - 1) / (self.rate * (1 + self.rate) ** self.time_period)
    
    def calculate_present_value_of_annuity(self) -> float:
        """
        Calculate the present value of annuity (P/A).
        
        This method converts a uniform series of payments to its equivalent
        present value at the beginning of the series, based on the amount.
        
        Returns
        -------
        float
            Present value of annuity: present_value_of_annuity_factor() * amount
            
        Notes
        -----
        The present value of annuity is calculated as:
        P = A * ((1 + i)^n - 1) / (i * (1 + i)^n)
        
        For zero interest rate, the factor equals the number of periods.
        
        Examples
        --------
        >>> factor = InterestFactor(rate=0.05, time_period=10, amount=100)
        >>> pva = factor.calculate_present_value_of_annuity()
        >>> print(f"Present value of annuity: {pva:.2f}")
        """
        return self.present_value_of_annuity_factor() * self.amount

