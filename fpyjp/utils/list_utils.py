# fpyjp/utils/list_utils.py
"""
Array utility functions for handling array operations and padding.

This module provides utility functions for array manipulation, padding, and value extraction
that are commonly used across financial simulation modules.
"""

from typing import Union, Optional, List

def ensure_list(
        value: Union[int, float, List[Union[int, float]]]
    ) -> List[float]:
    """
    Ensure a value is converted to a list of floats.
    
    This utility function converts scalar values to single-element lists
    and ensures all values are of float type.
    
    Parameters
    ----------
    value : Union[int, float, List[Union[int, float]]]
        Input value that can be scalar or a list of numbers.
        
    Returns
    -------
    List[float]
        List representation of the input value with float elements.
        
    Examples
    --------
    >>> ensure_list(5)
    [5.0]
    >>> ensure_list(3.14)
    [3.14]
    >>> ensure_list([1, 2, 3])
    [1.0, 2.0, 3.0]
    >>> ensure_list([1.5, 2.5])
    [1.5, 2.5]
    """
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def pad_list(
        values: List[float], 
        n_length: int, 
        start: int = 0, 
        pad_mode: str = 'zero'
    ) -> List[float]:
    """
    Pad an list to a specified length.
    
    Parameters
    ----------
    values : List[float]
        The input list to be padded.
    n_length : int
        The target length of the output list.
    start : int, optional
        The starting position where the input list should be placed in the output array.
        If negative, it is treated as an offset from the end.
        Default is 0.
    pad_mode : {'zero', 'last'}, optional
        The padding mode for filling empty positions:
        - 'zero' : Fill with zeros (default)
        - 'last' : Fill trailing positions with the last value from the input array
        Default is 'zero'.
    
    Returns
    -------
    List[float]
        The padded list of length `n_length`.
    
    Examples
    --------
    >>> pad_list([1.0, 2.0, 3.0], 5)
    [1.0, 2.0, 3.0, 0.0, 0.0]
    
    >>> pad_list([1.0, 2.0, 3.0], 5, start=1)
    [0.0, 1.0, 2.0, 3.0, 0.0]
    
    >>> pad_list([1.0, 2.0, 3.0], 5, start=-3)
    [0.0, 0.0, 1.0, 2.0, 3.0]
    
    >>> pad_list([1.0, 2.0, 3.0], 5, pad_mode='last')
    [1.0, 2.0, 3.0, 3.0, 3.0]
    """
    if n_length <= 0:
        return []
    
    if len(values) == 0:
        return [0.0] * n_length
    
    # Initialize result array
    result = [0.0] * n_length
    
    # Handle negative start values
    if start < 0:
        # Special case: start = -len(values) means place array at the end
        if start == -len(values):
            start = n_length - len(values)
        else:
            start = max(0, n_length + start)
    
    # Place original values at appropriate positions
    for i, value in enumerate(values):
        pos = start + i
        if 0 <= pos < n_length:
            result[pos] = value
    
    # Handle trailing padding when pad_mode='last'
    if pad_mode == 'last' and len(values) > 0:
        last_value = values[-1]
        # Find the position of the last element from the original array
        last_pos = -1
        for i in range(len(values) - 1, -1, -1):
            pos = start + i
            if 0 <= pos < n_length:
                last_pos = pos
                break
        
        # Fill positions after the last element with the last value
        if last_pos >= 0:
            for i in range(last_pos + 1, n_length):
                result[i] = last_value
    
    return result


def get_padded_value_at_period(
        values: List[float],
        period: int,
        n_length: Optional[int] = None,
        start: int = 0,
        pad_mode: str = 'zero'
    ) -> float:
    """
    Get value for a specific period with configurable out-of-bounds handling and array alignment.
    
    This function provides efficient access to array values with padding support, without
    actually creating the full padded array. It's particularly useful for time-series
    simulations where you need to access values at specific periods.
    
    Parameters
    ----------
    values : List[float]
        List of values indexed by period.
    period : int
        Target period index (0-based).
    n_length : Optional[int], optional
        The assumed total length of the padded array.
        Required when start < 0 for accurate results.
        If None and start < 0, a ValueError will be raised.
        Default is None.
    start : int, optional
        The starting position where the input array should be placed.
        If negative, it is treated as an offset from the end.
        Default is 0.
    pad_mode : {'zero', 'last'}, optional
        The padding mode for filling empty positions:
        - 'zero' : Fill with zeros (default)
        - 'last' : Fill trailing positions with the last value from the input array
        Default is 'zero'.
        
    Returns
    -------
    float
        Value for the specified period.
        
    Raises
    ------
    ValueError
        If start < 0 and n_length is None.
        
    Examples
    --------
    Basic usage with positive start:
    
    >>> values = [1.0, 2.0, 3.0]
    >>> get_padded_value_at_period(values, 1)
    2.0
    >>> get_padded_value_at_period(values, 3)
    0.0
    
    Usage with offset start:
    
    >>> get_padded_value_at_period(values, 1, start=1)
    1.0
    >>> get_padded_value_at_period(values, 0, start=1)
    0.0
    
    Usage with negative start (requires n_length):
    
    >>> get_padded_value_at_period(values, 3, n_length=5, start=-3)
    2.0
    >>> get_padded_value_at_period(values, 1, n_length=5, start=-3)
    0.0
    
    Usage with 'last' padding mode:
    
    >>> get_padded_value_at_period(values, 4, pad_mode='last')
    3.0
    >>> get_padded_value_at_period(values, 5, n_length=10, pad_mode='last')
    3.0
    """
    if not values or period < 0:
        return 0.0
    
    # start < 0の場合はn_lengthが必須
    if start < 0:
        if n_length is None:
            raise ValueError("n_length is required when start < 0")
        if period >= n_length or n_length <= 0:
            return 0.0
    
    # 正のstartまたはn_lengthが提供されている場合
    actual_start = start
    if start < 0:
        if start == -len(values):
            actual_start = n_length - len(values)
        else:
            actual_start = max(0, n_length + start)
    
    # Calculate the position in the original values array
    pos_in_values = period - actual_start
    
    if 0 <= pos_in_values < len(values):
        return values[pos_in_values]
    elif pos_in_values < 0:
        return 0.0
    else:
        # Position is after the end of values array
        if pad_mode == 'last':
            return values[-1]
        else:
            return 0.0
