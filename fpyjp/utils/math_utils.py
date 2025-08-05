# fpyjp/utils/math_ulils.py

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely performs division with zero-division and type checks.

    Parameters
    ----------
    numerator : float or int
        The numerator of the division. Must be a numeric type.
    denominator : float or int
        The denominator of the division. Must be a numeric type.
    default : float or int, optional
        The value to return if division cannot be performed safely (e.g., zero denominator).
        Default is 0.0.

    Returns
    -------
    float
        The result of numerator divided by denominator, or the default value if the denominator
        is too close to zero.

    Raises
    ------
    TypeError
        If any of the arguments is not a float or int.
    """
    if not isinstance(numerator, (int, float)):
        raise TypeError(f"numerator must be int or float, got {type(numerator).__name__}")
    if not isinstance(denominator, (int, float)):
        raise TypeError(f"denominator must be int or float, got {type(denominator).__name__}")
    if not isinstance(default, (int, float)):
        raise TypeError(f"default must be int or float, got {type(default).__name__}")

    if abs(denominator) < 1e-10:
        return float(default)

    return float(numerator) / float(denominator)
