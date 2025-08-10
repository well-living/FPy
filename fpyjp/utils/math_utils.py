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
        If any of the arguments is not a float or int. All type errors are reported together.
    """
    invalid_args = []
    for name, value in [('numerator', numerator), ('denominator', denominator), ('default', default)]:
        if not isinstance(value, (int, float)):
            invalid_args.append(f"{name} must be int or float, got {type(value).__name__}")

    if invalid_args:
        raise TypeError(" | ".join(invalid_args))

    if abs(denominator) < 1e-10:
        return float(default)

    return float(numerator) / float(denominator)

