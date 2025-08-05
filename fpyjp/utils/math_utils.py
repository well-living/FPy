
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全な除算（ゼロ除算対策）"""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator
