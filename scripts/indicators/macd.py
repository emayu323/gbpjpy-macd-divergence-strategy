"""
MACD (Moving Average Convergence Divergence) インジケーター
"""

import pandas as pd
from typing import Tuple


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = 'Close'
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence) を計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ
    fast : int
        短期EMA期間（デフォルト: 12）
    slow : int
        長期EMA期間（デフォルト: 26）
    signal : int
        シグナルライン期間（デフォルト: 9）
    price_col : str
        価格カラム名（デフォルト: 'Close'）

    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series]
        (MACD Line, Signal Line, Histogram)
    """
    # 短期・長期EMAを計算
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    # MACD Line = 短期EMA - 長期EMA
    macd_line = ema_fast - ema_slow

    # Signal Line = MACD LineのEMA
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram = MACD Line - Signal Line
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
