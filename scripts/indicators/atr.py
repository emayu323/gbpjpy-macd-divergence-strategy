"""
ATR (Average True Range) インジケーター
Wilder方式で計算
"""

import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR (Average True Range) を計算（Wilder方式）

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（'High', 'Low', 'Close'列を含む）
    period : int
        ATR期間（デフォルト: 14）

    Returns:
    --------
    pd.Series
        ATR値
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr
