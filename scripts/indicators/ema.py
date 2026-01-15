"""
EMA (Exponential Moving Average) インジケーター
"""

import pandas as pd


def calculate_ema(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.Series:
    """
    EMA (Exponential Moving Average) を計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ
    period : int
        EMA期間
    price_col : str
        価格カラム名（デフォルト: 'Close'）

    Returns:
    --------
    pd.Series
        EMA値
    """
    return df[price_col].ewm(span=period, adjust=False).mean()
