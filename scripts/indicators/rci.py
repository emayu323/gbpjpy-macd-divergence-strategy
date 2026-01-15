"""
RCI (Rank Correlation Index) インジケーター
PineScriptと完全に一致するロジック
"""

import numpy as np
import pandas as pd


def calculate_rci(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.Series:
    """
    RCI (Rank Correlation Index) を計算
    PineScriptと完全に一致するロジック

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（'Close'列を含む）
    period : int
        RCI期間
    price_col : str
        価格カラム名（デフォルト: 'Close'）

    Returns:
    --------
    pd.Series
        RCI値（-100〜+100）
    """
    rci_values = []
    for i in range(len(df)):
        if i < period - 1:
            rci_values.append(np.nan)
            continue

        # PineScriptと同じ順序でデータ取得（古い→新しい）
        prices = df[price_col].iloc[i - period + 1:i + 1].values

        sum_d2 = 0.0
        for idx in range(period):
            price_i = prices[idx]  # 古い順（idx=0が最も古い）

            # 価格ランクを計算（PineScript方式：平均順位）
            less = 0
            equal = 0
            for j in range(period):
                price_j = prices[j]
                if price_j < price_i:
                    less += 1
                elif price_j == price_i:
                    equal += 1

            rank_price = 1 + less + (equal - 1) / 2.0  # 平均順位
            rank_time = idx + 1  # 時間ランク（1からperiodまで）

            d = rank_time - rank_price
            sum_d2 += d * d

        n = period
        rci = (1 - (6 * sum_d2) / (n * (n * n - 1))) * 100
        rci_values.append(rci)

    return pd.Series(rci_values, index=df.index, name=f'RCI_{period}')
