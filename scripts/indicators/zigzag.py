"""
ZigZag インジケーター
sukepoyo_sub.pine のロジックに準拠
暫定ピボット（prospective pivot）対応版
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_zigzag_with_prospective(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    ZigZag インジケーターを計算（暫定ピボット対応版）
    sukepoyo_sub.pine のZigZagロジックに準拠

    Ver15: 確定ピボットに加えて、各バーでの暫定先端も記録
    - 上昇方向: 暫定先端 = 現在形成中の安値（次の谷候補）
    - 下降方向: 暫定先端 = 現在形成中の高値（次の山候補）

    sukepoyo_sub.pine準拠ポイント:
    - highestbars / lowestbars による窓内の極値検出
    - barssince シミュレーションによる方向判定
    - ピボット更新は厳密比較（> / <）を使用（>= / <= ではない）

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（'High', 'Low'列を含む）
    depth : int
        ZigZag深度（デフォルト: 12）
    deviation : float
        偏差閾値（デフォルト: 5.0）
    backstep : int
        バックステップ（デフォルト: 3）

    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]
        (確定高値, 確定安値, 暫定高値先端, 暫定安値先端, 暫定高値インデックス, 暫定安値インデックス)
    """
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    # 確定ピボット
    zz_highs = pd.Series(np.nan, index=df.index)
    zz_lows = pd.Series(np.nan, index=df.index)

    # 暫定ピボット（各バーでの最新先端）
    prospective_high = pd.Series(np.nan, index=df.index)
    prospective_low = pd.Series(np.nan, index=df.index)
    prospective_high_idx = pd.Series(np.nan, index=df.index)
    prospective_low_idx = pd.Series(np.nan, index=df.index)

    # ZigZag方向（1=上昇、-1=下降）
    zz_direction = pd.Series(0, index=df.index)

    # deviation閾値
    dev_th = deviation * 0.01  # GBPJPYの場合、mintick = 0.01

    # 方向追跡用の変数
    prev_dir = 0
    hr_since = 0
    lr_since = 0
    dir_change_since = 0

    # ピボット追跡（A=現在形成中、B=直近確定候補）
    zz_A_price = lows[0] if n > 0 else 0
    zz_A_idx = 0
    zz_B_price = lows[0] if n > 0 else 0
    zz_B_idx = 0

    for i in range(n):
        if i < depth:
            continue

        # highestbars / lowestbars
        window_highs = highs[i - depth + 1:i + 1]
        hb = np.argmax(window_highs) - (depth - 1)

        window_lows = lows[i - depth + 1:i + 1]
        lb = np.argmin(window_lows) - (depth - 1)

        # 高値/安値条件
        high_at_hb = highs[i + hb] if i + hb >= 0 else highs[0]
        high_cond = (high_at_hb - highs[i]) > dev_th

        low_at_lb = lows[i + lb] if i + lb >= 0 else lows[0]
        low_cond = (lows[i] - low_at_lb) > dev_th

        # barssince シミュレーション
        if not high_cond:
            hr_since = 0
        else:
            hr_since += 1

        if not low_cond:
            lr_since = 0
        else:
            lr_since += 1

        # 方向決定
        hr_gt_lr = hr_since > lr_since
        if not hr_gt_lr:
            dir_change_since = 0
        else:
            dir_change_since += 1

        current_dir = -1 if dir_change_since >= backstep else 1

        # 方向転換時の更新（確定ピボット記録）
        if current_dir != prev_dir and prev_dir != 0:
            zz_B_price = zz_A_price
            zz_B_idx = zz_A_idx

            # 確定したピボットを記録
            if prev_dir > 0:  # 上昇→下降: 高値ピボット確定
                zz_highs.iloc[zz_B_idx] = zz_B_price
            else:  # 下降→上昇: 安値ピボット確定
                zz_lows.iloc[zz_B_idx] = zz_B_price

        # ピボット更新 (sukepoyo_sub.pine準拠: 厳密比較 > / <)
        if current_dir > 0:  # 上昇方向
            if highs[i] > zz_B_price:  # >= ではなく > (sukepoyo_sub.pine準拠)
                zz_B_price = highs[i]
                zz_B_idx = i
                zz_A_price = lows[i]
                zz_A_idx = i
            if lows[i] < zz_A_price:
                zz_A_price = lows[i]
                zz_A_idx = i
        else:  # 下降方向
            if lows[i] < zz_B_price:  # <= ではなく < (sukepoyo_sub.pine準拠)
                zz_B_price = lows[i]
                zz_B_idx = i
                zz_A_price = highs[i]
                zz_A_idx = i
            if highs[i] > zz_A_price:
                zz_A_price = highs[i]
                zz_A_idx = i

        # 暫定先端を記録（各バーで現在の形成中ピボットを保存）
        zz_direction.iloc[i] = current_dir
        if current_dir > 0:  # 上昇方向: B=高値先端、A=安値先端（次の谷候補）
            prospective_high.iloc[i] = zz_B_price
            prospective_high_idx.iloc[i] = zz_B_idx
            prospective_low.iloc[i] = zz_A_price
            prospective_low_idx.iloc[i] = zz_A_idx
        else:  # 下降方向: B=安値先端、A=高値先端（次の山候補）
            prospective_low.iloc[i] = zz_B_price
            prospective_low_idx.iloc[i] = zz_B_idx
            prospective_high.iloc[i] = zz_A_price
            prospective_high_idx.iloc[i] = zz_A_idx

        prev_dir = current_dir

    return (zz_highs, zz_lows, prospective_high, prospective_low,
            prospective_high_idx, prospective_low_idx)


def calculate_zigzag(
    df: pd.DataFrame,
    depth: int = 5,
    deviation: float = 3.0,
    backstep: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """
    ZigZag インジケーターを計算（従来版 - 確定ピボットのみ）
    後方互換性のために残す

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（'High', 'Low'列を含む）
    depth : int
        ZigZag深度（デフォルト: 5）
    deviation : float
        偏差閾値（デフォルト: 3.0）
    backstep : int
        バックステップ（デフォルト: 2）

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        (確定高値, 確定安値)
    """
    result = calculate_zigzag_with_prospective(df, depth, deviation, backstep)
    return result[0], result[1]  # 確定高値、確定安値のみ返す


def get_latest_zigzag_level(
    zz_series: pd.Series,
    current_idx: int,
    lookback: Optional[int] = 50
) -> Optional[float]:
    """
    指定位置から直近のZigZagレベルを取得

    Parameters:
    -----------
    zz_series : pd.Series
        ZigZag高値または安値のSeries
    current_idx : int
        現在のインデックス位置
    lookback : int, optional
        遡る最大バー数（デフォルト: 50）

    Returns:
    --------
    float or None
        直近のZigZagレベル、見つからない場合はNone
    """
    if current_idx < 0 or current_idx >= len(zz_series):
        return None
    start_idx = max(0, current_idx - lookback) if lookback is not None else 0
    for i in range(current_idx, start_idx - 1, -1):
        if not pd.isna(zz_series.iloc[i]):
            return zz_series.iloc[i]
    return None
