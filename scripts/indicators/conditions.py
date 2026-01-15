"""
トレード条件判定モジュール
パーフェクトオーダー、RCI条件、エントリー条件など
"""

import pandas as pd
from typing import Any


def check_perfect_order(ema_short: float, ema_mid: float, ema_long: float) -> str:
    """
    EMAのパーフェクトオーダーをチェック

    Parameters:
    -----------
    ema_short : float
        短期EMA値
    ema_mid : float
        中期EMA値
    ema_long : float
        長期EMA値

    Returns:
    --------
    str
        'uptrend': 上昇トレンド（短期 > 中期 > 長期）
        'downtrend': 下降トレンド（短期 < 中期 < 長期）
        'none': パーフェクトオーダーなし
    """
    if pd.isna(ema_short) or pd.isna(ema_mid) or pd.isna(ema_long):
        return 'none'
    if ema_short > ema_mid > ema_long:
        return 'uptrend'
    elif ema_short < ema_mid < ema_long:
        return 'downtrend'
    else:
        return 'none'


def check_1h_rci_condition(rci_value: float, direction: str, threshold: float = 60) -> bool:
    """
    1時間足RCI条件をチェック

    Parameters:
    -----------
    rci_value : float
        1時間足RCI短期の値
    direction : str
        'long' or 'short'
    threshold : float
        閾値（デフォルト: 60）

    Returns:
    --------
    bool
        条件を満たしている場合True
    """
    if pd.isna(rci_value):
        return False

    if direction == 'long':
        # ロング: RCIが-60以下（売られすぎ）
        return rci_value <= -threshold
    elif direction == 'short':
        # ショート: RCIが+60以上（買われすぎ）
        return rci_value >= threshold

    return False


def check_5m_entry_long(
    rci_short_current: float,
    rci_short_previous: float,
    perfect_order_5m: str,
    rci_short_threshold: float = 60
) -> bool:
    """
    5分足の買いエントリートリガーをチェック

    条件:
    1. RCI短期(9)が-60以下（売られすぎ圏からの反転）
    2. RCI短期(9)が反転上昇（押し目完了）
    3. 5分足パーフェクトオーダーが上昇トレンド（20>30>40 EMA）

    Parameters:
    -----------
    rci_short_current : float
        現在のRCI短期値
    rci_short_previous : float
        1本前のRCI短期値
    perfect_order_5m : str
        5分足パーフェクトオーダー状態
    rci_short_threshold : float
        RCI閾値（デフォルト: 60）

    Returns:
    --------
    bool
        エントリー条件を満たす場合True
    """
    if any(pd.isna([rci_short_current, rci_short_previous])):
        return False

    condition1 = rci_short_current <= -rci_short_threshold
    condition2 = rci_short_current > rci_short_previous
    condition3 = perfect_order_5m == 'uptrend'

    return condition1 and condition2 and condition3


def check_5m_entry_short(
    rci_short_current: float,
    rci_short_previous: float,
    perfect_order_5m: str,
    rci_short_threshold: float = 60
) -> bool:
    """
    5分足の売りエントリートリガーをチェック

    条件:
    1. RCI短期(9)が+60以上（買われすぎ圏からの反転）
    2. RCI短期(9)が反転下落（戻り完了）
    3. 5分足パーフェクトオーダーが下降トレンド（20<30<40 EMA）

    Parameters:
    -----------
    rci_short_current : float
        現在のRCI短期値
    rci_short_previous : float
        1本前のRCI短期値
    perfect_order_5m : str
        5分足パーフェクトオーダー状態
    rci_short_threshold : float
        RCI閾値（デフォルト: 60）

    Returns:
    --------
    bool
        エントリー条件を満たす場合True
    """
    if any(pd.isna([rci_short_current, rci_short_previous])):
        return False

    condition1 = rci_short_current >= rci_short_threshold
    condition2 = rci_short_current < rci_short_previous
    condition3 = perfect_order_5m == 'downtrend'

    return condition1 and condition2 and condition3
