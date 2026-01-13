"""
System Ver6 - インジケーター計算モジュール
RCI, EMA, ZigZagの計算を提供
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def calculate_rci(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.Series:
    """
    RCI (Rank Correlation Index) を計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ
    period : int
        計算期間
    price_col : str
        価格カラム名（デフォルト: 'Close'）

    Returns:
    --------
    pd.Series
        RCI値 (-100 ~ +100)
    """
    rci_values = []

    for i in range(len(df)):
        if i < period - 1:
            rci_values.append(np.nan)
            continue

        # 期間内のデータを取得
        prices = df[price_col].iloc[i - period + 1:i + 1].values

        # 時間順位（最新が1位）
        time_ranks = np.arange(1, period + 1)

        # 価格順位（高い方が1位）
        price_ranks = period - pd.Series(prices).rank(method='average').values + 1

        # スピアマンの順位相関係数を計算
        d_squared_sum = np.sum((time_ranks - price_ranks) ** 2)
        rci = (1 - (6 * d_squared_sum) / (period * (period ** 2 - 1))) * 100
        rci_values.append(rci)

    return pd.Series(rci_values, index=df.index, name=f'RCI_{period}')


def calculate_ema(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.Series:
    """
    EMA (Exponential Moving Average) を計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ
    period : int
        計算期間
    price_col : str
        価格カラム名（デフォルト: 'Close'）

    Returns:
    --------
    pd.Series
        EMA値
    """
    return df[price_col].ewm(span=period, adjust=False).mean()


def calculate_zigzag(df: pd.DataFrame, depth: int = 5, deviation: float = 3.0,
                     backstep: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    ZigZag インジケーターを計算（簡易版）

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（High, Low列が必要）
    depth : int
        検出する最小のbar数
    deviation : float
        最小の価格変動（pips）
    backstep : int
        連続したピーク検出の最小間隔

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        (ZigZag高値系列, ZigZag安値系列)
    """
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    zz_highs = pd.Series(np.nan, index=df.index)
    zz_lows = pd.Series(np.nan, index=df.index)

    # ZigZagポイントを検出
    last_pivot_type = None  # 'high' or 'low'
    last_pivot_idx = 0
    last_pivot_price = 0

    for i in range(depth, n):
        # 高値チェック
        is_high_pivot = True
        current_high = highs[i - depth]

        for j in range(max(0, i - depth - backstep), i):
            if j == i - depth:
                continue
            if highs[j] > current_high:
                is_high_pivot = False
                break

        # 安値チェック
        is_low_pivot = True
        current_low = lows[i - depth]

        for j in range(max(0, i - depth - backstep), i):
            if j == i - depth:
                continue
            if lows[j] < current_low:
                is_low_pivot = False
                break

        # ZigZagポイントを記録
        if is_high_pivot and (last_pivot_type != 'high' or i - depth > last_pivot_idx + backstep):
            if last_pivot_type is None or last_pivot_type == 'low':
                zz_highs.iloc[i - depth] = current_high
                last_pivot_type = 'high'
                last_pivot_idx = i - depth
                last_pivot_price = current_high

        if is_low_pivot and (last_pivot_type != 'low' or i - depth > last_pivot_idx + backstep):
            if last_pivot_type is None or last_pivot_type == 'high':
                zz_lows.iloc[i - depth] = current_low
                last_pivot_type = 'low'
                last_pivot_idx = i - depth
                last_pivot_price = current_low

    return zz_highs, zz_lows


def get_latest_zigzag_level(zz_series: pd.Series, current_idx: int, lookback: int = 50) -> Optional[float]:
    """
    指定位置から直近のZigZagレベルを取得

    Parameters:
    -----------
    zz_series : pd.Series
        ZigZag系列（高値または安値）
    current_idx : int
        現在位置のインデックス
    lookback : int
        遡る最大期間

    Returns:
    --------
    Optional[float]
        直近のZigZagレベル（見つからない場合はNone）
    """
    if current_idx < 0 or current_idx >= len(zz_series):
        return None

    start_idx = max(0, current_idx - lookback)

    for i in range(current_idx, start_idx - 1, -1):
        if not pd.isna(zz_series.iloc[i]):
            return zz_series.iloc[i]

    return None


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
        'uptrend': 上昇トレンド (20 > 30 > 40)
        'downtrend': 下降トレンド (20 < 30 < 40)
        'none': トレンドなし
    """
    if pd.isna(ema_short) or pd.isna(ema_mid) or pd.isna(ema_long):
        return 'none'

    if ema_short > ema_mid > ema_long:
        return 'uptrend'
    elif ema_short < ema_mid < ema_long:
        return 'downtrend'
    else:
        return 'none'


def check_1h_setup_long(rci_mid_current: float, rci_mid_previous: float) -> bool:
    """
    1時間足の買いセットアップをチェック

    条件:
    - RCI中期(14)が0ラインより下（押し目ゾーン）
    - RCI中期(14)が前回より上昇

    Parameters:
    -----------
    rci_mid_current : float
        現在のRCI中期値
    rci_mid_previous : float
        前回のRCI中期値

    Returns:
    --------
    bool
        セットアップ条件を満たす場合True
    """
    if pd.isna(rci_mid_current) or pd.isna(rci_mid_previous):
        return False

    return rci_mid_current < 0 and rci_mid_current > rci_mid_previous


def check_1h_setup_short(rci_mid_current: float, rci_mid_previous: float) -> bool:
    """
    1時間足の売りセットアップをチェック

    条件:
    - RCI中期(14)が0ラインより上（戻り目ゾーン）
    - RCI中期(14)が前回より下落

    Parameters:
    -----------
    rci_mid_current : float
        現在のRCI中期値
    rci_mid_previous : float
        前回のRCI中期値

    Returns:
    --------
    bool
        セットアップ条件を満たす場合True
    """
    if pd.isna(rci_mid_current) or pd.isna(rci_mid_previous):
        return False

    return rci_mid_current > 0 and rci_mid_current < rci_mid_previous


def check_5m_entry_long(rci_short_current: float, rci_short_previous: float,
                        rci_mid_current: float, rci_mid_previous: float,
                        threshold: float = -60) -> bool:
    """
    5分足の買いエントリートリガーをチェック

    条件:
    1. RCI短期(9)が-60以下（売られすぎ）
    2. RCI短期(9)が反転上昇（フック）
    3. RCI中期(14)も上昇

    Parameters:
    -----------
    rci_short_current : float
        現在のRCI短期値
    rci_short_previous : float
        前回のRCI短期値
    rci_mid_current : float
        現在のRCI中期値
    rci_mid_previous : float
        前回のRCI中期値
    threshold : float
        売られすぎ閾値（デフォルト: -60）

    Returns:
    --------
    bool
        エントリー条件を満たす場合True
    """
    if any(pd.isna([rci_short_current, rci_short_previous, rci_mid_current, rci_mid_previous])):
        return False

    # 1. RCI短期が売られすぎゾーンにある
    condition1 = rci_short_current <= threshold

    # 2. RCI短期が反転上昇（フック）
    condition2 = rci_short_current > rci_short_previous

    # 3. RCI中期も上昇
    condition3 = rci_mid_current > rci_mid_previous

    return condition1 and condition2 and condition3


def check_5m_entry_short(rci_short_current: float, rci_short_previous: float,
                         rci_mid_current: float, rci_mid_previous: float,
                         threshold: float = 60) -> bool:
    """
    5分足の売りエントリートリガーをチェック

    条件:
    1. RCI短期(9)が+60以上（買われすぎ）
    2. RCI短期(9)が反転下落（フック）
    3. RCI中期(14)も下落

    Parameters:
    -----------
    rci_short_current : float
        現在のRCI短期値
    rci_short_previous : float
        前回のRCI短期値
    rci_mid_current : float
        現在のRCI中期値
    rci_mid_previous : float
        前回のRCI中期値
    threshold : float
        買われすぎ閾値（デフォルト: 60）

    Returns:
    --------
    bool
        エントリー条件を満たす場合True
    """
    if any(pd.isna([rci_short_current, rci_short_previous, rci_mid_current, rci_mid_previous])):
        return False

    # 1. RCI短期が買われすぎゾーンにある
    condition1 = rci_short_current >= threshold

    # 2. RCI短期が反転下落（フック）
    condition2 = rci_short_current < rci_short_previous

    # 3. RCI中期も下落
    condition3 = rci_mid_current < rci_mid_previous

    return condition1 and condition2 and condition3
