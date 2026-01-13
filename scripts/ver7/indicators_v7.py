"""
System Ver7 - インジケーター計算モジュール
RCI, EMA, MACD, ZigZag, ダイバージェンス検出
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from datetime import datetime, timedelta


def calculate_rci(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.Series:
    """RCI (Rank Correlation Index) を計算"""
    rci_values = []
    for i in range(len(df)):
        if i < period - 1:
            rci_values.append(np.nan)
            continue
        prices = df[price_col].iloc[i - period + 1:i + 1].values
        time_ranks = np.arange(1, period + 1)
        price_ranks = period - pd.Series(prices).rank(method='average').values + 1
        d_squared_sum = np.sum((time_ranks - price_ranks) ** 2)
        rci = (1 - (6 * d_squared_sum) / (period * (period ** 2 - 1))) * 100
        rci_values.append(rci)
    return pd.Series(rci_values, index=df.index, name=f'RCI_{period}')


def calculate_ema(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.Series:
    """EMA (Exponential Moving Average) を計算"""
    return df[price_col].ewm(span=period, adjust=False).mean()


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                   signal: int = 9, price_col: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
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
        価格カラム名

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


def calculate_zigzag(df: pd.DataFrame, depth: int = 5, deviation: float = 3.0,
                     backstep: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    ZigZag インジケーターを計算（簡易版）

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

    last_pivot_type = None
    last_pivot_idx = 0

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

        if is_low_pivot and (last_pivot_type != 'low' or i - depth > last_pivot_idx + backstep):
            if last_pivot_type is None or last_pivot_type == 'high':
                zz_lows.iloc[i - depth] = current_low
                last_pivot_type = 'low'
                last_pivot_idx = i - depth

    return zz_highs, zz_lows


def get_latest_zigzag_level(zz_series: pd.Series, current_idx: int, lookback: int = 50) -> Optional[float]:
    """指定位置から直近のZigZagレベルを取得"""
    if current_idx < 0 or current_idx >= len(zz_series):
        return None
    start_idx = max(0, current_idx - lookback)
    for i in range(current_idx, start_idx - 1, -1):
        if not pd.isna(zz_series.iloc[i]):
            return zz_series.iloc[i]
    return None


def check_perfect_order(ema_short: float, ema_mid: float, ema_long: float) -> str:
    """EMAのパーフェクトオーダーをチェック"""
    if pd.isna(ema_short) or pd.isna(ema_mid) or pd.isna(ema_long):
        return 'none'
    if ema_short > ema_mid > ema_long:
        return 'uptrend'
    elif ema_short < ema_mid < ema_long:
        return 'downtrend'
    else:
        return 'none'


def detect_divergence(df: pd.DataFrame, zz_highs: pd.Series, zz_lows: pd.Series,
                     macd_line: pd.Series, current_idx: int,
                     trend_4h: str) -> Optional[Dict]:
    """
    ダイバージェンスを検出

    Parameters:
    -----------
    df : pd.DataFrame
        1時間足データ
    zz_highs : pd.Series
        ZigZag高値系列
    zz_lows : pd.Series
        ZigZag安値系列
    macd_line : pd.Series
        MACDライン
    current_idx : int
        現在のインデックス
    trend_4h : str
        4時間足のトレンド ('uptrend' or 'downtrend')

    Returns:
    --------
    Optional[Dict]
        ダイバージェンス情報 or None
        {
            'type': 'hidden' or 'regular',
            'direction': 'long' or 'short',
            'detected_time': Timestamp,
            'price1': float, 'price2': float,
            'macd1': float, 'macd2': float
        }
    """
    if trend_4h == 'none':
        return None

    # 上昇トレンドの場合、安値（底）を見る
    if trend_4h == 'uptrend':
        return _detect_bullish_divergence(df, zz_lows, macd_line, current_idx)

    # 下降トレンドの場合、高値（頂点）を見る
    elif trend_4h == 'downtrend':
        return _detect_bearish_divergence(df, zz_highs, macd_line, current_idx)

    return None


def _detect_bullish_divergence(df: pd.DataFrame, zz_lows: pd.Series,
                               macd_line: pd.Series, current_idx: int) -> Optional[Dict]:
    """
    買いダイバージェンスを検出

    上昇トレンド中に安値（底）で発生：
    - ヒドゥン：価格が切り上げ（押し目）、MACDが切り下げ → トレンド継続
    - レギュラー：価格が切り下げ（安値更新）、MACDが切り上げ → 反転上昇
    """
    # 直近2つの安値を取得
    lows_list = []
    for i in range(current_idx, max(0, current_idx - 50), -1):
        if not pd.isna(zz_lows.iloc[i]):
            lows_list.append({
                'idx': i,
                'time': df.index[i],
                'price': zz_lows.iloc[i],
                'macd': macd_line.iloc[i]
            })
        if len(lows_list) >= 2:
            break

    if len(lows_list) < 2:
        return None

    # 新しい方がlows_list[0]、古い方がlows_list[1]
    newer = lows_list[0]
    older = lows_list[1]

    # 価格とMACDの関係を判定
    price_higher = newer['price'] > older['price']  # 価格が切り上げ
    price_lower = newer['price'] < older['price']   # 価格が切り下げ
    macd_higher = newer['macd'] > older['macd']     # MACDが切り上げ
    macd_lower = newer['macd'] < older['macd']      # MACDが切り下げ

    # ヒドゥン・ダイバージェンス（推奨）：価格↑、MACD↓
    if price_higher and macd_lower:
        return {
            'type': 'hidden',
            'direction': 'long',
            'detected_time': newer['time'],
            'price1': older['price'],
            'price2': newer['price'],
            'macd1': older['macd'],
            'macd2': newer['macd']
        }

    # レギュラー・ダイバージェンス：価格↓、MACD↑
    if price_lower and macd_higher:
        return {
            'type': 'regular',
            'direction': 'long',
            'detected_time': newer['time'],
            'price1': older['price'],
            'price2': newer['price'],
            'macd1': older['macd'],
            'macd2': newer['macd']
        }

    return None


def _detect_bearish_divergence(df: pd.DataFrame, zz_highs: pd.Series,
                               macd_line: pd.Series, current_idx: int) -> Optional[Dict]:
    """
    売りダイバージェンスを検出

    下降トレンド中に高値（頂点）で発生：
    - ヒドゥン：価格が切り下げ（戻り目）、MACDが切り上げ → トレンド継続
    - レギュラー：価格が切り上げ（高値更新）、MACDが切り下げ → 反転下落
    """
    # 直近2つの高値を取得
    highs_list = []
    for i in range(current_idx, max(0, current_idx - 50), -1):
        if not pd.isna(zz_highs.iloc[i]):
            highs_list.append({
                'idx': i,
                'time': df.index[i],
                'price': zz_highs.iloc[i],
                'macd': macd_line.iloc[i]
            })
        if len(highs_list) >= 2:
            break

    if len(highs_list) < 2:
        return None

    newer = highs_list[0]
    older = highs_list[1]

    price_higher = newer['price'] > older['price']
    price_lower = newer['price'] < older['price']
    macd_higher = newer['macd'] > older['macd']
    macd_lower = newer['macd'] < older['macd']

    # ヒドゥン・ダイバージェンス（推奨）：価格↓、MACD↑
    if price_lower and macd_higher:
        return {
            'type': 'hidden',
            'direction': 'short',
            'detected_time': newer['time'],
            'price1': older['price'],
            'price2': newer['price'],
            'macd1': older['macd'],
            'macd2': newer['macd']
        }

    # レギュラー・ダイバージェンス：価格↑、MACD↓
    if price_higher and macd_lower:
        return {
            'type': 'regular',
            'direction': 'short',
            'detected_time': newer['time'],
            'price1': older['price'],
            'price2': newer['price'],
            'macd1': older['macd'],
            'macd2': newer['macd']
        }

    return None


def check_5m_entry_long_v7(rci_short_current: float, rci_short_previous: float,
                           rci_mid_current: float, rci_mid_previous: float) -> bool:
    """
    Ver7: 5分足の買いエントリートリガーをチェック

    条件:
    1. RCI短期(9)が-60以下
    2. RCI短期(9)が反転上昇
    3. RCI中期(14)が-40以下で上昇
    """
    if any(pd.isna([rci_short_current, rci_short_previous, rci_mid_current, rci_mid_previous])):
        return False

    condition1 = rci_short_current <= -60
    condition2 = rci_short_current > rci_short_previous
    condition3 = rci_mid_current <= -40 and rci_mid_current > rci_mid_previous

    return condition1 and condition2 and condition3


def check_5m_entry_short_v7(rci_short_current: float, rci_short_previous: float,
                            rci_mid_current: float, rci_mid_previous: float) -> bool:
    """
    Ver7: 5分足の売りエントリートリガーをチェック

    条件:
    1. RCI短期(9)が+60以上
    2. RCI短期(9)が反転下落
    3. RCI中期(14)が+40以上で下落
    """
    if any(pd.isna([rci_short_current, rci_short_previous, rci_mid_current, rci_mid_previous])):
        return False

    condition1 = rci_short_current >= 60
    condition2 = rci_short_current < rci_short_previous
    condition3 = rci_mid_current >= 40 and rci_mid_current < rci_mid_previous

    return condition1 and condition2 and condition3
