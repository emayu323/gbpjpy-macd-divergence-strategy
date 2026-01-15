"""
System Ver14 - インジケーター計算モジュール
RCI, EMA, MACD, ZigZag, ダイバージェンス検出
Ver14: 1時間足RCI短期±60条件を追加
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from datetime import datetime, timedelta


def calculate_rci(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.Series:
    """
    RCI (Rank Correlation Index) を計算
    PineScriptと完全に一致するロジック
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) を計算（Wilder方式）"""
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


def calculate_zigzag(df: pd.DataFrame, depth: int = 5, deviation: float = 3.0,
                     backstep: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    ZigZag インジケーターを計算（sukepoyo_sub.pine方式）

    PineScriptのロジックを忠実に再現:
    - ta.highestbars / ta.lowestbars でdepth期間の最高値/最安値のバーを取得
    - deviation閾値でトレンド継続を判定
    - barssince でトレンド方向を決定
    - chart.point 相当で動的にピボットを追跡

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        (ZigZag高値系列, ZigZag安値系列) - 確定ピボットのみ
    """
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    zz_highs = pd.Series(np.nan, index=df.index)
    zz_lows = pd.Series(np.nan, index=df.index)

    # deviation閾値（PineScriptではsyminfo.mintickを使用、ここでは価格ポイントとして扱う）
    dev_th = deviation * 0.01  # GBPJPYの場合、mintick = 0.01

    # 方向追跡用の変数
    prev_dir = 0
    hr_since = 0  # 高値条件が真からの経過バー数
    lr_since = 0  # 安値条件が真からの経過バー数
    dir_change_since = 0  # 方向変化条件からの経過バー数

    # chart.point相当のピボット追跡
    zz_A_price = lows[0] if n > 0 else 0  # 現在形成中
    zz_A_idx = 0
    zz_B_price = lows[0] if n > 0 else 0  # 直近確定
    zz_B_idx = 0
    zz_C_price = highs[0] if n > 0 else 0  # 1つ前の確定
    zz_C_idx = 0

    for i in range(n):
        if i < depth:
            continue

        # highestbars: depth期間の最高値のインデックス（0が現在、負の値が過去）
        window_highs = highs[i - depth + 1:i + 1]
        hb = np.argmax(window_highs) - (depth - 1)  # 0 to -(depth-1)

        # lowestbars: depth期間の最安値のインデックス
        window_lows = lows[i - depth + 1:i + 1]
        lb = np.argmin(window_lows) - (depth - 1)

        # 高値条件: high[-hb] - high > dev_th
        high_at_hb = highs[i + hb] if i + hb >= 0 else highs[0]
        high_cond = (high_at_hb - highs[i]) > dev_th

        # 安値条件: low - low[-lb] > dev_th
        low_at_lb = lows[i + lb] if i + lb >= 0 else lows[0]
        low_cond = (lows[i] - low_at_lb) > dev_th

        # barssince のシミュレーション
        if not high_cond:
            hr_since = 0
        else:
            hr_since += 1

        if not low_cond:
            lr_since = 0
        else:
            lr_since += 1

        # 方向決定: barssince(not (hr > lr)) >= backstep ? -1 : 1
        hr_gt_lr = hr_since > lr_since
        if not hr_gt_lr:
            dir_change_since = 0
        else:
            dir_change_since += 1

        current_dir = -1 if dir_change_since >= backstep else 1

        # 方向転換時の更新
        if current_dir != prev_dir and prev_dir != 0:
            zz_C_price = zz_B_price
            zz_C_idx = zz_B_idx
            zz_B_price = zz_A_price
            zz_B_idx = zz_A_idx

            # 確定したピボットを記録
            if prev_dir > 0:  # 上昇→下降: 高値ピボット確定
                zz_highs.iloc[zz_B_idx] = zz_B_price
            else:  # 下降→上昇: 安値ピボット確定
                zz_lows.iloc[zz_B_idx] = zz_B_price

        # ピボット更新
        if current_dir > 0:  # 上昇方向
            if highs[i] > zz_B_price:
                zz_B_price = highs[i]
                zz_B_idx = i
                zz_A_price = lows[i]
                zz_A_idx = i
            if lows[i] < zz_A_price:
                zz_A_price = lows[i]
                zz_A_idx = i
        else:  # 下降方向
            if lows[i] < zz_B_price:
                zz_B_price = lows[i]
                zz_B_idx = i
                zz_A_price = highs[i]
                zz_A_idx = i
            if highs[i] > zz_A_price:
                zz_A_price = highs[i]
                zz_A_idx = i

        prev_dir = current_dir

    return zz_highs, zz_lows


def get_latest_zigzag_level(
    zz_series: pd.Series,
    current_idx: int,
    lookback: Optional[int] = 50
) -> Optional[float]:
    """指定位置から直近のZigZagレベルを取得"""
    if current_idx < 0 or current_idx >= len(zz_series):
        return None
    start_idx = max(0, current_idx - lookback) if lookback is not None else 0
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


def check_1h_rci_condition(rci_value: float, direction: str, threshold: float = 60) -> bool:
    """
    Ver14: 1時間足RCI条件をチェック

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


def check_5m_entry_long_v14(
    rci_short_current: float,
    rci_short_previous: float,
    perfect_order_5m: str,
    rci_short_threshold: float = 60
) -> bool:
    """
    Ver14: 5分足の買いエントリートリガーをチェック

    条件:
    1. RCI短期(9)が-60以下（売られすぎ圏からの反転）
    2. RCI短期(9)が反転上昇（押し目完了）
    3. 5分足パーフェクトオーダーが上昇トレンド（20>30>40 EMA）
    """
    if any(pd.isna([rci_short_current, rci_short_previous])):
        return False

    condition1 = rci_short_current <= -rci_short_threshold
    condition2 = rci_short_current > rci_short_previous
    condition3 = perfect_order_5m == 'uptrend'

    return condition1 and condition2 and condition3


def check_5m_entry_short_v14(
    rci_short_current: float,
    rci_short_previous: float,
    perfect_order_5m: str,
    rci_short_threshold: float = 60
) -> bool:
    """
    Ver14: 5分足の売りエントリートリガーをチェック

    条件:
    1. RCI短期(9)が+60以上（買われすぎ圏からの反転）
    2. RCI短期(9)が反転下落（戻り完了）
    3. 5分足パーフェクトオーダーが下降トレンド（20<30<40 EMA）
    """
    if any(pd.isna([rci_short_current, rci_short_previous])):
        return False

    condition1 = rci_short_current >= rci_short_threshold
    condition2 = rci_short_current < rci_short_previous
    condition3 = perfect_order_5m == 'downtrend'

    return condition1 and condition2 and condition3
