"""
System Ver15 - インジケーター計算モジュール
RCI, EMA, MACD, ZigZag, ダイバージェンス検出
Ver15: 暫定ピボット方式によるリアルタイムダイバージェンス検出
       （12本確定を待たずに最新のローソク足を暫定先端として使用）
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


def calculate_zigzag(df: pd.DataFrame, depth: int = 5, deviation: float = 3.0,
                     backstep: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    ZigZag インジケーターを計算（従来版 - 確定ピボットのみ）
    後方互換性のために残す
    """
    result = calculate_zigzag_with_prospective(df, depth, deviation, backstep)
    return result[0], result[1]  # 確定高値、確定安値のみ返す


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


def check_5m_entry_long_v15(
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
    """
    if any(pd.isna([rci_short_current, rci_short_previous])):
        return False

    condition1 = rci_short_current <= -rci_short_threshold
    condition2 = rci_short_current > rci_short_previous
    condition3 = perfect_order_5m == 'uptrend'

    return condition1 and condition2 and condition3


def check_5m_entry_short_v15(
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
    """
    if any(pd.isna([rci_short_current, rci_short_previous])):
        return False

    condition1 = rci_short_current >= rci_short_threshold
    condition2 = rci_short_current < rci_short_previous
    condition3 = perfect_order_5m == 'downtrend'

    return condition1 and condition2 and condition3
