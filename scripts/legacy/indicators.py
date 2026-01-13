# -*- coding: utf-8 -*-
"""
テクニカル指標計算モジュール
EMA, ZigZag, RCI, MACD, ダイバージェンス検出
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def calculate_ema(df, periods):
    """
    EMA（指数平滑移動平均）を計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（close列を含む）
    periods : list
        EMA期間のリスト（例: [20, 30, 40]）

    Returns:
    --------
    pd.DataFrame
        EMA列が追加されたデータフレーム
    """
    df = df.copy()
    for period in periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df


def calculate_zigzag(df, depth=12, deviation=5, backstep=3, mintick=0.001):
    """
    ZigZag指標を計算（PineScriptロジックを再現）

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（high, low列を含む）
    depth : int
        過去何本分を見るか
    deviation : int
        転換判定の閾値（ポイント数）
    backstep : int
        方向転換の猶予期間
    mintick : float
        最小ティック（GBPJPYは0.001）

    Returns:
    --------
    pd.DataFrame
        zigzag_high, zigzag_low, zigzag_direction列が追加されたデータフレーム
        direction: 1=上昇トレンド, -1=下降トレンド
    """
    df = df.copy()
    n = len(df)

    # 閾値計算
    dev_threshold = deviation * mintick

    # 初期化
    zigzag_high = np.full(n, np.nan)
    zigzag_low = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    # 最初の転換点を探す
    last_pivot_high_idx = 0
    last_pivot_low_idx = 0
    last_pivot_high = df['high'].iloc[0]
    last_pivot_low = df['low'].iloc[0]
    current_direction = 1  # 1: 上昇, -1: 下降

    for i in range(depth, n):
        # 過去depth本の中での最高値・最安値のインデックス
        window_high = df['high'].iloc[max(0, i-depth):i+1]
        window_low = df['low'].iloc[max(0, i-depth):i+1]

        highest_idx = window_high.idxmax()
        lowest_idx = window_low.idxmin()

        highest = df['high'].loc[highest_idx]
        lowest = df['low'].loc[lowest_idx]

        # 現在の方向に応じて転換判定
        if current_direction == 1:  # 上昇トレンド中
            # 新高値更新
            if df['high'].iloc[i] > last_pivot_high:
                last_pivot_high = df['high'].iloc[i]
                last_pivot_high_idx = i

            # 下降転換の判定
            bars_since_high = i - last_pivot_high_idx
            if bars_since_high >= backstep:
                if last_pivot_high - df['low'].iloc[i] > dev_threshold:
                    # 転換確定
                    zigzag_high[last_pivot_high_idx] = last_pivot_high
                    current_direction = -1
                    last_pivot_low = df['low'].iloc[i]
                    last_pivot_low_idx = i

        else:  # 下降トレンド中
            # 新安値更新
            if df['low'].iloc[i] < last_pivot_low:
                last_pivot_low = df['low'].iloc[i]
                last_pivot_low_idx = i

            # 上昇転換の判定
            bars_since_low = i - last_pivot_low_idx
            if bars_since_low >= backstep:
                if df['high'].iloc[i] - last_pivot_low > dev_threshold:
                    # 転換確定
                    zigzag_low[last_pivot_low_idx] = last_pivot_low
                    current_direction = 1
                    last_pivot_high = df['high'].iloc[i]
                    last_pivot_high_idx = i

        direction[i] = current_direction

    df['zigzag_high'] = zigzag_high
    df['zigzag_low'] = zigzag_low
    df['zigzag_direction'] = direction

    return df


def calculate_rci(df, periods):
    """
    RCI（順位相関指数）を計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（close列を含む）
    periods : dict
        RCI期間の辞書（例: {'short': 9, 'mid': 14, 'long': 18}）

    Returns:
    --------
    pd.DataFrame
        rci_short, rci_mid, rci_long列が追加されたデータフレーム
    """
    df = df.copy()

    for name, period in periods.items():
        rci_values = []

        for i in range(len(df)):
            if i < period - 1:
                rci_values.append(np.nan)
            else:
                # 過去period本の価格データ
                prices = df['close'].iloc[i-period+1:i+1].values

                # 時間順位（新しい方が高い順位）
                time_rank = np.arange(1, period + 1)

                # 価格順位（高い方が高い順位）
                price_rank = period - np.argsort(np.argsort(prices))

                # 順位差の2乗和
                d_squared_sum = np.sum((time_rank - price_rank) ** 2)

                # RCI計算: (1 - 6 * Σd² / (n³ - n)) * 100
                rci = (1 - (6 * d_squared_sum) / (period ** 3 - period)) * 100
                rci_values.append(rci)

        df[f'rci_{name}'] = rci_values

    return df


def calculate_macd(df, fast=6, slow=13, signal=4):
    """
    MACD指標を計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ（close列を含む）
    fast : int
        Fast EMA期間
    slow : int
        Slow EMA期間
    signal : int
        Signal EMA期間

    Returns:
    --------
    pd.DataFrame
        macd, macd_signal, macd_histogram列が追加されたデータフレーム
    """
    df = df.copy()

    # MACD = Fast EMA - Slow EMA
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow

    # Signal = MACDのEMA
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()

    # Histogram = MACD - Signal
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df


def detect_divergence(df, lookback=20, price_col='close', indicator_col='macd'):
    """
    ダイバージェンスを検出

    Parameters:
    -----------
    df : pd.DataFrame
        価格データとインジケーターを含むデータフレーム
    lookback : int
        過去何本分を見るか
    price_col : str
        価格列名
    indicator_col : str
        インジケーター列名（例: 'macd', 'rci_short'）

    Returns:
    --------
    pd.DataFrame
        bullish_divergence（強気）, bearish_divergence（弱気）列が追加されたデータフレーム
        hidden_bullish_divergence（ヒドゥン強気）, hidden_bearish_divergence（ヒドゥン弱気）も追加
    """
    df = df.copy()
    n = len(df)

    # 初期化
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    hidden_bullish_div = np.zeros(n, dtype=bool)
    hidden_bearish_div = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        # 過去lookback本のデータ
        price_window = df[price_col].iloc[i-lookback:i+1]
        indicator_window = df[indicator_col].iloc[i-lookback:i+1]

        # 現在と過去の安値・高値を探す
        current_price = price_window.iloc[-1]
        current_indicator = indicator_window.iloc[-1]

        # 安値を探す（強気ダイバージェンス用）
        low_indices = []
        for j in range(len(price_window) - 5, -1, -1):
            if price_window.iloc[j] == price_window.iloc[max(0, j-2):min(len(price_window), j+3)].min():
                low_indices.append(j)
                if len(low_indices) >= 2:
                    break

        # 高値を探す（弱気ダイバージェンス用）
        high_indices = []
        for j in range(len(price_window) - 5, -1, -1):
            if price_window.iloc[j] == price_window.iloc[max(0, j-2):min(len(price_window), j+3)].max():
                high_indices.append(j)
                if len(high_indices) >= 2:
                    break

        # 強気ダイバージェンス（Bullish Divergence）
        # 価格: 安値切り下げ、インジケーター: 安値切り上げ
        if len(low_indices) >= 2:
            idx1, idx2 = low_indices[0], low_indices[1]
            if (price_window.iloc[idx1] < price_window.iloc[idx2] and
                indicator_window.iloc[idx1] > indicator_window.iloc[idx2]):
                bullish_div[i] = True

        # 弱気ダイバージェンス（Bearish Divergence）
        # 価格: 高値切り上げ、インジケーター: 高値切り下げ
        if len(high_indices) >= 2:
            idx1, idx2 = high_indices[0], high_indices[1]
            if (price_window.iloc[idx1] > price_window.iloc[idx2] and
                indicator_window.iloc[idx1] < indicator_window.iloc[idx2]):
                bearish_div[i] = True

        # ヒドゥン強気ダイバージェンス（Hidden Bullish Divergence）
        # 価格: 安値切り上げ、インジケーター: 安値切り下げ
        if len(low_indices) >= 2:
            idx1, idx2 = low_indices[0], low_indices[1]
            if (price_window.iloc[idx1] > price_window.iloc[idx2] and
                indicator_window.iloc[idx1] < indicator_window.iloc[idx2]):
                hidden_bullish_div[i] = True

        # ヒドゥン弱気ダイバージェンス（Hidden Bearish Divergence）
        # 価格: 高値切り下げ、インジケーター: 高値切り上げ
        if len(high_indices) >= 2:
            idx1, idx2 = high_indices[0], high_indices[1]
            if (price_window.iloc[idx1] < price_window.iloc[idx2] and
                indicator_window.iloc[idx1] > indicator_window.iloc[idx2]):
                hidden_bearish_div[i] = True

    df['bullish_divergence'] = bullish_div
    df['bearish_divergence'] = bearish_div
    df['hidden_bullish_divergence'] = hidden_bullish_div
    df['hidden_bearish_divergence'] = hidden_bearish_div

    return df


def add_all_indicators(df, ema_periods, rci_periods, macd_params,
                       zigzag_short_params, zigzag_long_params):
    """
    すべての指標を一括で計算

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ
    ema_periods : list
        EMA期間のリスト
    rci_periods : dict
        RCI期間の辞書
    macd_params : dict
        MACDパラメータ
    zigzag_short_params : dict
        短期ZigZagパラメータ
    zigzag_long_params : dict
        長期ZigZagパラメータ

    Returns:
    --------
    pd.DataFrame
        すべての指標が追加されたデータフレーム
    """
    df = df.copy()

    # EMA
    df = calculate_ema(df, ema_periods)

    # RCI
    df = calculate_rci(df, rci_periods)

    # MACD
    df = calculate_macd(df, **macd_params)

    # ZigZag（短期）
    df_zz_short = calculate_zigzag(
        df,
        depth=zigzag_short_params['depth'],
        deviation=zigzag_short_params['deviation'],
        backstep=zigzag_short_params['backstep']
    )
    df['zigzag_short_high'] = df_zz_short['zigzag_high']
    df['zigzag_short_low'] = df_zz_short['zigzag_low']
    df['zigzag_short_direction'] = df_zz_short['zigzag_direction']

    # ZigZag（長期）
    df_zz_long = calculate_zigzag(
        df,
        depth=zigzag_long_params['depth'],
        deviation=zigzag_long_params['deviation'],
        backstep=zigzag_long_params['backstep']
    )
    df['zigzag_long_high'] = df_zz_long['zigzag_high']
    df['zigzag_long_low'] = df_zz_long['zigzag_low']
    df['zigzag_long_direction'] = df_zz_long['zigzag_direction']

    # ダイバージェンス（MACD）
    df = detect_divergence(df, lookback=20, indicator_col='macd')

    return df
