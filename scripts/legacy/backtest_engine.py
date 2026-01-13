# -*- coding: utf-8 -*-
"""
バックテストエンジン
エントリー条件評価、SL/TP計算、勝敗判定、パフォーマンス指標算出
"""

import pandas as pd
import numpy as np
import config


class BacktestEngine:
    """バックテストエンジンクラス"""

    def __init__(self, m5_data, h1_data):
        """
        初期化

        Parameters:
        -----------
        m5_data : pd.DataFrame
            5分足データ（指標計算済み）
        h1_data : pd.DataFrame
            1時間足データ（指標計算済み）
        """
        self.m5_data = m5_data.copy()
        self.h1_data = h1_data.copy()
        self.trades = []

    def get_h1_trend(self, timestamp):
        """
        指定時刻の1時間足トレンド情報を取得

        Parameters:
        -----------
        timestamp : pd.Timestamp
            判定したい時刻

        Returns:
        --------
        dict
            1時間足のトレンド情報
        """
        # 該当する1時間足のインデックスを探す
        h1_idx = self.h1_data[self.h1_data.index <= timestamp].index
        if len(h1_idx) == 0:
            return None

        latest_h1_idx = h1_idx[-1]
        h1_row = self.h1_data.loc[latest_h1_idx]

        # パーフェクトオーダー判定
        is_uptrend = (h1_row['ema_20'] > h1_row['ema_30'] > h1_row['ema_40'])
        is_downtrend = (h1_row['ema_20'] < h1_row['ema_30'] < h1_row['ema_40'])

        return {
            'timestamp': latest_h1_idx,
            'is_uptrend': is_uptrend,
            'is_downtrend': is_downtrend,
            'ema_20': h1_row['ema_20'],
            'ema_30': h1_row['ema_30'],
            'ema_40': h1_row['ema_40'],
            'rci_short': h1_row['rci_short'],
            'rci_mid': h1_row['rci_mid'],
            'rci_long': h1_row['rci_long'],
            'macd': h1_row['macd'],
            'macd_signal': h1_row['macd_signal'],
            'hidden_bullish_divergence': h1_row['hidden_bullish_divergence'],
            'hidden_bearish_divergence': h1_row['hidden_bearish_divergence'],
        }

    def calculate_sl_tp(self, entry_price, direction, zigzag_level):
        """
        SLとTPを計算

        Parameters:
        -----------
        entry_price : float
            エントリー価格
        direction : str
            'long' or 'short'
        zigzag_level : float
            ZigZagの直近高値/安値

        Returns:
        --------
        tuple
            (sl_price, tp_price, sl_pips)
        """
        # スプレッドを考慮したエントリー価格
        spread = config.SPREAD_PIPS * config.POINT_VALUE

        if direction == 'long':
            # ロングの場合: ZigZag安値がSL
            actual_entry = entry_price + spread
            sl_price = zigzag_level
            sl_pips = (actual_entry - sl_price) / config.POINT_VALUE
            tp_price = actual_entry + (sl_pips * config.RISK_REWARD_RATIO * config.POINT_VALUE)

        else:  # short
            # ショートの場合: ZigZag高値がSL
            actual_entry = entry_price - spread
            sl_price = zigzag_level
            sl_pips = (sl_price - actual_entry) / config.POINT_VALUE
            tp_price = actual_entry - (sl_pips * config.RISK_REWARD_RATIO * config.POINT_VALUE)

        return sl_price, tp_price, sl_pips

    def check_trade_outcome(self, entry_idx, direction, sl_price, tp_price):
        """
        トレードの勝敗を判定

        Parameters:
        -----------
        entry_idx : int
            エントリーのインデックス
        direction : str
            'long' or 'short'
        sl_price : float
            損切り価格
        tp_price : float
            利確価格

        Returns:
        --------
        dict
            {'result': 'win'/'loss', 'exit_idx': int, 'exit_price': float, 'pips': float}
        """
        # エントリー後の価格推移を確認（最大500本先まで）
        max_bars = min(500, len(self.m5_data) - entry_idx - 1)

        for i in range(1, max_bars + 1):
            current_idx = entry_idx + i
            current_row = self.m5_data.iloc[current_idx]

            if direction == 'long':
                # ロングの場合
                # TPヒット判定
                if current_row['high'] >= tp_price:
                    pips = (tp_price - self.m5_data.iloc[entry_idx]['close']) / config.POINT_VALUE
                    return {
                        'result': 'win',
                        'exit_idx': current_idx,
                        'exit_price': tp_price,
                        'pips': pips
                    }
                # SLヒット判定
                if current_row['low'] <= sl_price:
                    pips = (sl_price - self.m5_data.iloc[entry_idx]['close']) / config.POINT_VALUE
                    return {
                        'result': 'loss',
                        'exit_idx': current_idx,
                        'exit_price': sl_price,
                        'pips': pips
                    }

            else:  # short
                # ショートの場合
                # TPヒット判定
                if current_row['low'] <= tp_price:
                    pips = (self.m5_data.iloc[entry_idx]['close'] - tp_price) / config.POINT_VALUE
                    return {
                        'result': 'win',
                        'exit_idx': current_idx,
                        'exit_price': tp_price,
                        'pips': pips
                    }
                # SLヒット判定
                if current_row['high'] >= sl_price:
                    pips = (self.m5_data.iloc[entry_idx]['close'] - sl_price) / config.POINT_VALUE
                    return {
                        'result': 'loss',
                        'exit_idx': current_idx,
                        'exit_price': sl_price,
                        'pips': pips
                    }

        # タイムアウト（決着つかず）
        return None

    def evaluate_entry_conditions(self, idx, h1_trend, filters):
        """
        エントリー条件を評価

        Parameters:
        -----------
        idx : int
            5分足のインデックス
        h1_trend : dict
            1時間足のトレンド情報
        filters : dict
            フィルター条件の設定

        Returns:
        --------
        str or None
            'long', 'short', or None
        """
        if h1_trend is None:
            return None

        row = self.m5_data.iloc[idx]

        # トレード時間帯の確認
        hour = row.name.hour
        if not (config.TRADE_START_HOUR <= hour < config.TRADE_END_HOUR):
            return None

        # 重要指標発表前後の時間帯を回避
        if filters.get('avoid_news_times', True):
            for start_hour, end_hour in config.AVOID_NEWS_TIMES:
                if start_hour <= hour < end_hour:
                    return None

        # === ロングエントリー条件 ===
        if h1_trend['is_uptrend']:
            # 基本条件: 1時間足が上昇トレンド

            # フィルター1: 1時間足RCIの状態
            if filters.get('h1_rci_aligned', False):
                # RCI 3本とも上向きであること
                if not (h1_trend['rci_short'] > 0 and
                        h1_trend['rci_mid'] > 0 and
                        h1_trend['rci_long'] > 0):
                    return None

            # フィルター2: 1時間足MACDヒドゥンダイバージェンス
            if filters.get('h1_hidden_div', False):
                if not h1_trend['hidden_bullish_divergence']:
                    return None

            # フィルター3: 5分足EMAと価格の乖離率
            if filters.get('m5_ema_divergence', False):
                ema_20 = row['ema_20']
                price = row['close']
                divergence_pct = ((price - ema_20) / ema_20) * 100
                # 乖離が大きすぎる場合は見送り
                if abs(divergence_pct) > filters.get('max_ema_divergence_pct', 2.0):
                    return None

            # トリガー条件: 5分足でのシグナル
            trigger_met = False

            # トリガー1: MACDダイバージェンス
            if filters.get('trigger_macd_div', True):
                if row['bullish_divergence']:
                    trigger_met = True

            # トリガー2: RCI反転（底値圏から上昇）
            if filters.get('trigger_rci_reversal', False):
                if idx > 0:
                    prev_row = self.m5_data.iloc[idx - 1]
                    # RCI短期が-80以下から-80を上抜け
                    if (prev_row['rci_short'] <= -80 and
                        row['rci_short'] > -80):
                        trigger_met = True

            # トリガー3: ZigZagブレイク
            if filters.get('trigger_zigzag_break', False):
                # ZigZag短期が上昇転換した直後
                if idx > 0:
                    prev_row = self.m5_data.iloc[idx - 1]
                    if (prev_row['zigzag_short_direction'] == -1 and
                        row['zigzag_short_direction'] == 1):
                        trigger_met = True

            if trigger_met:
                return 'long'

        # === ショートエントリー条件 ===
        if h1_trend['is_downtrend']:
            # 基本条件: 1時間足が下降トレンド

            # フィルター1: 1時間足RCIの状態
            if filters.get('h1_rci_aligned', False):
                # RCI 3本とも下向きであること
                if not (h1_trend['rci_short'] < 0 and
                        h1_trend['rci_mid'] < 0 and
                        h1_trend['rci_long'] < 0):
                    return None

            # フィルター2: 1時間足MACDヒドゥンダイバージェンス
            if filters.get('h1_hidden_div', False):
                if not h1_trend['hidden_bearish_divergence']:
                    return None

            # フィルター3: 5分足EMAと価格の乖離率
            if filters.get('m5_ema_divergence', False):
                ema_20 = row['ema_20']
                price = row['close']
                divergence_pct = ((price - ema_20) / ema_20) * 100
                if abs(divergence_pct) > filters.get('max_ema_divergence_pct', 2.0):
                    return None

            # トリガー条件
            trigger_met = False

            # トリガー1: MACDダイバージェンス
            if filters.get('trigger_macd_div', True):
                if row['bearish_divergence']:
                    trigger_met = True

            # トリガー2: RCI反転（天井圏から下降）
            if filters.get('trigger_rci_reversal', False):
                if idx > 0:
                    prev_row = self.m5_data.iloc[idx - 1]
                    # RCI短期が+80以上から+80を下抜け
                    if (prev_row['rci_short'] >= 80 and
                        row['rci_short'] < 80):
                        trigger_met = True

            # トリガー3: ZigZagブレイク
            if filters.get('trigger_zigzag_break', False):
                if idx > 0:
                    prev_row = self.m5_data.iloc[idx - 1]
                    if (prev_row['zigzag_short_direction'] == 1 and
                        row['zigzag_short_direction'] == -1):
                        trigger_met = True

            if trigger_met:
                return 'short'

        return None

    def run_backtest(self, filters):
        """
        バックテストを実行

        Parameters:
        -----------
        filters : dict
            フィルター条件の設定

        Returns:
        --------
        list
            トレード結果のリスト
        """
        self.trades = []

        # 5分足データを順番に走査
        for idx in range(100, len(self.m5_data) - 500):  # 余裕を持たせる
            row = self.m5_data.iloc[idx]
            timestamp = row.name

            # 1時間足のトレンド情報を取得
            h1_trend = self.get_h1_trend(timestamp)

            # エントリー条件を評価
            direction = self.evaluate_entry_conditions(idx, h1_trend, filters)

            if direction is None:
                continue

            # ZigZag短期の直近高値/安値を取得
            entry_price = row['close']
            if direction == 'long':
                # 過去のZigZag安値を探す（現在価格より下のもの）
                recent_lows = self.m5_data.iloc[max(0, idx-50):idx]['zigzag_short_low'].dropna()
                if len(recent_lows) == 0:
                    continue
                # 現在価格より下の安値のみを使用
                valid_lows = recent_lows[recent_lows < entry_price]
                if len(valid_lows) == 0:
                    continue
                zigzag_level = valid_lows.iloc[-1]
            else:
                # 過去のZigZag高値を探す（現在価格より上のもの）
                recent_highs = self.m5_data.iloc[max(0, idx-50):idx]['zigzag_short_high'].dropna()
                if len(recent_highs) == 0:
                    continue
                # 現在価格より上の高値のみを使用
                valid_highs = recent_highs[recent_highs > entry_price]
                if len(valid_highs) == 0:
                    continue
                zigzag_level = valid_highs.iloc[-1]

            # SL/TP計算
            sl_price, tp_price, sl_pips = self.calculate_sl_tp(entry_price, direction, zigzag_level)

            # SL幅の妥当性チェック
            if sl_pips <= 0:  # 負のSL幅は無効
                continue
            if sl_pips < 3:  # SL幅が3pips未満は避ける（スプレッドで即損切りされる可能性）
                continue
            if sl_pips > 50:  # 50pips以上のSLは避ける
                continue

            # トレード結果を判定
            outcome = self.check_trade_outcome(idx, direction, sl_price, tp_price)

            if outcome is None:
                continue

            # トレード記録
            trade = {
                'entry_time': timestamp,
                'entry_price': entry_price,
                'direction': direction,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'sl_pips': sl_pips,
                'exit_time': self.m5_data.iloc[outcome['exit_idx']].name,
                'exit_price': outcome['exit_price'],
                'result': outcome['result'],
                'pips': outcome['pips']
            }
            self.trades.append(trade)

        return self.trades

    def calculate_performance(self):
        """
        パフォーマンス指標を計算

        Returns:
        --------
        dict
            パフォーマンス指標
        """
        if len(self.trades) == 0:
            return None

        df_trades = pd.DataFrame(self.trades)

        # 勝ちトレード・負けトレード
        wins = df_trades[df_trades['result'] == 'win']
        losses = df_trades[df_trades['result'] == 'loss']

        total_trades = len(df_trades)
        win_count = len(wins)
        loss_count = len(losses)

        # 勝率
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

        # 総利益・総損失
        total_profit = wins['pips'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pips'].sum()) if len(losses) > 0 else 0

        # プロフィットファクター
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # 最大ドローダウン（簡易版）
        cumulative_pips = df_trades['pips'].cumsum()
        running_max = cumulative_pips.cummax()
        drawdown = running_max - cumulative_pips
        max_drawdown = drawdown.max()

        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'total_profit_pips': total_profit,
            'total_loss_pips': total_loss,
            'net_profit_pips': total_profit - total_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pips': max_drawdown,
            'avg_win_pips': wins['pips'].mean() if len(wins) > 0 else 0,
            'avg_loss_pips': losses['pips'].mean() if len(losses) > 0 else 0,
        }
