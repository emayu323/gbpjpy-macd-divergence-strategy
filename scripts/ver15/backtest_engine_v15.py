"""
System Ver15 - バックテストエンジン
MACDダイバージェンス戦略のバックテスト実行
Ver15: 暫定ピボット方式によるリアルタイムダイバージェンス検出
       （12本確定を待たずに最新のローソク足を暫定先端として使用）
"""

import sys
from pathlib import Path

# 親ディレクトリをパスに追加（共通indicatorsモジュール参照用）
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import config_v15 as config

# 共通indicatorsモジュールからインポート
from indicators import (
    calculate_rci, calculate_ema, calculate_macd,
    calculate_zigzag, calculate_zigzag_with_prospective, calculate_atr,
    get_latest_zigzag_level, check_perfect_order, check_1h_rci_condition,
    check_5m_entry_long as check_5m_entry_long_v15,  # エイリアスで互換性維持
    check_5m_entry_short as check_5m_entry_short_v15  # エイリアスで互換性維持
)


@dataclass
class Trade:
    """トレード記録"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    profit_pips: float
    result: str
    exit_reason: str
    divergence_type: str  # 'hidden' or 'regular'


class BacktestEngineV15:
    """System Ver15 バックテストエンジン - 暫定ピボット方式"""

    def __init__(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame):
        self.df_5m = df_5m.copy()
        self.df_1h = df_1h.copy()
        self.df_4h = df_4h.copy()

        self.trades: List[Trade] = []
        self.current_position: Optional[Dict] = None

        # ダイバージェンス記録
        self.divergences: List[Dict] = []
        self.current_divergence: Optional[Dict] = None
        self.current_divergence_used = False

        # Ver15: 暫定ピボット用の前回値を追跡
        self.prev_prospective_low = np.nan
        self.prev_prospective_low_idx = np.nan
        self.prev_macd_at_prospective_low = np.nan

        self.prev_prospective_high = np.nan
        self.prev_prospective_high_idx = np.nan
        self.prev_macd_at_prospective_high = np.nan

        # インジケーター計算
        self._calculate_indicators()

    def _calculate_indicators(self):
        """全タイムフレームのインジケーターを計算"""
        print("インジケーターを計算中...")

        # 5分足のRCI
        self.df_5m['RCI_9'] = calculate_rci(self.df_5m, config.RCI_SHORT)
        self.df_5m['RCI_14'] = calculate_rci(self.df_5m, config.RCI_MID)

        # 5分足のZigZag（SL設定用）
        zz_highs_5m, zz_lows_5m = calculate_zigzag(
            self.df_5m,
            depth=config.ZIGZAG_5M_DEPTH,
            deviation=config.ZIGZAG_5M_DEVIATION,
            backstep=config.ZIGZAG_5M_BACKSTEP
        )
        self.df_5m['ZZ_High'] = zz_highs_5m
        self.df_5m['ZZ_Low'] = zz_lows_5m
        self.df_5m['ATR'] = calculate_atr(self.df_5m, config.SL_ATR_LENGTH)

        # 5分足のEMA（パーフェクトオーダー用）
        self.df_5m['EMA_20'] = calculate_ema(self.df_5m, config.EMA_5M_SHORT)
        self.df_5m['EMA_30'] = calculate_ema(self.df_5m, config.EMA_5M_MID)
        self.df_5m['EMA_40'] = calculate_ema(self.df_5m, config.EMA_5M_LONG)

        # 1時間足のMACD
        macd_line, signal_line, histogram = calculate_macd(
            self.df_1h,
            fast=config.MACD_FAST,
            slow=config.MACD_SLOW,
            signal=config.MACD_SIGNAL
        )
        self.df_1h['MACD'] = macd_line
        self.df_1h['MACD_Signal'] = signal_line
        self.df_1h['MACD_Hist'] = histogram

        # Ver15: 1時間足のZigZag（暫定ピボット対応）
        (zz_highs_1h, zz_lows_1h,
         prospective_high, prospective_low,
         prospective_high_idx, prospective_low_idx) = calculate_zigzag_with_prospective(
            self.df_1h,
            depth=config.ZIGZAG_1H_DEPTH,
            deviation=config.ZIGZAG_1H_DEVIATION,
            backstep=config.ZIGZAG_1H_BACKSTEP
        )
        self.df_1h['ZZ_High'] = zz_highs_1h
        self.df_1h['ZZ_Low'] = zz_lows_1h
        self.df_1h['Prospective_High'] = prospective_high
        self.df_1h['Prospective_Low'] = prospective_low
        self.df_1h['Prospective_High_Idx'] = prospective_high_idx
        self.df_1h['Prospective_Low_Idx'] = prospective_low_idx

        # 1時間足のEMA
        self.df_1h['EMA_200'] = calculate_ema(self.df_1h, 200)
        self.df_1h['EMA_20'] = calculate_ema(self.df_1h, config.EMA_1H_SHORT)
        self.df_1h['EMA_30'] = calculate_ema(self.df_1h, config.EMA_1H_MID)
        self.df_1h['EMA_40'] = calculate_ema(self.df_1h, config.EMA_1H_LONG)

        # 1時間足のRCI
        self.df_1h['RCI_9'] = calculate_rci(self.df_1h, config.RCI_SHORT)

        # 4時間足のEMA
        self.df_4h['EMA_50'] = calculate_ema(self.df_4h, config.EMA_4H)

        print(f"  5分足データ: {len(self.df_5m)} 件")
        print(f"  1時間足データ: {len(self.df_1h)} 件")
        print(f"  4時間足データ: {len(self.df_4h)} 件")

    def _get_4h_data_at_time(self, current_time: pd.Timestamp) -> Optional[pd.Series]:
        """指定時刻の4時間足データを取得"""
        mask = self.df_4h.index <= current_time
        if mask.sum() == 0:
            return None
        return self.df_4h.loc[mask].iloc[-1]

    def _get_1h_data_at_time(self, current_time: pd.Timestamp) -> Optional[Dict]:
        """指定時刻の1時間足データを取得"""
        mask = self.df_1h.index <= current_time
        if mask.sum() == 0:
            return None
        row = self.df_1h.loc[mask].iloc[-1]
        return row.to_dict()

    def _check_4h_trend(self, data_4h: pd.Series) -> str:
        """4時間足のトレンドを判定（50EMA単独判定）"""
        if data_4h is None:
            return 'none'
        close = data_4h['Close']
        ema_50 = data_4h.get('EMA_50', np.nan)
        if pd.isna(ema_50):
            return 'none'
        if close > ema_50:
            return 'uptrend'
        elif close < ema_50:
            return 'downtrend'
        return 'none'

    def _check_1h_perfect_order(self, data_1h: Dict) -> str:
        """1時間足のパーフェクトオーダーをチェック"""
        if data_1h is None:
            return 'none'
        ema_20 = data_1h.get('EMA_20', np.nan)
        ema_30 = data_1h.get('EMA_30', np.nan)
        ema_40 = data_1h.get('EMA_40', np.nan)
        return check_perfect_order(ema_20, ema_30, ema_40)

    def _check_1h_rci(self, data_1h: Dict, direction: str) -> bool:
        """1時間足RCI条件をチェック"""
        if data_1h is None:
            return False
        rci_value = data_1h.get('RCI_9', np.nan)
        return check_1h_rci_condition(rci_value, direction, config.RCI_1H_THRESHOLD)

    def _update_divergences(self, current_time: pd.Timestamp, trend_4h: str):
        """
        Ver15: 暫定ピボット方式でダイバージェンスを検出・更新

        12本確定を待たずに、各バーでの暫定先端を使ってダイバージェンスを検出。
        暫定先端が更新されるたびにダイバージェンスを再評価。
        """
        # 1時間足のインデックスを取得
        mask_1h = self.df_1h.index <= current_time
        if mask_1h.sum() == 0:
            return

        idx_1h = mask_1h.sum() - 1

        # Ver15: 暫定先端を使用
        current_prospective_low = self.df_1h['Prospective_Low'].iloc[idx_1h]
        current_prospective_low_idx = self.df_1h['Prospective_Low_Idx'].iloc[idx_1h]
        current_prospective_high = self.df_1h['Prospective_High'].iloc[idx_1h]
        current_prospective_high_idx = self.df_1h['Prospective_High_Idx'].iloc[idx_1h]

        # 暫定先端のインデックスでMACDを取得
        if not pd.isna(current_prospective_low_idx):
            low_idx = int(current_prospective_low_idx)
            current_macd_at_low = self.df_1h['MACD'].iloc[low_idx]
        else:
            current_macd_at_low = np.nan

        if not pd.isna(current_prospective_high_idx):
            high_idx = int(current_prospective_high_idx)
            current_macd_at_high = self.df_1h['MACD'].iloc[high_idx]
        else:
            current_macd_at_high = np.nan

        # 安値の暫定先端が更新された場合、ダイバージェンスをチェック
        if (not pd.isna(current_prospective_low) and
            not pd.isna(current_prospective_low_idx) and
            current_prospective_low_idx != self.prev_prospective_low_idx):

            if not pd.isna(self.prev_prospective_low) and not pd.isna(self.prev_macd_at_prospective_low):
                # ヒドゥン買い: 価格が切り上げ（押し目）、MACDが切り下げ
                if (current_prospective_low > self.prev_prospective_low and
                    current_macd_at_low < self.prev_macd_at_prospective_low):
                    self._set_divergence(current_time, 'hidden', 'long',
                                        current_prospective_low, current_macd_at_low)

                # レギュラー買い: 価格が切り下げ、MACDが切り上げ
                elif (current_prospective_low < self.prev_prospective_low and
                      current_macd_at_low > self.prev_macd_at_prospective_low):
                    self._set_divergence(current_time, 'regular', 'long',
                                        current_prospective_low, current_macd_at_low)

            # 前回値を更新
            self.prev_prospective_low = current_prospective_low
            self.prev_prospective_low_idx = current_prospective_low_idx
            self.prev_macd_at_prospective_low = current_macd_at_low

        # 高値の暫定先端が更新された場合、ダイバージェンスをチェック
        if (not pd.isna(current_prospective_high) and
            not pd.isna(current_prospective_high_idx) and
            current_prospective_high_idx != self.prev_prospective_high_idx):

            if not pd.isna(self.prev_prospective_high) and not pd.isna(self.prev_macd_at_prospective_high):
                # ヒドゥン売り: 価格が切り下げ（戻り目）、MACDが切り上げ
                if (current_prospective_high < self.prev_prospective_high and
                    current_macd_at_high > self.prev_macd_at_prospective_high):
                    self._set_divergence(current_time, 'hidden', 'short',
                                        current_prospective_high, current_macd_at_high)

                # レギュラー売り: 価格が切り上げ、MACDが切り下げ
                elif (current_prospective_high > self.prev_prospective_high and
                      current_macd_at_high < self.prev_macd_at_prospective_high):
                    self._set_divergence(current_time, 'regular', 'short',
                                        current_prospective_high, current_macd_at_high)

            # 前回値を更新
            self.prev_prospective_high = current_prospective_high
            self.prev_prospective_high_idx = current_prospective_high_idx
            self.prev_macd_at_prospective_high = current_macd_at_high

        # ダイバージェンスの有効期限チェック
        if self.current_divergence is not None:
            valid_time_threshold = current_time - timedelta(hours=config.DIVERGENCE_VALID_HOURS)
            if self.current_divergence['detected_time'] < valid_time_threshold:
                self.current_divergence = None
                self.current_divergence_used = False

    def _set_divergence(
        self,
        current_time: pd.Timestamp,
        div_type: str,
        direction: str,
        price: float,
        macd_value: float
    ):
        """最新のダイバージェンスを更新"""
        divergence = {
            'type': div_type,
            'direction': direction,
            'detected_time': current_time,
            'price': price,
            'macd': macd_value,
            'pivot_type': 'prospective'  # Ver15: 暫定ピボットであることを記録
        }
        self.current_divergence = divergence
        self.current_divergence_used = False
        self.divergences.append(divergence)
        print(f"ダイバージェンス検出 (暫定): {current_time} [{div_type}] [{direction}]")

    def _get_active_setup(self, current_time: pd.Timestamp, direction: str) -> bool:
        """
        指定方向の有効なダイバージェンスがあるかチェック
        """
        if self.current_divergence is None:
            return False

        valid_time_threshold = current_time - timedelta(hours=config.DIVERGENCE_VALID_HOURS)
        if self.current_divergence['detected_time'] < valid_time_threshold:
            return False

        if config.ENABLE_1DIV1ENTRY and self.current_divergence_used:
            return False

        return self.current_divergence['direction'] == direction

    def _check_5m_entry(self, idx: int, direction: str) -> bool:
        """5分足のエントリートリガーをチェック"""
        if idx == 0:
            return False

        current = self.df_5m.iloc[idx]
        previous = self.df_5m.iloc[idx - 1]

        rci_short_current = current['RCI_9']
        rci_short_previous = previous['RCI_9']

        # 5分足のパーフェクトオーダーをチェック
        ema_20 = current.get('EMA_20', np.nan)
        ema_30 = current.get('EMA_30', np.nan)
        ema_40 = current.get('EMA_40', np.nan)
        perfect_order_5m = check_perfect_order(ema_20, ema_30, ema_40)

        if direction == 'long':
            return check_5m_entry_long_v15(
                rci_short_current, rci_short_previous,
                perfect_order_5m,
                rci_short_threshold=config.RCI_OVERBOUGHT
            )
        elif direction == 'short':
            return check_5m_entry_short_v15(
                rci_short_current, rci_short_previous,
                perfect_order_5m,
                rci_short_threshold=config.RCI_OVERBOUGHT
            )

        return False

    def _calculate_sl_tp(self, idx: int, direction: str, entry_price: float) -> Tuple[float, float]:
        """SLとTPを計算"""
        atr = self.df_5m['ATR'].iloc[idx]
        if pd.isna(atr):
            atr = 0.0
        sl_buffer_price = atr * config.SL_ATR_MULT
        sl_fallback_price = atr * config.SL_FALLBACK_ATR_MULT
        sl_min_price = atr * config.SL_MIN_ATR_MULT
        pivot_idx = idx - config.ZIGZAG_5M_DEPTH

        if direction == 'long':
            zz_level = None
            if pivot_idx >= 0:
                zz_level = get_latest_zigzag_level(self.df_5m['ZZ_Low'], pivot_idx, lookback=None)
            if zz_level is not None:
                sl = zz_level - sl_buffer_price
            else:
                sl = entry_price - sl_fallback_price
            sl = min(sl, entry_price - sl_min_price)
        else:
            zz_level = None
            if pivot_idx >= 0:
                zz_level = get_latest_zigzag_level(self.df_5m['ZZ_High'], pivot_idx, lookback=None)
            if zz_level is not None:
                sl = zz_level + sl_buffer_price
            else:
                sl = entry_price + sl_fallback_price
            sl = max(sl, entry_price + sl_min_price)

        risk = abs(entry_price - sl)

        if direction == 'long':
            tp = entry_price + risk * config.RISK_REWARD_RATIO
        else:
            tp = entry_price - risk * config.RISK_REWARD_RATIO

        return sl, tp

    def _is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """取引時間帯かチェック"""
        hour = timestamp.hour
        return config.TRADING_START_HOUR <= hour < config.TRADING_END_HOUR

    def _check_exit(self, idx: int) -> Optional[Tuple[str, float]]:
        """ポジションの決済条件をチェック"""
        if self.current_position is None:
            return None

        current = self.df_5m.iloc[idx]
        direction = self.current_position['direction']
        sl = self.current_position['stop_loss']
        tp = self.current_position['take_profit']

        if direction == 'long':
            if current['High'] >= tp:
                return ('tp', tp)
            if current['Low'] <= sl:
                return ('sl', sl)
        else:
            if current['Low'] <= tp:
                return ('tp', tp)
            if current['High'] >= sl:
                return ('sl', sl)

        return None

    def run_backtest(self) -> pd.DataFrame:
        """バックテストを実行"""
        print("バックテスト開始...")
        print(f"期間: {self.df_5m.index[0]} ~ {self.df_5m.index[-1]}")
        print(f"暫定ピボット方式: {'有効' if config.USE_PROSPECTIVE_PIVOT else '無効'}")

        for idx in range(len(self.df_5m)):
            current_time = self.df_5m.index[idx]
            data_4h = self._get_4h_data_at_time(current_time)
            trend_4h = self._check_4h_trend(data_4h) if data_4h is not None else 'none'

            # ダイバージェンス更新（常に実行）
            self._update_divergences(current_time, trend_4h)

            # 既存ポジションの決済チェック
            if self.current_position is not None:
                exit_result = self._check_exit(idx)
                if exit_result is not None:
                    exit_reason, exit_price = exit_result
                    self._close_position(current_time, exit_price, exit_reason)

            # 新規エントリーチェック
            if self.current_position is None:
                if not self._is_trading_hours(current_time):
                    continue

                # 1H PO + 1H RCI±60 + Hidden/Regular Div + 5M PO
                data_1h = self._get_1h_data_at_time(current_time)
                perfect_order_1h = self._check_1h_perfect_order(data_1h)

                # 買いエントリーチェック
                if (perfect_order_1h == 'uptrend' and
                    self._check_1h_rci(data_1h, 'long') and
                    self._get_active_setup(current_time, 'long')):
                    if self._check_5m_entry(idx, 'long'):
                        self._open_position(idx, 'long')

                # 売りエントリーチェック
                elif (perfect_order_1h == 'downtrend' and
                      self._check_1h_rci(data_1h, 'short') and
                      self._get_active_setup(current_time, 'short')):
                    if self._check_5m_entry(idx, 'short'):
                        self._open_position(idx, 'short')

            # 進捗表示
            if idx % 10000 == 0:
                print(f"進捗: {idx}/{len(self.df_5m)} ({idx/len(self.df_5m)*100:.1f}%) "
                      f"- トレード数: {len(self.trades)} - ダイバージェンス数: {len(self.divergences)}")

        # 最後にポジションが残っている場合は強制決済
        if self.current_position is not None:
            last_time = self.df_5m.index[-1]
            last_price = self.df_5m.iloc[-1]['Close']
            self._close_position(last_time, last_price, 'end_of_data')

        print(f"バックテスト完了: {len(self.trades)}件のトレード, {len(self.divergences)}件のダイバージェンス")

        return self._create_trades_dataframe()

    def _open_position(self, idx: int, direction: str):
        """ポジションを開く"""
        current = self.df_5m.iloc[idx]
        entry_price = current['Close']
        entry_time = current.name

        sl, tp = self._calculate_sl_tp(idx, direction, entry_price)

        div_type = self.current_divergence['type'] if self.current_divergence is not None else 'unknown'
        if config.ENABLE_1DIV1ENTRY:
            self.current_divergence_used = True

        self.current_position = {
            'direction': direction,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'stop_loss': sl,
            'take_profit': tp,
            'entry_idx': idx,
            'divergence_type': div_type
        }

    def _close_position(self, exit_time: datetime, exit_price: float, exit_reason: str):
        """ポジションを閉じる"""
        pos = self.current_position
        direction = pos['direction']
        entry_price = pos['entry_price']

        if direction == 'long':
            profit_pips = (exit_price - entry_price) / config.POINT_VALUE
        else:
            profit_pips = (entry_price - exit_price) / config.POINT_VALUE

        result = 'win' if profit_pips > 0 else ('loss' if profit_pips < 0 else 'breakeven')

        trade = Trade(
            entry_time=pos['entry_time'],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=pos['stop_loss'],
            take_profit=pos['take_profit'],
            profit_pips=profit_pips,
            result=result,
            exit_reason=exit_reason,
            divergence_type=pos['divergence_type']
        )

        self.trades.append(trade)
        self.current_position = None

    def _create_trades_dataframe(self) -> pd.DataFrame:
        """トレードリストをDataFrameに変換"""
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        cumulative_profit_jpy = 0

        for trade in self.trades:
            holding_time = trade.exit_time - trade.entry_time
            holding_hours = holding_time.total_seconds() / 3600
            profit_jpy = trade.profit_pips * 100 * config.LOT_SIZE
            cumulative_profit_jpy += profit_jpy
            profit_pct = (profit_jpy / config.INITIAL_CAPITAL_JPY) * 100

            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'holding_hours': round(holding_hours, 2),
                'profit_pips': trade.profit_pips,
                'profit_pct': round(profit_pct, 4),
                'profit_jpy': round(profit_jpy, 0),
                'cumulative_profit_jpy': round(cumulative_profit_jpy, 0),
                'result': trade.result,
                'exit_reason': trade.exit_reason,
                'divergence_type': trade.divergence_type
            })

        return pd.DataFrame(trades_data)

    def get_divergences_dataframe(self) -> pd.DataFrame:
        """ダイバージェンスリストをDataFrameに変換"""
        if not self.divergences:
            return pd.DataFrame()
        return pd.DataFrame(self.divergences)

    def get_statistics(self) -> Dict:
        """統計情報を計算"""
        if not self.trades:
            return {}

        total_trades = len(self.trades)
        wins = len([t for t in self.trades if t.result == 'win'])
        losses = len([t for t in self.trades if t.result == 'loss'])

        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        total_profit = sum(t.profit_pips for t in self.trades)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.profit_pips for t in self.trades if t.profit_pips > 0)
        gross_loss = abs(sum(t.profit_pips for t in self.trades if t.profit_pips < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        tp_exits = len([t for t in self.trades if t.exit_reason == 'tp'])
        sl_exits = len([t for t in self.trades if t.exit_reason == 'sl'])

        hidden_trades = len([t for t in self.trades if t.divergence_type == 'hidden'])
        regular_trades = len([t for t in self.trades if t.divergence_type == 'regular'])

        total_profit_jpy = total_profit * 100 * config.LOT_SIZE
        expectancy_pct = (total_profit_jpy / config.INITIAL_CAPITAL_JPY) * 100 / total_trades if total_trades > 0 else 0

        # 最大ドローダウンの計算
        cumulative = 0
        peak = 0
        max_drawdown = 0
        max_drawdown_pct = 0

        for trade in self.trades:
            profit_jpy = trade.profit_pips * 100 * config.LOT_SIZE
            cumulative += profit_jpy
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = (drawdown / (config.INITIAL_CAPITAL_JPY + peak)) * 100

        # 保有時間の平均
        holding_times = []
        for trade in self.trades:
            holding_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            holding_times.append(holding_time)
        avg_holding_hours = np.mean(holding_times) if holding_times else 0

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_profit_pips': total_profit,
            'avg_profit_pips': avg_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'tp_exits': tp_exits,
            'sl_exits': sl_exits,
            'avg_win': gross_profit / wins if wins > 0 else 0,
            'avg_loss': gross_loss / losses if losses > 0 else 0,
            'total_divergences': len(self.divergences),
            'hidden_divergence_trades': hidden_trades,
            'regular_divergence_trades': regular_trades,
            'total_profit_jpy': total_profit_jpy,
            'expectancy_pct': expectancy_pct,
            'max_drawdown_jpy': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_holding_hours': avg_holding_hours
        }

    def get_monthly_summary(self) -> pd.DataFrame:
        """月別損益サマリーを生成"""
        if not self.trades:
            return pd.DataFrame()

        monthly_data = {}
        for trade in self.trades:
            year_month = trade.entry_time.strftime('%Y-%m')
            if year_month not in monthly_data:
                monthly_data[year_month] = {
                    'trades': 0,
                    'wins': 0,
                    'profit_pips': 0,
                    'profit_jpy': 0
                }
            monthly_data[year_month]['trades'] += 1
            if trade.result == 'win':
                monthly_data[year_month]['wins'] += 1
            monthly_data[year_month]['profit_pips'] += trade.profit_pips
            monthly_data[year_month]['profit_jpy'] += trade.profit_pips * 100 * config.LOT_SIZE

        df = pd.DataFrame.from_dict(monthly_data, orient='index')
        df.index.name = 'month'
        df['win_rate'] = (df['wins'] / df['trades'] * 100).round(2)
        df['profit_pips'] = df['profit_pips'].round(2)
        df['profit_jpy'] = df['profit_jpy'].round(0).astype(int)
        df = df[['trades', 'wins', 'win_rate', 'profit_pips', 'profit_jpy']]
        return df.sort_index()

    def get_yearly_summary(self) -> pd.DataFrame:
        """年別損益サマリーを生成"""
        if not self.trades:
            return pd.DataFrame()

        yearly_data = {}
        for trade in self.trades:
            year = trade.entry_time.strftime('%Y')
            if year not in yearly_data:
                yearly_data[year] = {
                    'trades': 0,
                    'wins': 0,
                    'profit_pips': 0,
                    'profit_jpy': 0
                }
            yearly_data[year]['trades'] += 1
            if trade.result == 'win':
                yearly_data[year]['wins'] += 1
            yearly_data[year]['profit_pips'] += trade.profit_pips
            yearly_data[year]['profit_jpy'] += trade.profit_pips * 100 * config.LOT_SIZE

        df = pd.DataFrame.from_dict(yearly_data, orient='index')
        df.index.name = 'year'
        df['win_rate'] = (df['wins'] / df['trades'] * 100).round(2)
        df['profit_pips'] = df['profit_pips'].round(2)
        df['profit_jpy'] = df['profit_jpy'].round(0).astype(int)
        df = df[['trades', 'wins', 'win_rate', 'profit_pips', 'profit_jpy']]
        return df.sort_index()
