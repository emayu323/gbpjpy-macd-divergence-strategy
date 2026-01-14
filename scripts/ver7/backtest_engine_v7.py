"""
System Ver7 - バックテストエンジン
MACDダイバージェンス戦略のバックテスト実行
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import config_v7 as config
from indicators_v7 import (
    calculate_rci, calculate_ema, calculate_macd, calculate_zigzag,
    get_latest_zigzag_level, check_perfect_order,
    detect_divergence, check_5m_entry_long_v7, check_5m_entry_short_v7
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


class BacktestEngineV7:
    """System Ver7 バックテストエンジン"""

    def __init__(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame):
        self.df_5m = df_5m.copy()
        self.df_1h = df_1h.copy()
        self.df_4h = df_4h.copy()

        self.trades: List[Trade] = []
        self.current_position: Optional[Dict] = None

        # ダイバージェンス記録
        self.divergences: List[Dict] = []
        self.active_divergences: List[Dict] = []  # 有効期間内のダイバージェンス

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

        # 1時間足のZigZag（ダイバージェンス検出用）
        zz_highs_1h, zz_lows_1h = calculate_zigzag(
            self.df_1h,
            depth=config.ZIGZAG_1H_DEPTH,
            deviation=config.ZIGZAG_1H_DEVIATION,
            backstep=config.ZIGZAG_1H_BACKSTEP
        )
        self.df_1h['ZZ_High'] = zz_highs_1h
        self.df_1h['ZZ_Low'] = zz_lows_1h

        # 1時間足のEMA（パターンC用）
        self.df_1h['EMA_200'] = calculate_ema(self.df_1h, 200)

        # 4時間足のEMA
        self.df_4h['EMA_20'] = calculate_ema(self.df_4h, config.EMA_SHORT)
        self.df_4h['EMA_30'] = calculate_ema(self.df_4h, config.EMA_MID)
        self.df_4h['EMA_40'] = calculate_ema(self.df_4h, config.EMA_LONG)

        print("インジケーター計算完了")

    def _get_1h_data_at_time(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """指定時刻の1時間足データを取得"""
        mask = self.df_1h.index <= timestamp
        if mask.sum() == 0:
            return None
        return self.df_1h[mask].iloc[-1]

    def _get_4h_data_at_time(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """指定時刻の4時間足データを取得"""
        mask = self.df_4h.index <= timestamp
        if mask.sum() == 0:
            return None
        return self.df_4h[mask].iloc[-1]

    def _check_4h_trend(self, data_4h: pd.Series) -> str:
        """4時間足のトレンドをチェック（4時間足50EMA単独）"""
        # パターンA: 価格とEMA_20（50EMA）の単独比較
        if pd.isna(data_4h['Close']) or pd.isna(data_4h['EMA_20']):
            return 'none'
        if data_4h['Close'] > data_4h['EMA_20']:
            return 'uptrend'
        elif data_4h['Close'] < data_4h['EMA_20']:
            return 'downtrend'
        else:
            return 'none'

    def _update_divergences(self, current_time: pd.Timestamp, trend_4h: str):
        """
        ダイバージェンスを検出・更新

        Parameters:
        -----------
        current_time : pd.Timestamp
            現在時刻
        trend_4h : str
            4時間足のトレンド
        """
        # 1時間足のインデックスを取得
        mask_1h = self.df_1h.index <= current_time
        if mask_1h.sum() == 0:
            return

        idx_1h = mask_1h.sum() - 1

        # ダイバージェンス検出
        divergence = detect_divergence(
            self.df_1h,
            self.df_1h['ZZ_High'],
            self.df_1h['ZZ_Low'],
            self.df_1h['MACD'],
            idx_1h,
            trend_4h
        )

        if divergence is not None:
            # 同じ時刻のダイバージェンスが既に記録されているかチェック
            already_exists = any(
                d['detected_time'] == divergence['detected_time']
                for d in self.divergences
            )

            if not already_exists:
                self.divergences.append(divergence)
                print(f"ダイバージェンス検出: {divergence['detected_time']} "
                      f"[{divergence['type']}] [{divergence['direction']}]")

        # 有効期間内のダイバージェンスのみを抽出
        valid_time_threshold = current_time - timedelta(hours=config.DIVERGENCE_VALID_HOURS)
        self.active_divergences = [
            d for d in self.divergences
            if d['detected_time'] >= valid_time_threshold
        ]

    def _get_active_setup(self, direction: str) -> bool:
        """
        指定方向の有効なダイバージェンスがあるかチェック

        Parameters:
        -----------
        direction : str
            'long' or 'short'

        Returns:
        --------
        bool
            有効なダイバージェンスがある場合True
        """
        return any(d['direction'] == direction for d in self.active_divergences)

    def _check_5m_entry(self, idx: int, direction: str) -> bool:
        """5分足のエントリートリガーをチェック"""
        if idx == 0:
            return False

        current = self.df_5m.iloc[idx]
        previous = self.df_5m.iloc[idx - 1]

        rci_short_current = current['RCI_9']
        rci_short_previous = previous['RCI_9']
        rci_mid_current = current['RCI_14']
        rci_mid_previous = previous['RCI_14']

        if direction == 'long':
            return check_5m_entry_long_v7(
                rci_short_current, rci_short_previous,
                rci_mid_current, rci_mid_previous
            )
        elif direction == 'short':
            return check_5m_entry_short_v7(
                rci_short_current, rci_short_previous,
                rci_mid_current, rci_mid_previous
            )

        return False

    def _calculate_sl_tp(self, idx: int, direction: str, entry_price: float) -> Tuple[float, float]:
        """SLとTPを計算"""
        if direction == 'long':
            zz_level = get_latest_zigzag_level(self.df_5m['ZZ_Low'], idx, lookback=50)
            if zz_level is not None:
                sl = zz_level - config.SL_BUFFER_PIPS * config.POINT_VALUE
            else:
                sl = entry_price - config.SL_DEFAULT_PIPS * config.POINT_VALUE
        else:
            zz_level = get_latest_zigzag_level(self.df_5m['ZZ_High'], idx, lookback=50)
            if zz_level is not None:
                sl = zz_level + config.SL_BUFFER_PIPS * config.POINT_VALUE
            else:
                sl = entry_price + config.SL_DEFAULT_PIPS * config.POINT_VALUE

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

        for idx in range(len(self.df_5m)):
            current_time = self.df_5m.index[idx]

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

                # 4時間足トレンドチェック
                data_4h = self._get_4h_data_at_time(current_time)
                if data_4h is None:
                    continue

                trend_4h = self._check_4h_trend(data_4h)
                if trend_4h == 'none':
                    continue

                # ダイバージェンス更新
                self._update_divergences(current_time, trend_4h)

                # 買いエントリーチェック
                if trend_4h == 'uptrend' and self._get_active_setup('long'):
                    if self._check_5m_entry(idx, 'long'):
                        self._open_position(idx, 'long')

                # 売りエントリーチェック
                elif trend_4h == 'downtrend' and self._get_active_setup('short'):
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

        # 最新のダイバージェンスタイプを取得
        div_type = 'unknown'
        for d in reversed(self.active_divergences):
            if d['direction'] == direction:
                div_type = d['type']
                break

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
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'profit_pips': trade.profit_pips,
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

        # ダイバージェンスタイプ別集計
        hidden_trades = len([t for t in self.trades if t.divergence_type == 'hidden'])
        regular_trades = len([t for t in self.trades if t.divergence_type == 'regular'])

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
            'regular_divergence_trades': regular_trades
        }
