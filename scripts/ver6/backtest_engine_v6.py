"""
System Ver6 - バックテストエンジン
マルチタイムフレーム対応のバックテスト実行
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import config_v6 as config
from indicators_v6 import (
    calculate_rci, calculate_ema, calculate_zigzag,
    get_latest_zigzag_level, check_perfect_order,
    check_1h_setup_long, check_1h_setup_short,
    check_5m_entry_long, check_5m_entry_short
)


@dataclass
class Trade:
    """トレード記録"""
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    profit_pips: float
    result: str  # 'win', 'loss', 'breakeven'
    exit_reason: str  # 'tp', 'sl', 'time'


class BacktestEngineV6:
    """System Ver6 バックテストエンジン"""

    def __init__(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame):
        """
        Parameters:
        -----------
        df_5m : pd.DataFrame
            5分足データ
        df_1h : pd.DataFrame
            1時間足データ
        df_4h : pd.DataFrame
            4時間足データ
        """
        self.df_5m = df_5m.copy()
        self.df_1h = df_1h.copy()
        self.df_4h = df_4h.copy()

        self.trades: List[Trade] = []
        self.current_position: Optional[Dict] = None

        # インジケーター計算
        self._calculate_indicators()

    def _calculate_indicators(self):
        """全タイムフレームのインジケーターを計算"""
        print("インジケーターを計算中...")

        # 5分足のRCI
        self.df_5m['RCI_9'] = calculate_rci(self.df_5m, config.RCI_SHORT)
        self.df_5m['RCI_14'] = calculate_rci(self.df_5m, config.RCI_MID)

        # 1時間足のRCI
        self.df_1h['RCI_14'] = calculate_rci(self.df_1h, config.RCI_MID)
        self.df_1h['RCI_18'] = calculate_rci(self.df_1h, config.RCI_LONG)

        # 4時間足のEMA
        self.df_4h['EMA_20'] = calculate_ema(self.df_4h, config.EMA_SHORT)
        self.df_4h['EMA_30'] = calculate_ema(self.df_4h, config.EMA_MID)
        self.df_4h['EMA_40'] = calculate_ema(self.df_4h, config.EMA_LONG)

        # 5分足のZigZag（SL設定用）
        zz_highs, zz_lows = calculate_zigzag(
            self.df_5m,
            depth=config.ZIGZAG_DEPTH,
            deviation=config.ZIGZAG_DEVIATION,
            backstep=config.ZIGZAG_BACKSTEP
        )
        self.df_5m['ZZ_High'] = zz_highs
        self.df_5m['ZZ_Low'] = zz_lows

        print("インジケーター計算完了")

    def _get_1h_data_at_time(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """指定時刻の1時間足データを取得"""
        # timestamp以前の最新の1時間足を取得
        mask = self.df_1h.index <= timestamp
        if mask.sum() == 0:
            return None
        return self.df_1h[mask].iloc[-1]

    def _get_4h_data_at_time(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """指定時刻の4時間足データを取得"""
        # timestamp以前の最新の4時間足を取得
        mask = self.df_4h.index <= timestamp
        if mask.sum() == 0:
            return None
        return self.df_4h[mask].iloc[-1]

    def _get_1h_previous_data(self, current_1h: pd.Series) -> Optional[pd.Series]:
        """1時間足の1本前のデータを取得"""
        idx = self.df_1h.index.get_loc(current_1h.name)
        if idx == 0:
            return None
        return self.df_1h.iloc[idx - 1]

    def _check_4h_trend(self, data_4h: pd.Series) -> str:
        """4時間足のトレンドをチェック"""
        return check_perfect_order(
            data_4h['EMA_20'],
            data_4h['EMA_30'],
            data_4h['EMA_40']
        )

    def _check_1h_setup(self, data_1h_current: pd.Series, data_1h_previous: pd.Series,
                        trend_4h: str) -> Optional[str]:
        """
        1時間足のセットアップをチェック

        Returns:
        --------
        Optional[str]
            'long': 買いセットアップ
            'short': 売りセットアップ
            None: セットアップなし
        """
        if data_1h_previous is None:
            return None

        rci_mid_current = data_1h_current['RCI_14']
        rci_mid_previous = data_1h_previous['RCI_14']

        # 4時間足が上昇トレンド → 買いセットアップをチェック
        if trend_4h == 'uptrend':
            if check_1h_setup_long(rci_mid_current, rci_mid_previous):
                return 'long'

        # 4時間足が下降トレンド → 売りセットアップをチェック
        elif trend_4h == 'downtrend':
            if check_1h_setup_short(rci_mid_current, rci_mid_previous):
                return 'short'

        return None

    def _check_5m_entry(self, idx: int, setup_direction: str) -> bool:
        """
        5分足のエントリートリガーをチェック

        Parameters:
        -----------
        idx : int
            5分足のインデックス
        setup_direction : str
            セットアップ方向 ('long' or 'short')

        Returns:
        --------
        bool
            エントリー条件を満たす場合True
        """
        if idx == 0:
            return False

        current = self.df_5m.iloc[idx]
        previous = self.df_5m.iloc[idx - 1]

        rci_short_current = current['RCI_9']
        rci_short_previous = previous['RCI_9']
        rci_mid_current = current['RCI_14']
        rci_mid_previous = previous['RCI_14']

        if setup_direction == 'long':
            return check_5m_entry_long(
                rci_short_current, rci_short_previous,
                rci_mid_current, rci_mid_previous,
                threshold=config.RCI_OVERSOLD
            )
        elif setup_direction == 'short':
            return check_5m_entry_short(
                rci_short_current, rci_short_previous,
                rci_mid_current, rci_mid_previous,
                threshold=config.RCI_OVERBOUGHT
            )

        return False

    def _calculate_sl_tp(self, idx: int, direction: str, entry_price: float) -> Tuple[float, float]:
        """
        SLとTPを計算

        Parameters:
        -----------
        idx : int
            エントリー時の5分足インデックス
        direction : str
            'long' or 'short'
        entry_price : float
            エントリー価格

        Returns:
        --------
        Tuple[float, float]
            (stop_loss, take_profit)
        """
        # ZigZagレベルを取得
        if direction == 'long':
            zz_level = get_latest_zigzag_level(self.df_5m['ZZ_Low'], idx, lookback=50)
            if zz_level is not None:
                sl = zz_level - config.SL_BUFFER_PIPS * config.POINT_VALUE
            else:
                sl = entry_price - config.SL_DEFAULT_PIPS * config.POINT_VALUE
        else:  # short
            zz_level = get_latest_zigzag_level(self.df_5m['ZZ_High'], idx, lookback=50)
            if zz_level is not None:
                sl = zz_level + config.SL_BUFFER_PIPS * config.POINT_VALUE
            else:
                sl = entry_price + config.SL_DEFAULT_PIPS * config.POINT_VALUE

        # リスク幅を計算
        risk = abs(entry_price - sl)

        # TP = Entry + (Risk × RR)
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
        """
        ポジションの決済条件をチェック

        Returns:
        --------
        Optional[Tuple[str, float]]
            (exit_reason, exit_price) or None
        """
        if self.current_position is None:
            return None

        current = self.df_5m.iloc[idx]
        direction = self.current_position['direction']
        sl = self.current_position['stop_loss']
        tp = self.current_position['take_profit']

        if direction == 'long':
            # TP到達
            if current['High'] >= tp:
                return ('tp', tp)
            # SL到達
            if current['Low'] <= sl:
                return ('sl', sl)
        else:  # short
            # TP到達
            if current['Low'] <= tp:
                return ('tp', tp)
            # SL到達
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

            # 新規エントリーチェック（ポジションがない場合のみ）
            if self.current_position is None:
                # 取引時間帯チェック
                if not self._is_trading_hours(current_time):
                    continue

                # 1時間足データを取得
                data_1h = self._get_1h_data_at_time(current_time)
                if data_1h is None:
                    continue

                data_1h_prev = self._get_1h_previous_data(data_1h)
                if data_1h_prev is None:
                    continue

                # 4時間足データを取得
                data_4h = self._get_4h_data_at_time(current_time)
                if data_4h is None:
                    continue

                # 4時間足トレンドチェック
                trend_4h = self._check_4h_trend(data_4h)
                if trend_4h == 'none':
                    continue

                # 1時間足セットアップチェック
                setup = self._check_1h_setup(data_1h, data_1h_prev, trend_4h)
                if setup is None:
                    continue

                # 5分足エントリートリガーチェック
                if self._check_5m_entry(idx, setup):
                    self._open_position(idx, setup)

            # 進捗表示
            if idx % 10000 == 0:
                print(f"進捗: {idx}/{len(self.df_5m)} ({idx/len(self.df_5m)*100:.1f}%)")

        # 最後にポジションが残っている場合は強制決済
        if self.current_position is not None:
            last_time = self.df_5m.index[-1]
            last_price = self.df_5m.iloc[-1]['Close']
            self._close_position(last_time, last_price, 'end_of_data')

        print(f"バックテスト完了: {len(self.trades)}件のトレード")

        return self._create_trades_dataframe()

    def _open_position(self, idx: int, direction: str):
        """ポジションを開く"""
        current = self.df_5m.iloc[idx]
        entry_price = current['Close']
        entry_time = current.name

        # SL/TP計算
        sl, tp = self._calculate_sl_tp(idx, direction, entry_price)

        self.current_position = {
            'direction': direction,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'stop_loss': sl,
            'take_profit': tp,
            'entry_idx': idx
        }

    def _close_position(self, exit_time: datetime, exit_price: float, exit_reason: str):
        """ポジションを閉じる"""
        pos = self.current_position
        direction = pos['direction']
        entry_price = pos['entry_price']

        # 利益計算（pips）
        if direction == 'long':
            profit_pips = (exit_price - entry_price) / config.POINT_VALUE
        else:
            profit_pips = (entry_price - exit_price) / config.POINT_VALUE

        # 結果判定
        if profit_pips > 0:
            result = 'win'
        elif profit_pips < 0:
            result = 'loss'
        else:
            result = 'breakeven'

        # トレード記録
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
            exit_reason=exit_reason
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
                'exit_reason': trade.exit_reason
            })

        return pd.DataFrame(trades_data)

    def get_statistics(self) -> Dict:
        """統計情報を計算"""
        if not self.trades:
            return {}

        df_trades = self._create_trades_dataframe()

        total_trades = len(self.trades)
        wins = len([t for t in self.trades if t.result == 'win'])
        losses = len([t for t in self.trades if t.result == 'loss'])

        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        total_profit = sum(t.profit_pips for t in self.trades)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.profit_pips for t in self.trades if t.profit_pips > 0)
        gross_loss = abs(sum(t.profit_pips for t in self.trades if t.profit_pips < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # TP/SL別集計
        tp_exits = len([t for t in self.trades if t.exit_reason == 'tp'])
        sl_exits = len([t for t in self.trades if t.exit_reason == 'sl'])

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
            'avg_loss': gross_loss / losses if losses > 0 else 0
        }
