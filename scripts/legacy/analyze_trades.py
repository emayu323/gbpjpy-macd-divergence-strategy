# -*- coding: utf-8 -*-
"""
トレードデータ分析スクリプト
問題点を特定するための詳細分析
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import config
import indicators
from backtest_engine import BacktestEngine


def load_data_quick():
    """データを素早く読み込み（最近2年間）"""
    print("データ読み込み中...")

    h1_data = pd.read_csv(config.H1_DATA_FILE)
    m5_data = pd.read_csv(config.M5_DATA_FILE)

    h1_data.columns = h1_data.columns.str.lower()
    m5_data.columns = m5_data.columns.str.lower()

    time_col = 'local time' if 'local time' in h1_data.columns else 'time'

    h1_data['time'] = pd.to_datetime(h1_data[time_col], dayfirst=True)
    m5_data['time'] = pd.to_datetime(m5_data[time_col], dayfirst=True)

    jst = pytz.timezone(config.TIMEZONE)
    h1_data['time'] = h1_data['time'].dt.tz_convert(jst)
    m5_data['time'] = m5_data['time'].dt.tz_convert(jst)

    # 最近2年間
    cutoff_date = datetime.now(jst) - timedelta(days=730)
    h1_data = h1_data[h1_data['time'] >= cutoff_date].copy()
    m5_data = m5_data[m5_data['time'] >= cutoff_date].copy()

    h1_data.set_index('time', inplace=True)
    m5_data.set_index('time', inplace=True)

    if time_col in h1_data.columns:
        h1_data.drop(columns=[time_col], inplace=True)
    if time_col in m5_data.columns:
        m5_data.drop(columns=[time_col], inplace=True)

    h1_data.sort_index(inplace=True)
    m5_data.sort_index(inplace=True)

    print("指標計算中...")
    h1_data = indicators.add_all_indicators(
        h1_data, config.EMA_PERIODS, config.RCI_PERIODS, config.MACD_PARAMS,
        config.ZIGZAG_SHORT, config.ZIGZAG_LONG
    )
    m5_data = indicators.add_all_indicators(
        m5_data, config.EMA_PERIODS, config.RCI_PERIODS, config.MACD_PARAMS,
        config.ZIGZAG_SHORT, config.ZIGZAG_LONG
    )

    return m5_data, h1_data


def analyze_trades_detail(trades):
    """トレードの詳細分析"""
    if len(trades) == 0:
        print("トレードがありません")
        return

    df = pd.DataFrame(trades)

    print("\n" + "=" * 60)
    print("トレード詳細分析")
    print("=" * 60)

    # SL幅の分布
    print("\n【SL幅の分布】")
    print(f"  平均SL幅: {df['sl_pips'].mean():.2f} pips")
    print(f"  中央値: {df['sl_pips'].median():.2f} pips")
    print(f"  最小: {df['sl_pips'].min():.2f} pips")
    print(f"  最大: {df['sl_pips'].max():.2f} pips")
    print(f"  25%点: {df['sl_pips'].quantile(0.25):.2f} pips")
    print(f"  75%点: {df['sl_pips'].quantile(0.75):.2f} pips")

    # SL幅が大きいトレード
    large_sl = df[df['sl_pips'] > 30]
    print(f"\n  SL幅が30pips超のトレード数: {len(large_sl)} ({len(large_sl)/len(df)*100:.1f}%)")

    # 方向別の勝率
    print("\n【方向別の勝率】")
    for direction in ['long', 'short']:
        dir_trades = df[df['direction'] == direction]
        if len(dir_trades) > 0:
            wins = len(dir_trades[dir_trades['result'] == 'win'])
            win_rate = wins / len(dir_trades) * 100
            print(f"  {direction.upper()}: {win_rate:.2f}% (勝ち{wins}/{len(dir_trades)})")

    # 勝ちトレードと負けトレードの比較
    print("\n【勝敗の比較】")
    wins = df[df['result'] == 'win']
    losses = df[df['result'] == 'loss']

    print(f"  勝ちトレードの平均pips: {wins['pips'].mean():.2f}")
    print(f"  負けトレードの平均pips: {losses['pips'].mean():.2f}")
    print(f"  理論上の平均利益（SL×2）: {(wins['sl_pips'] * 2).mean():.2f}")
    print(f"  理論上の平均損失（-SL）: {-losses['sl_pips'].mean():.2f}")

    # サンプルトレード
    print("\n【サンプルトレード（最初の5件）】")
    for i, trade in df.head(5).iterrows():
        print(f"\n  トレード #{i+1}")
        print(f"    エントリー: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}")
        print(f"    方向: {trade['direction']}")
        print(f"    価格: {trade['entry_price']:.3f}")
        print(f"    SL: {trade['sl_price']:.3f} ({trade['sl_pips']:.2f} pips)")
        print(f"    TP: {trade['tp_price']:.3f}")
        print(f"    結果: {trade['result']} ({trade['pips']:.2f} pips)")
        print(f"    保有時間: {(trade['exit_time'] - trade['entry_time'])}")


def test_alternative_strategies(m5_data, h1_data):
    """代替戦略をテスト"""
    print("\n" + "=" * 60)
    print("代替戦略のテスト")
    print("=" * 60)

    # より緩い条件での戦略
    strategies = [
        {
            'name': 'リスクリワード1:1.5（緩和版）',
            'rr_ratio': 1.5,
            'filters': {
                'h1_rci_aligned': True,
                'h1_hidden_div': False,
                'm5_ema_divergence': False,
                'trigger_macd_div': True,
                'trigger_rci_reversal': False,
                'trigger_zigzag_break': False,
            }
        },
        {
            'name': 'リスクリワード1:1（さらに緩和）',
            'rr_ratio': 1.0,
            'filters': {
                'h1_rci_aligned': True,
                'h1_hidden_div': False,
                'm5_ema_divergence': False,
                'trigger_macd_div': True,
                'trigger_rci_reversal': False,
                'trigger_zigzag_break': False,
            }
        },
    ]

    results = []

    for strategy in strategies:
        print(f"\n【{strategy['name']}】")

        # 一時的にRR比率を変更
        original_rr = config.RISK_REWARD_RATIO
        config.RISK_REWARD_RATIO = strategy['rr_ratio']

        engine = BacktestEngine(m5_data, h1_data)
        trades = engine.run_backtest(strategy['filters'])
        performance = engine.calculate_performance()

        # RR比率を元に戻す
        config.RISK_REWARD_RATIO = original_rr

        if performance:
            print(f"  総トレード数: {performance['total_trades']}")
            print(f"  勝率: {performance['win_rate']:.2f}%")
            print(f"  プロフィットファクター: {performance['profit_factor']:.2f}")
            print(f"  純利益: {performance['net_profit_pips']:.2f} pips")

            results.append({
                'strategy': strategy['name'],
                'performance': performance,
                'trades': trades
            })

    return results


def main():
    print("\n" + "=" * 60)
    print("トレード詳細分析")
    print("=" * 60)

    # データ読み込み
    m5_data, h1_data = load_data_quick()

    # ベースライン戦略でトレード実行
    print("\nベースライン戦略でトレード実行中...")
    filters = {
        'h1_rci_aligned': True,
        'h1_hidden_div': False,
        'm5_ema_divergence': False,
        'trigger_macd_div': True,
        'trigger_rci_reversal': False,
        'trigger_zigzag_break': False,
    }

    engine = BacktestEngine(m5_data, h1_data)
    trades = engine.run_backtest(filters)

    # トレード詳細分析
    analyze_trades_detail(trades)

    # 代替戦略のテスト
    alt_results = test_alternative_strategies(m5_data, h1_data)

    # 最良の結果を探す
    print("\n" + "=" * 60)
    print("代替戦略の比較")
    print("=" * 60)

    for result in alt_results:
        perf = result['performance']
        print(f"\n{result['strategy']}:")
        print(f"  勝率: {perf['win_rate']:.2f}% | PF: {perf['profit_factor']:.2f} | 純利益: {perf['net_profit_pips']:.2f} pips")

        # 目標達成チェック
        if perf['win_rate'] >= 50 and perf['profit_factor'] >= 1.5:
            print("  ✓ 目標に近い結果です！")


if __name__ == '__main__':
    main()
