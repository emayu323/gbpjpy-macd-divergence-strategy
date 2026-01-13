# -*- coding: utf-8 -*-
"""
改善版バックテスト実行スクリプト
- SL計算バグ修正済み
- 重要指標時間帯フィルター追加
- 複数のRR比率でテスト
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


def run_comprehensive_test(m5_data, h1_data):
    """包括的なテストを実行"""

    print("\n" + "=" * 70)
    print("改善版バックテスト - 包括的テスト")
    print("=" * 70)

    # テストする戦略の組み合わせ
    strategies = []

    # RR比率のバリエーション
    rr_ratios = [1.0, 1.5, 2.0]

    # 重要指標回避のON/OFF
    avoid_news_options = [True, False]

    # フィルター条件のバリエーション
    filter_sets = [
        {
            'name': 'ベースライン',
            'h1_rci_aligned': False,
            'trigger_macd_div': True,
        },
        {
            'name': 'RCI方向一致',
            'h1_rci_aligned': True,
            'trigger_macd_div': True,
        },
        {
            'name': 'RCI + EMA乖離率',
            'h1_rci_aligned': True,
            'm5_ema_divergence': True,
            'max_ema_divergence_pct': 1.5,
            'trigger_macd_div': True,
        },
    ]

    # すべての組み合わせを生成
    for filter_set in filter_sets:
        for rr_ratio in rr_ratios:
            for avoid_news in avoid_news_options:
                strategy = {
                    'name': f"{filter_set['name']} / RR 1:{rr_ratio} / 指標回避:{'ON' if avoid_news else 'OFF'}",
                    'rr_ratio': rr_ratio,
                    'filters': {**filter_set, 'avoid_news_times': avoid_news}
                }
                strategies.append(strategy)

    print(f"\n合計 {len(strategies)} 通りの戦略をテストします\n")

    all_results = []
    best_result = None
    best_score = -999999

    for i, strategy in enumerate(strategies, 1):
        print(f"[{i}/{len(strategies)}] {strategy['name']}")

        # RR比率を一時的に変更
        original_rr = config.RISK_REWARD_RATIO
        config.RISK_REWARD_RATIO = strategy['rr_ratio']

        # バックテスト実行
        engine = BacktestEngine(m5_data, h1_data)
        trades = engine.run_backtest(strategy['filters'])
        performance = engine.calculate_performance()

        # RR比率を元に戻す
        config.RISK_REWARD_RATIO = original_rr

        if performance is None or performance['total_trades'] == 0:
            print("  ⚠ トレードなし\n")
            continue

        # 結果表示
        print(f"  トレード数: {performance['total_trades']}")
        print(f"  勝率: {performance['win_rate']:.2f}%")
        print(f"  PF: {performance['profit_factor']:.2f}")
        print(f"  純利益: {performance['net_profit_pips']:.2f} pips\n")

        # スコア計算（勝率50%以上 かつ PF1.5以上を目標）
        score = 0
        if performance['win_rate'] >= 50:
            # 勝率クリア
            if performance['profit_factor'] >= 1.5:
                # PFもクリア → 純利益をスコアとする
                score = performance['net_profit_pips']
            else:
                # 勝率のみクリア → ペナルティ付き
                score = performance['net_profit_pips'] - 5000
        else:
            # 勝率未達 → 大きなペナルティ
            score = performance['net_profit_pips'] - 10000

        result = {
            'strategy': strategy,
            'performance': performance,
            'trades': trades,
            'score': score
        }
        all_results.append(result)

        # ベスト更新
        if score > best_score:
            best_score = score
            best_result = result

    return all_results, best_result


def display_results(all_results, best_result):
    """結果を表示"""

    print("\n" + "=" * 70)
    print("全戦略の結果一覧（スコア順）")
    print("=" * 70)

    # スコア順にソート
    all_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'#':<3} {'戦略名':<50} {'勝率':<8} {'PF':<7} {'純利益':<12}")
    print("-" * 90)

    for i, result in enumerate(all_results[:15], 1):  # 上位15件のみ表示
        perf = result['performance']
        name = result['strategy']['name']
        # 名前が長い場合は省略
        if len(name) > 48:
            name = name[:45] + "..."
        print(f"{i:<3} {name:<50} {perf['win_rate']:>6.2f}% {perf['profit_factor']:>6.2f} {perf['net_profit_pips']:>10.2f} pips")

    if len(all_results) > 15:
        print(f"\n... 他 {len(all_results) - 15} 件")

    # ベスト戦略の詳細
    print("\n" + "=" * 70)
    print("🏆 最良の戦略")
    print("=" * 70)

    if best_result is None:
        print("⚠ 目標を達成する戦略が見つかりませんでした")
        return

    strategy = best_result['strategy']
    perf = best_result['performance']

    print(f"\n【戦略名】")
    print(f"  {strategy['name']}")

    print(f"\n【パフォーマンス】")
    print(f"  総トレード数: {perf['total_trades']}")
    print(f"  勝ちトレード: {perf['win_count']}")
    print(f"  負けトレード: {perf['loss_count']}")
    print(f"  勝率: {perf['win_rate']:.2f}% {'✓ 目標達成' if perf['win_rate'] >= 50 else '✗ 目標未達'}")
    print(f"  プロフィットファクター: {perf['profit_factor']:.2f} {'✓ 目標達成' if perf['profit_factor'] >= 1.5 else '✗ 目標未達'}")
    print(f"  純利益: {perf['net_profit_pips']:.2f} pips")
    print(f"  最大ドローダウン: {perf['max_drawdown_pips']:.2f} pips")
    print(f"  平均利益: {perf['avg_win_pips']:.2f} pips")
    print(f"  平均損失: {perf['avg_loss_pips']:.2f} pips")

    print(f"\n【エントリー条件】")
    print(f"  リスクリワード比率: 1:{strategy['rr_ratio']}")
    print(f"  重要指標回避: {'ON' if strategy['filters'].get('avoid_news_times', False) else 'OFF'}")

    filters = strategy['filters']
    print(f"\n  1時間足:")
    print(f"    - パーフェクトオーダー（EMA 20 > 30 > 40）")
    if filters.get('h1_rci_aligned'):
        print(f"    - RCI 3本とも同方向")

    print(f"\n  5分足:")
    print(f"    - MACDダイバージェンス発生")
    if filters.get('m5_ema_divergence'):
        print(f"    - EMA20との乖離率が{filters.get('max_ema_divergence_pct', 2.0)}%以内")

    print(f"\n  その他:")
    print(f"    - トレード時間: {config.TRADE_START_HOUR}:00 - {config.TRADE_END_HOUR}:00 JST")
    if filters.get('avoid_news_times'):
        print(f"    - 重要指標回避時間帯:")
        for start, end in config.AVOID_NEWS_TIMES:
            print(f"      {start}:00 - {end}:00")

    # 目標達成状況のサマリー
    print(f"\n【目標達成状況】")
    win_rate_ok = perf['win_rate'] >= 50
    pf_ok = perf['profit_factor'] >= 1.5

    if win_rate_ok and pf_ok:
        print(f"  ✅ 目標を達成しました！")
        print(f"     勝率50%以上 ✓")
        print(f"     プロフィットファクター1.5以上 ✓")
    elif win_rate_ok:
        print(f"  ⚠️  勝率は達成しましたが、PFが目標未達です")
        print(f"     勝率50%以上 ✓")
        print(f"     プロフィットファクター1.5以上 ✗ (現在: {perf['profit_factor']:.2f})")
    else:
        print(f"  ❌ 目標未達成")
        print(f"     勝率50%以上 ✗ (現在: {perf['win_rate']:.2f}%)")
        print(f"     プロフィットファクター1.5以上 {'✓' if pf_ok else '✗'} (現在: {perf['profit_factor']:.2f})")


def main():
    print("\n" + "=" * 70)
    print("GBPJPY トレード戦略 改善版バックテスト")
    print("=" * 70)
    print("\n【改善内容】")
    print("  ✓ SL計算バグを修正")
    print("  ✓ 重要指標発表前後の回避時間帯を追加")
    print("  ✓ 複数のリスクリワード比率でテスト")
    print("  ✓ ZigZagレベルの妥当性チェックを強化")
    print()

    # データ読み込み
    m5_data, h1_data = load_data_quick()

    # 包括的テスト実行
    all_results, best_result = run_comprehensive_test(m5_data, h1_data)

    # 結果表示
    display_results(all_results, best_result)

    print("\n" + "=" * 70)
    print("分析完了")
    print("=" * 70)


if __name__ == '__main__':
    main()
