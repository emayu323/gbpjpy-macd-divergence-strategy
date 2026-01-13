# -*- coding: utf-8 -*-
"""
バックテスト分析メインスクリプト（軽量版）
最近2年間のデータのみを使用して高速にテスト
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import config
import indicators
from backtest_engine import BacktestEngine


def load_and_preprocess_data():
    """
    データを読み込んで前処理を実行（最近2年間のみ）

    Returns:
    --------
    tuple
        (m5_data, h1_data) - 前処理済みのデータフレーム
    """
    print("=" * 60)
    print("データ読み込み開始（最近2年間のみ）")
    print("=" * 60)

    # 1時間足データ読み込み
    print(f"1時間足データを読み込み中: {config.H1_DATA_FILE}")
    h1_data = pd.read_csv(config.H1_DATA_FILE)

    # 5分足データ読み込み
    print(f"5分足データを読み込み中: {config.M5_DATA_FILE}")
    m5_data = pd.read_csv(config.M5_DATA_FILE)

    print(f"1時間足データ（全期間）: {len(h1_data):,} 行")
    print(f"5分足データ（全期間）: {len(m5_data):,} 行")

    # カラム名を統一（小文字化）
    h1_data.columns = h1_data.columns.str.lower()
    m5_data.columns = m5_data.columns.str.lower()

    # タイムスタンプカラムの名前を確認
    time_col = 'local time' if 'local time' in h1_data.columns else 'time'

    # タイムスタンプをインデックスに設定（dayfirst=Trueで警告を回避）
    h1_data['time'] = pd.to_datetime(h1_data[time_col], dayfirst=True)
    m5_data['time'] = pd.to_datetime(m5_data[time_col], dayfirst=True)

    # 日本時間に変換（CSVデータは既にGMT+0900のタイムゾーン情報を含む）
    jst = pytz.timezone(config.TIMEZONE)
    # 既にtz-awareなので、tz_convertで日本時間に変換
    h1_data['time'] = h1_data['time'].dt.tz_convert(jst)
    m5_data['time'] = m5_data['time'].dt.tz_convert(jst)

    # **最近2年間のデータのみに絞る**
    cutoff_date = datetime.now(jst) - timedelta(days=730)  # 2年前
    print(f"\nデータを {cutoff_date.date()} 以降に制限します...")
    h1_data = h1_data[h1_data['time'] >= cutoff_date].copy()
    m5_data = m5_data[m5_data['time'] >= cutoff_date].copy()

    print(f"1時間足データ（2年間）: {len(h1_data):,} 行")
    print(f"5分足データ（2年間）: {len(m5_data):,} 行")

    # インデックスに設定
    h1_data.set_index('time', inplace=True)
    m5_data.set_index('time', inplace=True)

    # 元のtime列を削除（重複を避ける）
    if time_col in h1_data.columns:
        h1_data.drop(columns=[time_col], inplace=True)
    if time_col in m5_data.columns:
        m5_data.drop(columns=[time_col], inplace=True)

    # データを時系列順にソート
    h1_data.sort_index(inplace=True)
    m5_data.sort_index(inplace=True)

    print("\n" + "=" * 60)
    print("指標計算開始")
    print("=" * 60)

    # 1時間足の指標計算
    print("1時間足の指標を計算中...")
    h1_data = indicators.add_all_indicators(
        h1_data,
        ema_periods=config.EMA_PERIODS,
        rci_periods=config.RCI_PERIODS,
        macd_params=config.MACD_PARAMS,
        zigzag_short_params=config.ZIGZAG_SHORT,
        zigzag_long_params=config.ZIGZAG_LONG
    )
    print("  ✓ 1時間足の指標計算完了")

    # 5分足の指標計算
    print("5分足の指標を計算中（これには数分かかる場合があります）...")
    m5_data = indicators.add_all_indicators(
        m5_data,
        ema_periods=config.EMA_PERIODS,
        rci_periods=config.RCI_PERIODS,
        macd_params=config.MACD_PARAMS,
        zigzag_short_params=config.ZIGZAG_SHORT,
        zigzag_long_params=config.ZIGZAG_LONG
    )
    print("  ✓ 5分足の指標計算完了")

    print("\nデータ前処理完了\n")

    return m5_data, h1_data


def run_optimization(m5_data, h1_data):
    """
    エントリー条件の最適化を実行（軽量版：主要な戦略のみテスト）

    Parameters:
    -----------
    m5_data : pd.DataFrame
        5分足データ
    h1_data : pd.DataFrame
        1時間足データ

    Returns:
    --------
    tuple
        (best_filters, best_performance, all_results)
    """
    print("=" * 60)
    print("エントリー条件の最適化開始（主要戦略のみ）")
    print("=" * 60)

    # テストするフィルター条件の組み合わせ（主要なもののみ）
    filter_combinations = [
        # ベースライン
        {
            'name': 'ベースライン（MACDダイバージェンスのみ）',
            'h1_rci_aligned': False,
            'h1_hidden_div': False,
            'm5_ema_divergence': False,
            'trigger_macd_div': True,
            'trigger_rci_reversal': False,
            'trigger_zigzag_break': False,
        },
        # RCI追加
        {
            'name': '1時間足RCI方向一致',
            'h1_rci_aligned': True,
            'h1_hidden_div': False,
            'm5_ema_divergence': False,
            'trigger_macd_div': True,
            'trigger_rci_reversal': False,
            'trigger_zigzag_break': False,
        },
        # RCI + EMA乖離率
        {
            'name': '1時間足RCI + 5分足EMA乖離率',
            'h1_rci_aligned': True,
            'h1_hidden_div': False,
            'm5_ema_divergence': True,
            'max_ema_divergence_pct': 2.0,
            'trigger_macd_div': True,
            'trigger_rci_reversal': False,
            'trigger_zigzag_break': False,
        },
        # RCI反転トリガー追加
        {
            'name': 'RCI + EMA乖離率 + RCI反転トリガー',
            'h1_rci_aligned': True,
            'h1_hidden_div': False,
            'm5_ema_divergence': True,
            'max_ema_divergence_pct': 2.0,
            'trigger_macd_div': True,
            'trigger_rci_reversal': True,
            'trigger_zigzag_break': False,
        },
    ]

    all_results = []
    best_performance = None
    best_filters = None
    best_score = -999999

    for i, filters in enumerate(filter_combinations, 1):
        print(f"\n[{i}/{len(filter_combinations)}] テスト中: {filters['name']}")

        # バックテストエンジン初期化
        engine = BacktestEngine(m5_data, h1_data)

        # バックテスト実行
        trades = engine.run_backtest(filters)

        # パフォーマンス計算
        performance = engine.calculate_performance()

        if performance is None:
            print(f"  ⚠ トレードなし")
            continue

        # 結果表示
        print(f"  総トレード数: {performance['total_trades']}")
        print(f"  勝率: {performance['win_rate']:.2f}%")
        print(f"  プロフィットファクター: {performance['profit_factor']:.2f}")
        print(f"  純利益: {performance['net_profit_pips']:.2f} pips")

        # スコア計算
        score = 0
        if performance['win_rate'] >= 50 and performance['profit_factor'] >= 2.0:
            # 目標達成
            score = performance['net_profit_pips']
        else:
            # 目標未達成の場合はペナルティ
            score = performance['net_profit_pips'] - 10000

        # 結果を記録
        result = {
            'filters': filters,
            'performance': performance,
            'score': score,
            'trades': trades
        }
        all_results.append(result)

        # ベスト更新
        if score > best_score:
            best_score = score
            best_performance = performance
            best_filters = filters

    print("\n" + "=" * 60)
    print("最適化完了")
    print("=" * 60)

    return best_filters, best_performance, all_results


def main():
    """メイン処理"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "GBPJPY トレード戦略バックテスト（軽量版）" + " " * 7 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    # データ読み込みと前処理
    m5_data, h1_data = load_and_preprocess_data()

    # 最適化実行
    best_filters, best_performance, all_results = run_optimization(m5_data, h1_data)

    # 結果表示
    print("\n" + "=" * 60)
    print("最適な戦略")
    print("=" * 60)

    if best_filters is None:
        print("⚠ 目標を達成する戦略が見つかりませんでした")
        print("\n全戦略の結果:")
        for result in all_results:
            perf = result['performance']
            name = result['filters']['name']
            print(f"\n{name}:")
            print(f"  勝率: {perf['win_rate']:.2f}%, PF: {perf['profit_factor']:.2f}, 純利益: {perf['net_profit_pips']:.2f} pips")
        return

    print(f"\n戦略名: {best_filters['name']}\n")

    print("【パフォーマンス】")
    print(f"  総トレード数: {best_performance['total_trades']}")
    print(f"  勝ちトレード: {best_performance['win_count']}")
    print(f"  負けトレード: {best_performance['loss_count']}")
    print(f"  勝率: {best_performance['win_rate']:.2f}%")
    print(f"  プロフィットファクター: {best_performance['profit_factor']:.2f}")
    print(f"  純利益: {best_performance['net_profit_pips']:.2f} pips")
    print(f"  最大ドローダウン: {best_performance['max_drawdown_pips']:.2f} pips")
    print(f"  平均利益: {best_performance['avg_win_pips']:.2f} pips")
    print(f"  平均損失: {best_performance['avg_loss_pips']:.2f} pips")

    print("\n【エントリー条件】")
    print("1時間足:")
    print(f"  - パーフェクトオーダー（EMA 20 > 30 > 40）")
    if best_filters['h1_rci_aligned']:
        print(f"  - RCI 3本とも同方向")
    if best_filters.get('h1_hidden_div'):
        print(f"  - ヒドゥンダイバージェンス発生")

    print("\n5分足:")
    if best_filters['trigger_macd_div']:
        print(f"  - MACDダイバージェンス発生")
    if best_filters.get('trigger_rci_reversal'):
        print(f"  - RCI反転（±80ラインブレイク）")
    if best_filters.get('trigger_zigzag_break'):
        print(f"  - ZigZag短期の方向転換")
    if best_filters.get('m5_ema_divergence'):
        print(f"  - EMA20との乖離率が{best_filters.get('max_ema_divergence_pct', 2.0)}%以内")

    print("\n【損益設定】")
    print(f"  - 損切り: ZigZag短期の直近高値/安値")
    print(f"  - 利確: 損切り幅 × {config.RISK_REWARD_RATIO}")
    print(f"  - スプレッド: {config.SPREAD_PIPS} pips")

    print("\n" + "=" * 60)
    print("全戦略の結果比較")
    print("=" * 60)

    # 全結果をソート
    all_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'順位':<4} {'戦略名':<40} {'勝率':<8} {'PF':<8} {'純利益':<12}")
    print("-" * 80)

    for rank, result in enumerate(all_results, 1):
        perf = result['performance']
        name = result['filters']['name']
        print(f"{rank:<4} {name:<40} {perf['win_rate']:>6.2f}% {perf['profit_factor']:>6.2f} {perf['net_profit_pips']:>10.2f} pips")

    print("\n" + "=" * 60)
    print("分析完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
