# -*- coding: utf-8 -*-
"""
レポート生成モジュール
バックテスト結果の可視化とレポート出力
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os


def plot_equity_curve(trades, output_path='results/equity_curve.png'):
    """
    エクイティカーブを描画

    Parameters:
    -----------
    trades : list
        トレード結果のリスト
    output_path : str
        出力ファイルパス
    """
    if len(trades) == 0:
        print("トレードデータがありません")
        return

    df = pd.DataFrame(trades)
    df['cumulative_pips'] = df['pips'].cumsum()

    plt.figure(figsize=(14, 7))
    plt.plot(df['exit_time'], df['cumulative_pips'], linewidth=2, color='#2E86AB')
    plt.fill_between(df['exit_time'], df['cumulative_pips'], alpha=0.3, color='#2E86AB')

    plt.title('Equity Curve (累積損益)', fontsize=16, fontweight='bold')
    plt.xlabel('日時', fontsize=12)
    plt.ylabel('累積pips', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ディレクトリ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"エクイティカーブを保存しました: {output_path}")
    plt.close()


def plot_win_loss_distribution(trades, output_path='results/win_loss_distribution.png'):
    """
    勝ちトレード・負けトレードの分布を描画

    Parameters:
    -----------
    trades : list
        トレード結果のリスト
    output_path : str
        出力ファイルパス
    """
    if len(trades) == 0:
        print("トレードデータがありません")
        return

    df = pd.DataFrame(trades)
    wins = df[df['result'] == 'win']['pips']
    losses = df[df['result'] == 'loss']['pips']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 勝ちトレード分布
    axes[0].hist(wins, bins=20, color='#06A77D', alpha=0.7, edgecolor='black')
    axes[0].set_title('勝ちトレードの分布', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('pips', fontsize=12)
    axes[0].set_ylabel('頻度', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(wins.mean(), color='red', linestyle='--', linewidth=2, label=f'平均: {wins.mean():.2f} pips')
    axes[0].legend()

    # 負けトレード分布
    axes[1].hist(losses, bins=20, color='#D00000', alpha=0.7, edgecolor='black')
    axes[1].set_title('負けトレードの分布', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('pips', fontsize=12)
    axes[1].set_ylabel('頻度', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(losses.mean(), color='blue', linestyle='--', linewidth=2, label=f'平均: {losses.mean():.2f} pips')
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"勝敗分布グラフを保存しました: {output_path}")
    plt.close()


def plot_monthly_performance(trades, output_path='results/monthly_performance.png'):
    """
    月次パフォーマンスを描画

    Parameters:
    -----------
    trades : list
        トレード結果のリスト
    output_path : str
        出力ファイルパス
    """
    if len(trades) == 0:
        print("トレードデータがありません")
        return

    df = pd.DataFrame(trades)
    df['month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')

    monthly = df.groupby('month')['pips'].sum().reset_index()
    monthly['month'] = monthly['month'].astype(str)

    colors = ['#06A77D' if x > 0 else '#D00000' for x in monthly['pips']]

    plt.figure(figsize=(16, 6))
    plt.bar(monthly['month'], monthly['pips'], color=colors, edgecolor='black', alpha=0.8)
    plt.title('月次パフォーマンス', fontsize=16, fontweight='bold')
    plt.xlabel('月', fontsize=12)
    plt.ylabel('pips', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"月次パフォーマンスグラフを保存しました: {output_path}")
    plt.close()


def generate_csv_report(trades, performance, output_path='results/backtest_report.csv'):
    """
    トレード詳細をCSVで出力

    Parameters:
    -----------
    trades : list
        トレード結果のリスト
    performance : dict
        パフォーマンス指標
    output_path : str
        出力ファイルパス
    """
    if len(trades) == 0:
        print("トレードデータがありません")
        return

    df = pd.DataFrame(trades)

    # CSVに保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"トレード詳細CSVを保存しました: {output_path}")


def generate_summary_report(performance, filters, output_path='results/summary.txt'):
    """
    サマリーレポートをテキストファイルで出力

    Parameters:
    -----------
    performance : dict
        パフォーマンス指標
    filters : dict
        使用したフィルター条件
    output_path : str
        出力ファイルパス
    """
    if performance is None:
        print("パフォーマンスデータがありません")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("GBPJPY トレード戦略バックテスト結果サマリー\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"戦略名: {filters.get('name', '不明')}\n\n")

        f.write("【パフォーマンス】\n")
        f.write(f"  総トレード数: {performance['total_trades']}\n")
        f.write(f"  勝ちトレード: {performance['win_count']}\n")
        f.write(f"  負けトレード: {performance['loss_count']}\n")
        f.write(f"  勝率: {performance['win_rate']:.2f}%\n")
        f.write(f"  プロフィットファクター: {performance['profit_factor']:.2f}\n")
        f.write(f"  純利益: {performance['net_profit_pips']:.2f} pips\n")
        f.write(f"  最大ドローダウン: {performance['max_drawdown_pips']:.2f} pips\n")
        f.write(f"  平均利益: {performance['avg_win_pips']:.2f} pips\n")
        f.write(f"  平均損失: {performance['avg_loss_pips']:.2f} pips\n\n")

        f.write("【エントリー条件】\n")
        f.write("1時間足:\n")
        f.write("  - パーフェクトオーダー（EMA 20 > 30 > 40）\n")
        if filters.get('h1_rci_aligned'):
            f.write("  - RCI 3本とも同方向\n")
        if filters.get('h1_hidden_div'):
            f.write("  - ヒドゥンダイバージェンス発生\n")

        f.write("\n5分足:\n")
        if filters.get('trigger_macd_div'):
            f.write("  - MACDダイバージェンス発生\n")
        if filters.get('trigger_rci_reversal'):
            f.write("  - RCI反転（±80ラインブレイク）\n")
        if filters.get('trigger_zigzag_break'):
            f.write("  - ZigZag短期の方向転換\n")
        if filters.get('m5_ema_divergence'):
            f.write(f"  - EMA20との乖離率が{filters.get('max_ema_divergence_pct', 2.0)}%以内\n")

        f.write("\n【損益設定】\n")
        f.write("  - 損切り: ZigZag短期の直近高値/安値\n")
        f.write("  - 利確: 損切り幅 × 2.0\n")
        f.write("  - スプレッド: 1.5 pips\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"サマリーレポートを保存しました: {output_path}")


def generate_full_report(trades, performance, filters):
    """
    全レポートを生成

    Parameters:
    -----------
    trades : list
        トレード結果のリスト
    performance : dict
        パフォーマンス指標
    filters : dict
        使用したフィルター条件
    """
    print("\n" + "=" * 60)
    print("レポート生成中...")
    print("=" * 60)

    # グラフ生成
    plot_equity_curve(trades)
    plot_win_loss_distribution(trades)
    plot_monthly_performance(trades)

    # CSV出力
    generate_csv_report(trades, performance)

    # サマリー出力
    generate_summary_report(performance, filters)

    print("\nすべてのレポートが生成されました")
    print("=" * 60)
