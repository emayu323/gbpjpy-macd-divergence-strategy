"""
System Ver6 - バックテスト実行スクリプト
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

import config_v6 as config
from backtest_engine_v6 import BacktestEngineV6


def load_data() -> tuple:
    """データを読み込み"""
    print("=" * 60)
    print("データ読み込み中...")
    print("=" * 60)

    # 5分足データ
    print(f"5分足データ: {config.DATA_5M}")
    df_5m = pd.read_csv(config.DATA_5M)
    df_5m['Local time'] = pd.to_datetime(df_5m['Local time'], dayfirst=True)
    df_5m.set_index('Local time', inplace=True)
    print(f"  期間: {df_5m.index[0]} ~ {df_5m.index[-1]}")
    print(f"  データ数: {len(df_5m):,}行")

    # 1時間足データ
    print(f"\n1時間足データ: {config.DATA_1H}")
    df_1h = pd.read_csv(config.DATA_1H)
    df_1h['Local time'] = pd.to_datetime(df_1h['Local time'], dayfirst=True)
    df_1h.set_index('Local time', inplace=True)
    print(f"  期間: {df_1h.index[0]} ~ {df_1h.index[-1]}")
    print(f"  データ数: {len(df_1h):,}行")

    # 4時間足データ
    print(f"\n4時間足データ: {config.DATA_4H}")
    df_4h = pd.read_csv(config.DATA_4H)
    df_4h['Local time'] = pd.to_datetime(df_4h['Local time'], dayfirst=True)
    df_4h.set_index('Local time', inplace=True)
    print(f"  期間: {df_4h.index[0]} ~ {df_4h.index[-1]}")
    print(f"  データ数: {len(df_4h):,}行")

    print("\nデータ読み込み完了")
    return df_5m, df_1h, df_4h


def print_statistics(stats: dict):
    """統計情報を表示"""
    print("\n" + "=" * 60)
    print("バックテスト結果")
    print("=" * 60)

    print(f"\n【トレード概要】")
    print(f"  総トレード数:     {stats['total_trades']:,}件")
    print(f"  勝ちトレード:     {stats['wins']:,}件")
    print(f"  負けトレード:     {stats['losses']:,}件")
    print(f"  勝率:             {stats['win_rate']:.2f}%")

    print(f"\n【損益】")
    print(f"  総利益:           {stats['total_profit_pips']:,.2f} pips")
    print(f"  平均利益:         {stats['avg_profit_pips']:.2f} pips")
    print(f"  総利益額:         {stats['gross_profit']:,.2f} pips")
    print(f"  総損失額:         {stats['gross_loss']:,.2f} pips")
    print(f"  プロフィットファクター: {stats['profit_factor']:.2f}")

    print(f"\n【平均値】")
    print(f"  平均勝ちトレード: {stats['avg_win']:.2f} pips")
    print(f"  平均負けトレード: {stats['avg_loss']:.2f} pips")

    print(f"\n【決済内訳】")
    print(f"  TP決済:           {stats['tp_exits']:,}件")
    print(f"  SL決済:           {stats['sl_exits']:,}件")

    print("\n" + "=" * 60)


def save_results(df_trades: pd.DataFrame, stats: dict):
    """結果を保存"""
    # 出力ディレクトリ作成
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # トレード詳細をCSV保存
    trades_file = config.TRADES_CSV
    df_trades.to_csv(trades_file, index=False, encoding='utf-8-sig')
    print(f"\nトレード詳細を保存: {trades_file}")

    # レポートをテキスト保存
    report_file = config.REPORT_FILE
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("System Ver6 バックテストレポート\n")
        f.write("=" * 60 + "\n")
        f.write(f"\n実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        f.write(f"\n【設定】\n")
        f.write(f"  通貨ペア:         {config.SYMBOL}\n")
        f.write(f"  バックテスト期間: {config.BACKTEST_START} ~ {config.BACKTEST_END}\n")
        f.write(f"  取引時間帯:       {config.TRADING_START_HOUR}:00 ~ {config.TRADING_END_HOUR}:00\n")
        f.write(f"  リスクリワード:   1:{config.RISK_REWARD_RATIO}\n")

        f.write(f"\n【インジケーター設定】\n")
        f.write(f"  RCI短期:          {config.RCI_SHORT}\n")
        f.write(f"  RCI中期:          {config.RCI_MID}\n")
        f.write(f"  RCI長期:          {config.RCI_LONG}\n")
        f.write(f"  EMA短期:          {config.EMA_SHORT}\n")
        f.write(f"  EMA中期:          {config.EMA_MID}\n")
        f.write(f"  EMA長期:          {config.EMA_LONG}\n")
        f.write(f"  ZigZag Depth:     {config.ZIGZAG_DEPTH}\n")

        f.write(f"\n【トレード概要】\n")
        f.write(f"  総トレード数:     {stats['total_trades']:,}件\n")
        f.write(f"  勝ちトレード:     {stats['wins']:,}件\n")
        f.write(f"  負けトレード:     {stats['losses']:,}件\n")
        f.write(f"  勝率:             {stats['win_rate']:.2f}%\n")

        f.write(f"\n【損益】\n")
        f.write(f"  総利益:           {stats['total_profit_pips']:,.2f} pips\n")
        f.write(f"  平均利益:         {stats['avg_profit_pips']:.2f} pips\n")
        f.write(f"  総利益額:         {stats['gross_profit']:,.2f} pips\n")
        f.write(f"  総損失額:         {stats['gross_loss']:,.2f} pips\n")
        f.write(f"  プロフィットファクター: {stats['profit_factor']:.2f}\n")

        f.write(f"\n【平均値】\n")
        f.write(f"  平均勝ちトレード: {stats['avg_win']:.2f} pips\n")
        f.write(f"  平均負けトレード: {stats['avg_loss']:.2f} pips\n")

        f.write(f"\n【決済内訳】\n")
        f.write(f"  TP決済:           {stats['tp_exits']:,}件\n")
        f.write(f"  SL決済:           {stats['sl_exits']:,}件\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"レポートを保存: {report_file}")


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("System Ver6 - バックテスト")
    print("=" * 60)

    # データ読み込み
    df_5m, df_1h, df_4h = load_data()

    # バックテスト実行
    print("\n" + "=" * 60)
    engine = BacktestEngineV6(df_5m, df_1h, df_4h)
    df_trades = engine.run_backtest()

    # 統計情報取得
    stats = engine.get_statistics()

    # 結果表示
    print_statistics(stats)

    # 結果保存
    save_results(df_trades, stats)

    print("\n処理完了")


if __name__ == '__main__':
    main()
