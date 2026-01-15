"""
System Ver14 - MACD Divergence Strategy バックテスト実行スクリプト
Ver14: 1H PO + Hidden/Regular Div + 1H RCI短期±60 + 5M RCI短期 + 5M パーフェクトオーダー
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

import config_v14 as config
from backtest_engine_v14 import BacktestEngineV14


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
    print("System Ver14 - MACD Divergence Strategy")
    print("バックテスト結果")
    print("=" * 60)

    print(f"\n【サマリー統計】")
    print(f"  総トレード数:         {stats['total_trades']:,}件")
    print(f"  勝率:                 {stats['win_rate']:.2f}%")
    print(f"  期待値:               {stats['expectancy_pct']:.4f}%/トレード")
    print(f"  最大ドローダウン:     {stats['max_drawdown_pct']:.2f}%")
    print(f"  プロフィットファクター: {stats['profit_factor']:.2f}")

    print(f"\n【トレード内訳】")
    print(f"  勝ちトレード:         {stats['wins']:,}件")
    print(f"  負けトレード:         {stats['losses']:,}件")
    print(f"  TP決済:               {stats['tp_exits']:,}件")
    print(f"  SL決済:               {stats['sl_exits']:,}件")

    print(f"\n【損益（pips）】")
    print(f"  総利益:               {stats['total_profit_pips']:,.2f} pips")
    print(f"  平均利益:             {stats['avg_profit_pips']:.2f} pips")
    print(f"  平均勝ちトレード:     {stats['avg_win']:.2f} pips")
    print(f"  平均負けトレード:     {stats['avg_loss']:.2f} pips")

    print(f"\n【損益（金額: 初期資金¥{config.INITIAL_CAPITAL_JPY:,}、{config.LOT_SIZE}ロット）】")
    print(f"  総利益:               ¥{stats['total_profit_jpy']:,.0f}")
    print(f"  最大ドローダウン:     ¥{stats['max_drawdown_jpy']:,.0f}")

    print(f"\n【保有時間】")
    print(f"  平均保有時間:         {stats['avg_holding_hours']:.1f}時間")

    print(f"\n【ダイバージェンス】")
    print(f"  総検出数:             {stats['total_divergences']:,}件")
    print(f"  ヒドゥン型トレード:   {stats['hidden_divergence_trades']:,}件")
    print(f"  レギュラー型トレード: {stats['regular_divergence_trades']:,}件")

    print("\n" + "=" * 60)


def save_results(df_trades: pd.DataFrame, df_divergences: pd.DataFrame, stats: dict,
                 df_monthly: pd.DataFrame = None, df_yearly: pd.DataFrame = None):
    """結果を保存"""
    # 出力ディレクトリ作成
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # トレード詳細をCSV保存
    trades_file = config.TRADES_CSV
    df_trades.to_csv(trades_file, index=False, encoding='utf-8-sig')
    print(f"\nトレード詳細を保存: {trades_file}")

    # ダイバージェンス詳細をCSV保存
    if not df_divergences.empty:
        div_file = config.DIVERGENCES_CSV
        df_divergences.to_csv(div_file, index=False, encoding='utf-8-sig')
        print(f"ダイバージェンス詳細を保存: {div_file}")

    # 月別サマリーをCSV保存
    if df_monthly is not None and not df_monthly.empty:
        monthly_file = f"{config.RESULTS_DIR}/monthly_summary_v14.csv"
        df_monthly.to_csv(monthly_file, encoding='utf-8-sig')
        print(f"月別サマリーを保存: {monthly_file}")

    # 年別サマリーをCSV保存
    if df_yearly is not None and not df_yearly.empty:
        yearly_file = f"{config.RESULTS_DIR}/yearly_summary_v14.csv"
        df_yearly.to_csv(yearly_file, encoding='utf-8-sig')
        print(f"年別サマリーを保存: {yearly_file}")

    # レポートをテキスト保存
    report_file = config.REPORT_FILE
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("System Ver14 - MACD Divergence Strategy\n")
        f.write("バックテストレポート\n")
        f.write("=" * 60 + "\n")
        f.write(f"\n実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        f.write(f"\n【設定】\n")
        f.write(f"  通貨ペア:         {config.SYMBOL}\n")
        f.write(f"  バックテスト期間: {config.BACKTEST_START} ~ {config.BACKTEST_END}\n")
        f.write(f"  取引時間帯:       {config.TRADING_START_HOUR}:00 ~ {config.TRADING_END_HOUR}:00\n")
        f.write(f"  初期資金:         ¥{config.INITIAL_CAPITAL_JPY:,}\n")
        f.write(f"  ロットサイズ:     {config.LOT_SIZE}ロット ({config.POSITION_SIZE:,}通貨)\n")
        f.write(f"  リスクリワード:   1:{config.RISK_REWARD_RATIO}\n")
        f.write(f"  ダイバージェンス有効期間: {config.DIVERGENCE_VALID_HOURS}時間\n")

        f.write(f"\n【インジケーター設定】\n")
        f.write(f"  RCI短期:          {config.RCI_SHORT}\n")
        f.write(f"  RCI中期:          {config.RCI_MID}\n")
        f.write(f"  1H RCI閾値:       ±{config.RCI_1H_THRESHOLD}\n")
        f.write(f"  MACD:             ({config.MACD_FAST}, {config.MACD_SLOW}, {config.MACD_SIGNAL})\n")
        f.write(f"  4H EMA:           {config.EMA_4H}\n")
        f.write(f"  1H EMA (PO):      ({config.EMA_1H_SHORT}, {config.EMA_1H_MID}, {config.EMA_1H_LONG})\n")
        f.write(f"  5M EMA (PO):      ({config.EMA_5M_SHORT}, {config.EMA_5M_MID}, {config.EMA_5M_LONG})\n")
        f.write(f"  ZigZag 1H:        Depth {config.ZIGZAG_1H_DEPTH}\n")
        f.write(f"  ZigZag 5M:        Depth {config.ZIGZAG_5M_DEPTH}\n")

        f.write(f"\n【サマリー統計】\n")
        f.write(f"  総トレード数:         {stats['total_trades']:,}件\n")
        f.write(f"  勝率:                 {stats['win_rate']:.2f}%\n")
        f.write(f"  期待値:               {stats['expectancy_pct']:.4f}%/トレード\n")
        f.write(f"  最大ドローダウン:     {stats['max_drawdown_pct']:.2f}%\n")
        f.write(f"  プロフィットファクター: {stats['profit_factor']:.2f}\n")

        f.write(f"\n【トレード内訳】\n")
        f.write(f"  勝ちトレード:         {stats['wins']:,}件\n")
        f.write(f"  負けトレード:         {stats['losses']:,}件\n")
        f.write(f"  TP決済:               {stats['tp_exits']:,}件\n")
        f.write(f"  SL決済:               {stats['sl_exits']:,}件\n")

        f.write(f"\n【損益（pips）】\n")
        f.write(f"  総利益:               {stats['total_profit_pips']:,.2f} pips\n")
        f.write(f"  平均利益:             {stats['avg_profit_pips']:.2f} pips\n")
        f.write(f"  総利益pips:           {stats['gross_profit']:,.2f} pips\n")
        f.write(f"  総損失pips:           {stats['gross_loss']:,.2f} pips\n")
        f.write(f"  平均勝ちトレード:     {stats['avg_win']:.2f} pips\n")
        f.write(f"  平均負けトレード:     {stats['avg_loss']:.2f} pips\n")

        f.write(f"\n【損益（金額）】\n")
        f.write(f"  総利益:               ¥{stats['total_profit_jpy']:,.0f}\n")
        f.write(f"  最大ドローダウン:     ¥{stats['max_drawdown_jpy']:,.0f}\n")

        f.write(f"\n【保有時間】\n")
        f.write(f"  平均保有時間:         {stats['avg_holding_hours']:.1f}時間\n")

        f.write(f"\n【ダイバージェンス】\n")
        f.write(f"  総検出数:             {stats['total_divergences']:,}件\n")
        f.write(f"  ヒドゥン型トレード:   {stats['hidden_divergence_trades']:,}件\n")
        f.write(f"  レギュラー型トレード: {stats['regular_divergence_trades']:,}件\n")

        # 年別サマリー
        if df_yearly is not None and not df_yearly.empty:
            f.write(f"\n【年別損益サマリー】\n")
            f.write(df_yearly.to_string() + "\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"レポートを保存: {report_file}")


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("System Ver14 - MACD Divergence Strategy")
    print("バックテスト実行")
    print("=" * 60)

    # データ読み込み
    df_5m, df_1h, df_4h = load_data()

    # バックテスト実行
    print("\n" + "=" * 60)
    engine = BacktestEngineV14(df_5m, df_1h, df_4h)
    df_trades = engine.run_backtest()

    # ダイバージェンス情報取得
    df_divergences = engine.get_divergences_dataframe()

    # 統計情報取得
    stats = engine.get_statistics()

    # 月別・年別サマリー取得
    df_monthly = engine.get_monthly_summary()
    df_yearly = engine.get_yearly_summary()

    # 結果表示
    print_statistics(stats)

    # 年別サマリー表示
    if not df_yearly.empty:
        print("\n【年別損益サマリー】")
        print(df_yearly.to_string())
        print()

    # 結果保存
    save_results(df_trades, df_divergences, stats, df_monthly, df_yearly)

    print("\n処理完了")


if __name__ == '__main__':
    main()
