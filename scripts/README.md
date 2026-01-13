# Scripts フォルダ

このフォルダには、GBPJPY取引戦略のバックテスト・分析用スクリプトが整理されています。

## 📁 フォルダ構成

### `ver7/` - System Ver7（最新版・推奨）
**MACD Divergence Strategy**

- `config_v7.py` - 設定ファイル
- `indicators_v7.py` - インジケーター計算（RCI, EMA, MACD, ダイバージェンス検出）
- `backtest_engine_v7.py` - バックテストエンジン
- `run_backtest_v7.py` - 実行スクリプト

**パフォーマンス**:
- 勝率: 57.28%
- 総利益: +2,737 pips
- プロフィットファクター: 1.39

**使い方**:
```bash
cd /Users/emayusuke/Desktop/トレード戦略_claude
source venv/bin/activate
python scripts/ver7/run_backtest_v7.py
```

---

### `ver6/` - System Ver6
**RCI Multi-Timeframe Strategy**

- `config_v6.py` - 設定ファイル
- `indicators_v6.py` - インジケーター計算（RCI, EMA, ZigZag）
- `backtest_engine_v6.py` - バックテストエンジン
- `run_backtest_v6.py` - 実行スクリプト

**パフォーマンス**:
- 勝率: 34.92%
- 総利益: -4,917 pips
- プロフィットファクター: 0.75

**注意**: Ver6は損失を出す結果となったため、使用非推奨。

---

### `legacy/` - 旧バージョン・実験的スクリプト

初期バージョンのスクリプト群。参考用。

- `backtest_analysis.py` - 最初のバックテスト分析スクリプト
- `backtest_analysis_lite.py` - 軽量版バックテスト
- `backtest_improved.py` - 改善版バックテスト
- `analyze_trades.py` - トレード分析スクリプト
- `config.py` - 初期設定ファイル
- `indicators.py` - 初期インジケーター
- `backtest_engine.py` - 初期バックテストエンジン
- `report_generator.py` - レポート生成

---

### `pinescript/` - TradingView用スクリプト

- `GBPJPY_System_Ver7.pine` - System Ver7のPineScript実装
- `sukepoyo_sub.pine` - その他のインジケーター

**使い方**:
1. TradingViewのPineエディタにコピー
2. 5分足チャート、GBPJPYで使用

---

## 🚀 推奨の使用方法

### 1. 最新版（Ver7）でバックテスト実行

```bash
cd /Users/emayusuke/Desktop/トレード戦略_claude
source venv/bin/activate
python scripts/ver7/run_backtest_v7.py
```

結果は `results_v7/` フォルダに出力されます。

### 2. TradingViewでリアルタイム監視

`scripts/pinescript/GBPJPY_System_Ver7.pine` をTradingViewで使用。

---

## 📊 バックテスト結果の確認

各バージョンの結果は以下のフォルダに保存されています：

- `results_v6/` - Ver6の結果
- `results_v7/` - Ver7の結果（推奨）

---

## 🔧 開発履歴

1. **初期バージョン** (legacy) - RCI + ZigZagベースの戦略
2. **Ver6** (2024) - マルチタイムフレーム + RCIセットアップ → 損失
3. **Ver7** (2024) - **MACDダイバージェンス戦略** → 勝率57%、収益化成功 ✅

---

## 📝 注意事項

- Ver7が現在の最良バージョンです
- 実際の取引前に必ずデモ口座でテストしてください
- パラメータ調整は慎重に行ってください
