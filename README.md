# GBPJPY トレード戦略 - System Ver7

MACD Divergence Strategyを使用したGBPJPY取引システム

---

## 📊 パフォーマンス概要（Ver7）

**検証期間**: 2015年11月 ～ 2025年11月（約10年間）

| 指標 | 数値 |
|------|------|
| **勝率** | **59.39%** ✅ |
| **総利益** | **+4,135 pips** |
| **プロフィットファクター** | **1.45** |
| **総トレード数** | 490件 |
| **月平均トレード** | 4.1回 |

---

## 🎯 戦略の特徴

### System Ver7 - MACD Divergence Strategy

**基本理念**: 構造的な反転予兆（ダイバージェンス）を確認してからの精密エントリー

#### 環境認識
1. **4時間足**: EMA(20,30,40)でトレンド判定
2. **1時間足**: MACDダイバージェンス検出
   - ヒドゥン・ダイバージェンス（トレンド継続）
   - レギュラー・ダイバージェンス（反転）

#### エントリー条件
- **5分足**: RCI短期(9)とRCI中期(14)の反転シグナル
- **有効期間**: ダイバージェンス検出後12時間以内

#### リスク管理
- **SL**: 5分足ZigZag(Depth 5)の直近高安値
- **TP**: リスクリワード 1:1.5

詳細は [ENTRY_LOGIC.md](./ENTRY_LOGIC.md) を参照

---

## 📁 プロジェクト構成

```
トレード戦略_claude/
├── ENTRY_LOGIC.md                 # 戦略ロジックの詳細説明
├── PineScript_使用方法.md          # TradingView用の使い方ガイド
├── README.md                      # このファイル
├── requirements.txt               # Python依存パッケージ
│
├── ローソク足データ/               # 価格データ（CSV）
│   ├── GBPJPY_5M_2015-2025.csv   # 5分足
│   ├── GBPJPY_1H_2013-2025.csv   # 1時間足
│   └── GBPJPY_4H_2013-2025.csv   # 4時間足
│
├── scripts/                       # スクリプト集
│   ├── ver7/                      # System Ver7（最新版・推奨）
│   │   ├── config_v7.py
│   │   ├── indicators_v7.py
│   │   ├── backtest_engine_v7.py
│   │   └── run_backtest_v7.py
│   ├── ver6/                      # System Ver6（非推奨）
│   ├── legacy/                    # 旧バージョン
│   └── pinescript/                # TradingView用
│       └── GBPJPY_System_Ver7.pine
│
├── results_v7/                    # Ver7のバックテスト結果
│   ├── backtest_report_v7.txt    # レポート
│   ├── trades_v7.csv             # トレード詳細
│   └── divergences_v7.csv        # ダイバージェンス記録
│
└── venv/                          # Python仮想環境
```

---

## 🚀 クイックスタート

### 1. バックテスト実行

```bash
# 仮想環境を有効化
source venv/bin/activate

# Ver7バックテスト実行
python scripts/ver7/run_backtest_v7.py
```

結果は `results_v7/` フォルダに出力されます。

### 2. TradingViewで使用

1. `scripts/pinescript/GBPJPY_System_Ver7.pine` をコピー
2. TradingViewのPineエディタに貼り付け
3. 5分足チャート、GBPJPYで実行

詳細は [PineScript_使用方法.md](./PineScript_使用方法.md) を参照

---

## 📈 システムバージョン比較

| バージョン | 戦略 | 勝率 | 総利益 | PF | 評価 |
|-----------|------|------|--------|-----|------|
| **Ver7** | **MACDダイバージェンス** | **57.28%** | **+2,737 pips** | **1.39** | ✅ **推奨** |
| Ver6 | RCIマルチタイムフレーム | 34.92% | -4,917 pips | 0.75 | ❌ 損失 |
| Legacy | 初期バージョン | - | - | - | 🔧 実験的 |

---

## 🛠️ 環境構築

### 必要な環境
- Python 3.8以上
- pandas, numpy

### セットアップ

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# または仮想環境を使用
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📊 詳細なバックテスト結果

### Ver7の主な特徴

#### ダイバージェンスタイプ別
- **ヒドゥン型**: 293件（71.1%） - トレンド継続シグナル
- **レギュラー型**: 119件（28.9%） - 反転シグナル

#### 決済内訳
- **TP到達**: 235件（57.0%）
- **SL到達**: 177件（43.0%）

#### 平均値
- **平均勝ちトレード**: 41.49 pips
- **平均負けトレード**: 40.08 pips

詳細は `results_v7/backtest_report_v7.txt` を参照

---

## 📝 ドキュメント

- [ENTRY_LOGIC.md](./ENTRY_LOGIC.md) - 戦略の詳細ロジック
- [PineScript_使用方法.md](./PineScript_使用方法.md) - TradingView用ガイド
- [scripts/README.md](./scripts/README.md) - スクリプト詳細

---

## ⚠️ 免責事項

このシステムは教育目的で開発されています。

- **実際の取引前に必ずデモ口座でテスト**してください
- 過去のパフォーマンスは将来の結果を保証しません
- 投資は自己責任で行ってください
- 損失のリスクを十分に理解した上で使用してください

---

## 📜 ライセンス

個人利用のみ

---

## 🔄 更新履歴

- **2025-01-13**: Ver7実装、PineScript作成、スクリプト整理
- **2025-01-12**: Ver6検証（損失判明）、Ver7開発開始
- **2024**: 初期バージョン開発

---

## 📧 問い合わせ

プロジェクト関連の質問は Issue で受け付けています。

---

**現在の推奨バージョン**: System Ver7 ✅
