"""
System Ver6 - 設定ファイル
ENTRY_LOGIC.mdに基づくバックテスト設定
"""

# ==========================================
# データファイルパス
# ==========================================
DATA_DIR = "./ローソク足データ"
DATA_5M = f"{DATA_DIR}/GBPJPY_5M_2015-2025.csv"
DATA_1H = f"{DATA_DIR}/GBPJPY_1H_2013-2025.csv"
DATA_4H = f"{DATA_DIR}/GBPJPY_4H_2013-2025.csv"

# ==========================================
# 取引設定
# ==========================================
SYMBOL = "GBPJPY"
INITIAL_CAPITAL = 1000000  # 初期資本（円）
POSITION_SIZE = 10000      # 1ポジションあたりの通貨量

# ==========================================
# 取引時間帯
# ==========================================
TRADING_START_HOUR = 8   # 日本時間 08:00
TRADING_END_HOUR = 24    # 日本時間 24:00 (0:00)

# ==========================================
# インジケーター設定
# ==========================================

# RCI (Rank Correlation Index)
RCI_SHORT = 9   # 短期（5分足エントリートリガー用）
RCI_MID = 14    # 中期（5分足エントリー確認、1時間足セットアップ用）
RCI_LONG = 18   # 長期（1時間足環境認識用）

# RCI閾値
RCI_OVERBOUGHT = 60    # 買われすぎ
RCI_OVERSOLD = -60     # 売られすぎ
RCI_ZERO_LINE = 0      # ゼロライン

# EMA (Exponential Moving Average) - 4時間足トレンド判定用
EMA_SHORT = 20
EMA_MID = 30
EMA_LONG = 40

# ZigZag - 損切り基準
ZIGZAG_DEPTH = 5
ZIGZAG_DEVIATION = 3
ZIGZAG_BACKSTEP = 2

# ==========================================
# リスク管理
# ==========================================
RISK_REWARD_RATIO = 1.5  # リスクリワードレシオ (1:1.5)

# Stop Loss設定
SL_BUFFER_PIPS = 2       # ZigZag高値/安値からのバッファ（pips）
SL_DEFAULT_PIPS = 20     # ZigZagが見つからない場合のデフォルトSL幅

# スプレッド
SPREAD_PIPS = 0.5        # 想定スプレッド（pips）

# ==========================================
# バックテスト期間
# ==========================================
# 5分足データの期間に合わせる（2015-2025）
BACKTEST_START = "2015-11-15"
BACKTEST_END = "2025-11-24"

# ==========================================
# 出力設定
# ==========================================
RESULTS_DIR = "./results_v6"
REPORT_FILE = f"{RESULTS_DIR}/backtest_report_v6.txt"
TRADES_CSV = f"{RESULTS_DIR}/trades_v6.csv"

# ==========================================
# その他
# ==========================================
POINT_VALUE = 0.01  # GBPJPYの1ポイント（0.01円）
