"""
System Ver7 - MACD Divergence Strategy 設定ファイル
ENTRY_LOGIC.md (Ver7) に基づくバックテスト設定
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

# RCI (Rank Correlation Index) - 5分足エントリートリガー用
RCI_SHORT = 9   # 短期
RCI_MID = 14    # 中期

# RCI閾値（Ver7で変更）
RCI_OVERBOUGHT = 60       # 買われすぎ（短期用）
RCI_OVERSOLD = -60        # 売られすぎ（短期用）
RCI_MID_OVERBOUGHT = 40   # 買われすぎ（中期用）Ver7で追加
RCI_MID_OVERSOLD = -40    # 売られすぎ（中期用）Ver7で追加

# EMA (Exponential Moving Average) - 4時間足トレンド判定用
EMA_SHORT = 20
EMA_MID = 30
EMA_LONG = 40

# MACD (Moving Average Convergence Divergence) - 1時間足ダイバージェンス検出用
MACD_FAST = 12      # 短期EMA
MACD_SLOW = 26      # 長期EMA
MACD_SIGNAL = 9     # シグナルライン

# ZigZag - ダイバージェンス検出用（1時間足）
ZIGZAG_1H_DEPTH = 12      # Ver7: Depth 12でダイバージェンス検出
ZIGZAG_1H_DEVIATION = 5
ZIGZAG_1H_BACKSTEP = 3

# ZigZag - 損切り基準（5分足）
ZIGZAG_5M_DEPTH = 5
ZIGZAG_5M_DEVIATION = 3
ZIGZAG_5M_BACKSTEP = 2

# ==========================================
# ダイバージェンス設定
# ==========================================
DIVERGENCE_VALID_HOURS = 12  # ダイバージェンス有効期間（時間）

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
RESULTS_DIR = "./results_v7"
REPORT_FILE = f"{RESULTS_DIR}/backtest_report_v7.txt"
TRADES_CSV = f"{RESULTS_DIR}/trades_v7.csv"
DIVERGENCES_CSV = f"{RESULTS_DIR}/divergences_v7.csv"  # ダイバージェンス記録

# ==========================================
# その他
# ==========================================
POINT_VALUE = 0.01  # GBPJPYの1ポイント（0.01円）
