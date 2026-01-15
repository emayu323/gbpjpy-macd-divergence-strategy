"""
共通インジケーターモジュール
各バージョンのバックテストから参照して使用

使用例:
    from indicators import calculate_rci, calculate_ema, calculate_macd
    from indicators import calculate_zigzag_with_prospective
    from indicators import check_perfect_order, check_5m_entry_long
"""

# RCI
from .rci import calculate_rci

# EMA
from .ema import calculate_ema

# MACD
from .macd import calculate_macd

# ATR
from .atr import calculate_atr

# ZigZag
from .zigzag import (
    calculate_zigzag,
    calculate_zigzag_with_prospective,
    get_latest_zigzag_level
)

# トレード条件
from .conditions import (
    check_perfect_order,
    check_1h_rci_condition,
    check_5m_entry_long,
    check_5m_entry_short
)

__all__ = [
    # RCI
    'calculate_rci',
    # EMA
    'calculate_ema',
    # MACD
    'calculate_macd',
    # ATR
    'calculate_atr',
    # ZigZag
    'calculate_zigzag',
    'calculate_zigzag_with_prospective',
    'get_latest_zigzag_level',
    # 条件判定
    'check_perfect_order',
    'check_1h_rci_condition',
    'check_5m_entry_long',
    'check_5m_entry_short',
]
