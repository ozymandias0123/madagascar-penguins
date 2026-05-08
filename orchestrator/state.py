"""
orchestrator/state.py — Shared state that flows through every LangGraph node.

All agents read from and write to this single TypedDict.
The DataFrame (df) is carried in-memory; LangGraph does NOT checkpoint it.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class TradingState(TypedDict):
    # ── Raw market data ───────────────────────────────────────
    symbol:        str
    df:            Any          # pd.DataFrame — passed by reference, not serialised
    session:       str          # 'london' | 'new_york'
    current_price: float
    atr:           float
    structure:     str          # e.g. 'bullish_bos'
    fvg_zones:     List[Dict]
    ob:            Dict         # order block
    context:       Dict         # {'regime', 'volatility', 'adx', 'atr_ratio'}
    htf_bias:      str          # 'bullish' | 'bearish' | 'neutral'
    htf_structure: Dict

    # ── Rule-based signal (pre-agent) ─────────────────────────
    signal:        Optional[Dict]   # {'type', 'entry_price', 'zone', 'quality', ...}
    confidence:    float            # ML model confidence 0–1
    quality_score: float            # ICT quality score

    # ── Derived trade levels ──────────────────────────────────
    entry_price:   float
    sl_price:      float
    tp_price:      float
    sl_distance:   float
    tp_distance:   float
    zone_type:     str      # 'premium' | 'discount' | 'equilibrium'
    liquidity_swept: bool
    is_silver_bullet: bool
    hour:          int
    balance:       float

    # ── Agent outputs (populated by each node) ────────────────
    market_analysis:  Optional[Dict]   # Skipper     (GPT-4o)
    risk_assessment:  Optional[Dict]   # Kowalski    (Claude) — includes news
    validation:       Optional[Dict]   # Private     (DeepSeek-chat)

    # ── Final orchestrator decision ───────────────────────────
    approved:          bool
    final_action:      str             # 'buy' | 'sell' | 'skip'
    rejection_reasons: List[str]
    execution_result:  Optional[Dict]  # filled by TradeEngine
    rico_notification: Optional[Dict]  # filled by Rico (Telegram)

    # ── Metadata ──────────────────────────────────────────────
    timestamp: str
    iteration: int
