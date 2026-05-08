"""
orchestrator/graph.py — LangGraph StateGraph (Penguin Squad)

Flow:
  START
    │
    ▼
  skipper_node       (Skipper  — GPT-4o)       ICT market analysis
    │
    ▼
  kowalski_node      (Kowalski — Claude)        Risk + live news check
    │
    ▼
  rico_node          (Rico     — DeepSeek)      Final validation
    │
    ├── approved=True  ──► execution_node  (TradeEngine — MT5 execution)
    │                           │
    └── approved=False ──► skip_node       (log reasons)
                                │
                      (both paths merge)
                                │
                                ▼
                      private_notify_node   (Private — Telegram notification)
                                │
                                ▼
                              END
"""

import logging
from typing import Dict, Literal

from langgraph.graph import END, START, StateGraph

from orchestrator.state import TradingState
from agents.skipper      import Skipper
from agents.kowalski     import Kowalski
from agents.rico         import Rico          # Final Validator (DeepSeek)
from agents.private      import Private       # Telegram Bot & Notifier
from agents.trade_engine import TradeEngine


# ── Singleton agent instances ─────────────────────────────────────────────────

_skipper      = Skipper()
_kowalski     = Kowalski()
_rico         = Rico()           # Final Validator (DeepSeek)
_private      = Private()        # Telegram Notifier
_trade_engine = TradeEngine()


# ══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def skipper_node(state: TradingState) -> Dict:
    """Skipper — ICT Market Analyst (GPT-4o). Populates: market_analysis."""
    logging.info("[GRAPH] 🎖  Node: skipper_node")
    return _skipper.run(state)


def kowalski_node(state: TradingState) -> Dict:
    """
    Kowalski — Risk Manager + Live News (Claude Sonnet).
    Populates: risk_assessment.
    """
    logging.info("[GRAPH] 🧠 Node: kowalski_node")
    return _kowalski.run(state)


def rico_node(state: TradingState) -> Dict:
    """Rico — Final Validator (DeepSeek). Populates: validation, approved."""
    logging.info("[GRAPH] 🃏 Node: rico_node")
    return _rico.run(state)


def execution_node(state: TradingState) -> Dict:
    """TradeEngine — rule-based MT5 execution. Populates: execution_result."""
    logging.info("[GRAPH] ⚙️  Node: execution_node")
    return _trade_engine.run(state)


def skip_node(state: TradingState) -> Dict:
    """Terminal rejection node — logs reasons, passes to Private."""
    reasons = state.get("rejection_reasons", [])
    logging.info(f"[GRAPH] ⏭️  Node: skip_node | {reasons}")
    return {
        "execution_result": {
            "executed": False,
            "reason":   "rejected_by_orchestrator",
            "action":   "skip",
            "details":  reasons,
        }
    }


def private_notify_node(state: TradingState) -> Dict:
    """
    Private — Telegram Notifier.
    Always called last. Sends trade alert or skip notification.
    """
    logging.info("[GRAPH] 💬 Node: private_notify_node")
    return _private.run(state)


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE
# ══════════════════════════════════════════════════════════════════════════════

def route_after_rico(
    state: TradingState,
) -> Literal["execution_node", "skip_node"]:
    """
    After Rico's validation:
      → execution_node  if approved + valid action + Kowalski approved
      → skip_node       otherwise
    """
    approved     = state.get("approved", False)
    final_action = state.get("final_action", "skip")
    rm_approved  = (state.get("risk_assessment") or {}).get("approved", False)

    if approved and final_action in ("buy", "sell") and rm_approved:
        logging.info(f"[GRAPH] → execution_node (action={final_action})")
        return "execution_node"

    logging.info(
        f"[GRAPH] → skip_node "
        f"(approved={approved}, action={final_action}, kowalski={rm_approved})"
    )
    return "skip_node"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(engine=None):
    """
    Compile and return the LangGraph StateGraph.

    Parameters
    ----------
    engine : PersistentTradingEngine | None
        Live engine for trade execution and Private's stats queries.
    """
    if engine is not None:
        _trade_engine.set_engine(engine)
        _private.set_engine(engine)

    # Start Private's Telegram command bot in background
    _private.start_bot()

    graph = StateGraph(TradingState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    graph.add_node("skipper_node",          skipper_node)
    graph.add_node("kowalski_node",         kowalski_node)
    graph.add_node("rico_node",             rico_node)
    graph.add_node("execution_node",        execution_node)
    graph.add_node("skip_node",             skip_node)
    graph.add_node("private_notify_node",   private_notify_node)

    # ── Linear edges ───────────────────────────────────────────────────────
    graph.add_edge(START,            "skipper_node")
    graph.add_edge("skipper_node",   "kowalski_node")
    graph.add_edge("kowalski_node",  "rico_node")

    # ── Conditional: Rico → Execute | Skip ────────────────────────────────
    graph.add_conditional_edges(
        "rico_node",
        route_after_rico,
        {
            "execution_node": "execution_node",
            "skip_node":      "skip_node",
        }
    )

    # ── Both paths → Private notify → END ─────────────────────────────────
    graph.add_edge("execution_node",        "private_notify_node")
    graph.add_edge("skip_node",             "private_notify_node")
    graph.add_edge("private_notify_node",   END)

    compiled = graph.compile()
    logging.info(
        "[GRAPH] Penguin Squad compiled — "
        "Skipper → Kowalski → Rico(validate) → [Execute|Skip] → Private(notify)"
    )
    return compiled


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def print_graph_structure() -> None:
    print()
    print("  .---.     .---.     .---.     .---.")
    print(" ( o o )   ( - - )   ( ~ ~ )   ( ^ ^ )")
    print("  \\ = /     \\ w /     \\ o /     \\ _ /")
    print(" SKIPPER  KOWALSKI    RICO     PRIVATE")
    print(" GPT-4o    Claude   DeepSeek  Telegram")
    print()
    print("  SKIPPER → KOWALSKI(+news) → RICO(validate) → [EXECUTE|SKIP] → PRIVATE → TG")
    print()


def get_private() -> Private:
    """Return the shared Private instance (for engine to call notify_* directly)."""
    return _private
