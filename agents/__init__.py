"""
agents/__init__.py — Penguin Squad

  Skipper      — ICT Market Analyst     (GPT-4o)
  Kowalski     — Risk + News Strategist (Claude Sonnet)
  Rico         — Final Validator        (DeepSeek-chat)
  Private      — Telegram Bot & Alerts  (rule-based + Telegram API)
  TradeEngine  — MT5 Execution          (rule-based)
"""

from agents.skipper      import Skipper
from agents.kowalski     import Kowalski
from agents.rico         import Rico
from agents.private      import Private
from agents.trade_engine import TradeEngine

__all__ = [
    "Skipper",
    "Kowalski",
    "Rico",
    "Private",
    "TradeEngine",
]
