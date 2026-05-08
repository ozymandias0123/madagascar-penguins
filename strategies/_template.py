"""
strategies/_template.py  <-- copy this file to add a new strategy

INSTRUCTIONS:
  1. Copy this file:    cp _template.py my_strategy.py
  2. Rename the class and set `name`
  3. Implement generate_signals()
  4. That's it — it auto-loads on next run

The leading underscore means this file is NOT auto-loaded.
Remove it from your copy: my_strategy.py (no underscore).
"""

import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class TemplateStrategy(BaseStrategy):

    name        = "Template"
    description = "Copy me and replace the logic"
    version     = "1.0"

    def generate_signals(
        self,
        df:       pd.DataFrame,
        context:  Dict[str, Any],
        session:  str,
        htf_bias: str,
    ) -> List[Dict]:

        signals = []

        # ---- your logic here ----------------------------------------
        # df columns available:
        #   open, high, low, close, volume
        #   rsi, ema_fast, ema_slow, atr, adx, bb_upper, bb_lower, ...
        #
        # context keys:
        #   adx, atr_ratio, volatility ('low'|'normal'|'high'), regime
        #
        # session  : 'london' | 'new_york' | 'asian' | 'off'
        # htf_bias : 'bullish' | 'bearish' | 'neutral'

        last  = df.iloc[-2]   # last closed candle
        prev  = df.iloc[-3]   # one before that

        entry = float(last["close"])
        atr   = float(last.get("atr", 10))

        # example: simple RSI cross (replace with your logic)
        rsi_now  = float(last.get("rsi", 50))
        rsi_prev = float(prev.get("rsi", 50))

        if rsi_prev < 30 < rsi_now and htf_bias != "bearish":
            sl = entry - 1.5 * atr
            tp = entry + 3.0 * atr
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     5.0,
                "zone":        {"high": entry + atr * 0.5, "low": entry - atr * 0.5},
                "pattern_key": "rsi_oversold_cross",
                "strategy":    self.name,
                "notes":       f"RSI crossed 30 from below ({rsi_prev:.1f} -> {rsi_now:.1f})",
            })

        return signals
