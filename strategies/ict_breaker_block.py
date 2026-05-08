"""
strategies/ict_breaker_block.py
ICT Breaker Block

Logic
-----
  A Breaker Block forms when:
    1. An order block (OB) is VIOLATED — price sweeps through it.
    2. The violated OB then becomes a "breaker" — the opposite side
       of that block acts as new support/resistance.

  Bullish Breaker:
    - A bearish OB (bullish candle before a down-move) existed.
    - Price drove DOWN through the OB low (sweeping sell-side liquidity).
    - Price then REVERSES and trades BACK UP through the OB high.
    - The broken OB high now becomes a support level.
    - Entry: when price retraces back to the top of the original OB (now support).

  Bearish Breaker:
    - A bullish OB (bearish candle before an up-move) existed.
    - Price drove UP through the OB high (sweeping buy-side liquidity).
    - Price then REVERSES and trades BACK DOWN through the OB low.
    - Entry: retest of the bottom of the original OB (now resistance).

  Scan window: last 40 bars for the OB formation.

  SL : 1×ATR beyond the breaker level
  TP : 2×risk
  Quality: 5 base, +1 clean retest (price touched zone), +1 htf_bias,
           +1 ADX>25, +1 session kill-zone, +1 volume spike
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from strategies.base_strategy import BaseStrategy


class ICTBreakerBlock(BaseStrategy):

    name        = "ICTBreakerBlock"
    description = "ICT Breaker Block: violated OB becomes S/R for retest entry"
    version     = "1.0"

    SCAN_BARS  = 40
    SL_ATR_MUL = 1.0
    RR         = 2.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.SCAN_BARS + 5:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        open_  = df["open"].values
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)
        vol_s  = df["volume"].values if "volume" in df.columns else None

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        entry   = float(close[i])
        atr_val = float(atr_s[i])
        signals = []

        scan_start = max(0, i - self.SCAN_BARS)

        # ── Bullish breaker scan ──────────────────────────────────────────────
        breaker = self._find_bullish_breaker(
            open_, close, high, low, scan_start, i)
        if breaker is not None:
            ob_high, ob_low, break_idx = breaker
            # Price retesting the breaker top (support)
            retest_zone_high = ob_high
            retest_zone_low  = ob_high - 0.5 * atr_val
            if retest_zone_low <= entry <= retest_zone_high * 1.001:
                vol_ok = self._vol_spike(vol_s, i)
                sl   = ob_low - self.SL_ATR_MUL * atr_val
                risk = abs(entry - sl)
                if risk > 1e-10:
                    tp   = entry + self.RR * risk
                    qual = self._quality(vol_ok, session, htf_bias, "buy", context)
                    signals.append({
                        "type":        "buy",
                        "entry_price": round(entry, 5),
                        "sl_price":    round(sl, 5),
                        "tp_price":    round(tp, 5),
                        "quality":     qual,
                        "zone":        {"high": round(retest_zone_high, 5),
                                        "low":  round(retest_zone_low, 5)},
                        "pattern_key": "ict_breaker_buy",
                        "strategy":    self.name,
                        "notes":       (f"Bull breaker top={round(ob_high,5)} | "
                                        f"original OB [{round(ob_low,5)}-{round(ob_high,5)}]"),
                    })

        # ── Bearish breaker scan ──────────────────────────────────────────────
        breaker2 = self._find_bearish_breaker(
            open_, close, high, low, scan_start, i)
        if breaker2 is not None:
            ob_high2, ob_low2, _ = breaker2
            retest_zone_low2  = ob_low2
            retest_zone_high2 = ob_low2 + 0.5 * atr_val
            if retest_zone_low2 * 0.999 <= entry <= retest_zone_high2:
                vol_ok2 = self._vol_spike(vol_s, i)
                sl2   = ob_high2 + self.SL_ATR_MUL * atr_val
                risk2 = abs(sl2 - entry)
                if risk2 > 1e-10:
                    tp2  = entry - self.RR * risk2
                    qual2 = self._quality(vol_ok2, session, htf_bias, "sell", context)
                    signals.append({
                        "type":        "sell",
                        "entry_price": round(entry, 5),
                        "sl_price":    round(sl2, 5),
                        "tp_price":    round(tp2, 5),
                        "quality":     qual2,
                        "zone":        {"high": round(retest_zone_high2, 5),
                                        "low":  round(retest_zone_low2, 5)},
                        "pattern_key": "ict_breaker_sell",
                        "strategy":    self.name,
                        "notes":       (f"Bear breaker bot={round(ob_low2,5)} | "
                                        f"original OB [{round(ob_low2,5)}-{round(ob_high2,5)}]"),
                    })

        return signals

    # ── Pattern detection ─────────────────────────────────────────────────────

    @staticmethod
    def _find_bullish_breaker(
        open_, close, high, low, start, i
    ) -> Optional[Tuple[float, float, int]]:
        """
        Scan for a bullish breaker:
          - Find a bearish candle (OB candidate)
          - Price sweeps BELOW its low (at some later bar)
          - Price then closes ABOVE its high (breaker confirmed)
          - Then price comes back (current bar) to retest the OB high
        """
        for j in range(start, i - 3):
            if float(close[j]) >= float(open_[j]):
                continue  # need bearish OB candle
            ob_high = float(high[j])
            ob_low  = float(low[j])
            # Check sweep below OB low in subsequent bars
            swept = False
            recovered = False
            for k in range(j + 1, i):
                if float(low[k]) < ob_low:
                    swept = True
                if swept and float(close[k]) > ob_high:
                    recovered = True
                    return (ob_high, ob_low, k)
        return None

    @staticmethod
    def _find_bearish_breaker(
        open_, close, high, low, start, i
    ) -> Optional[Tuple[float, float, int]]:
        """Scan for a bearish breaker: bullish OB swept high then recovered below."""
        for j in range(start, i - 3):
            if float(close[j]) <= float(open_[j]):
                continue  # need bullish OB candle
            ob_high = float(high[j])
            ob_low  = float(low[j])
            swept = False
            for k in range(j + 1, i):
                if float(high[k]) > ob_high:
                    swept = True
                if swept and float(close[k]) < ob_low:
                    return (ob_high, ob_low, k)
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _vol_spike(vol, i) -> bool:
        if vol is None or i < 20:
            return False
        avg = float(np.nanmean(vol[i - 20: i]))
        return avg > 0 and float(vol[i]) > avg * 1.3

    @staticmethod
    def _quality(vol_ok, session, htf_bias, sig_type, context) -> float:
        score = 5.0
        score += 1.0  # retest zone already confirmed
        bias_ok = (sig_type == "buy"  and htf_bias == "bullish") or \
                  (sig_type == "sell" and htf_bias == "bearish")
        if bias_ok:                          score += 1.0
        if context.get("adx", 0) > 25:      score += 1.0
        if session in ("london", "new_york"): score += 1.0
        if vol_ok:                           score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values; l = df["low"].values; c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr
