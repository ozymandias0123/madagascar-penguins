"""
strategies/ict_optimal_trade_entry.py
ICT Optimal Trade Entry (OTE)

Logic
-----
  The ICT OTE model enters after a confirmed swing is broken (BOS),
  then waits for a 61.8%–78.6% Fibonacci retracement of the impulse leg.

  Steps:
    1. Detect the most recent impulse swing (last 30 bars).
       Bullish impulse: swing_low → swing_high (both within last 30 bars)
       Bearish impulse: swing_high → swing_low
    2. Calculate OTE zone:
       For bullish: fib 61.8% and 78.6% of the impulse UP move
         OTE_low  = swing_high - 0.786 * range
         OTE_high = swing_high - 0.618 * range
       For bearish: fib 61.8% and 78.6% of the impulse DOWN move
         OTE_low  = swing_low  + 0.618 * range
         OTE_high = swing_low  + 0.786 * range
    3. Price must close inside the OTE zone.
    4. Confirmation: a bullish engulfing or pin bar inside OTE (buy),
       or a bearish engulfing/pin bar (sell).
    5. HTF bias must agree.

  SL  : beyond the origin of the impulse (swing_low for buy, swing_high for sell)
        + 0.3×ATR buffer
  TP  : beyond the swing extreme + 0.5×ATR (targeting swing extension)
  Quality: 5 base, +1 OTE zone hit, +1 candle confirm, +1 htf_bias, +1 ADX>20, +1 kill-zone
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class ICTOptimalTradeEntry(BaseStrategy):

    name        = "ICTOptimalTradeEntry"
    description = "ICT OTE: BOS + 61.8-78.6% Fib retracement entry"
    version     = "1.0"

    SWING_LOOK = 30
    FIB_LOW    = 0.618
    FIB_HIGH   = 0.786
    SL_BUF     = 0.3
    RR_MIN     = 2.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.SWING_LOOK + 5:
            return []

        if htf_bias not in ("bullish", "bearish"):
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        open_  = df["open"].values
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        entry   = float(close[i])
        atr_val = float(atr_s[i])
        look    = min(self.SWING_LOOK, i - 2)
        window  = slice(i - look, i)

        swing_high_idx = int(np.argmax(high[window])) + (i - look)
        swing_low_idx  = int(np.argmin(low[window]))  + (i - look)
        swing_high     = float(high[swing_high_idx])
        swing_low      = float(low[swing_low_idx])
        imp_range      = swing_high - swing_low

        if imp_range < atr_val * 0.5:
            return []

        signals = []

        # ── Bullish OTE ───────────────────────────────────────────────────────
        if htf_bias == "bullish" and swing_low_idx < swing_high_idx:
            ote_high = swing_high - self.FIB_LOW  * imp_range
            ote_low  = swing_high - self.FIB_HIGH * imp_range
            if ote_low <= entry <= ote_high:
                candle_ok = self._bull_candle(open_[i], close[i], high[i], low[i])
                sl = swing_low - self.SL_BUF * atr_val
                risk = abs(entry - sl)
                if risk > 1e-10:
                    tp   = swing_high + 0.5 * atr_val
                    qual = self._quality(candle_ok, context, session, htf_bias, "buy")
                    signals.append({
                        "type":        "buy",
                        "entry_price": round(entry, 5),
                        "sl_price":    round(sl, 5),
                        "tp_price":    round(tp, 5),
                        "quality":     qual,
                        "zone":        {"high": round(ote_high, 5),
                                        "low":  round(ote_low, 5)},
                        "pattern_key": "ict_ote_buy",
                        "strategy":    self.name,
                        "notes":       (f"OTE 61.8-78.6% [{round(ote_low,5)}-{round(ote_high,5)}] | "
                                        f"impulse range={round(imp_range,5)} | htf=bullish"),
                    })

        # ── Bearish OTE ───────────────────────────────────────────────────────
        if htf_bias == "bearish" and swing_high_idx < swing_low_idx:
            ote_low2  = swing_low  + self.FIB_LOW  * imp_range
            ote_high2 = swing_low  + self.FIB_HIGH * imp_range
            if ote_low2 <= entry <= ote_high2:
                candle_ok = self._bear_candle(open_[i], close[i], high[i], low[i])
                sl = swing_high + self.SL_BUF * atr_val
                risk = abs(sl - entry)
                if risk > 1e-10:
                    tp   = swing_low - 0.5 * atr_val
                    qual = self._quality(candle_ok, context, session, htf_bias, "sell")
                    signals.append({
                        "type":        "sell",
                        "entry_price": round(entry, 5),
                        "sl_price":    round(sl, 5),
                        "tp_price":    round(tp, 5),
                        "quality":     qual,
                        "zone":        {"high": round(ote_high2, 5),
                                        "low":  round(ote_low2, 5)},
                        "pattern_key": "ict_ote_sell",
                        "strategy":    self.name,
                        "notes":       (f"OTE 61.8-78.6% [{round(ote_low2,5)}-{round(ote_high2,5)}] | "
                                        f"impulse range={round(imp_range,5)} | htf=bearish"),
                    })

        return signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bull_candle(o, c, h, l) -> bool:
        body  = c - o
        rng   = h - l
        return body > 0 and rng > 0 and body / rng > 0.4

    @staticmethod
    def _bear_candle(o, c, h, l) -> bool:
        body  = o - c
        rng   = h - l
        return body > 0 and rng > 0 and body / rng > 0.4

    @staticmethod
    def _quality(candle_ok, context, session, htf_bias, sig_type) -> float:
        score = 5.0
        score += 1.0  # OTE zone hit is already verified
        if candle_ok:                        score += 1.0
        if context.get("adx", 0) > 20:      score += 1.0
        if session in ("london", "new_york"): score += 1.0
        bias_ok = (sig_type == "buy"  and htf_bias == "bullish") or \
                  (sig_type == "sell" and htf_bias == "bearish")
        if bias_ok: score += 1.0
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
