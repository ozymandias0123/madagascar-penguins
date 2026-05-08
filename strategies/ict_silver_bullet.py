"""
strategies/ict_silver_bullet.py
ICT Silver Bullet

Logic
-----
  The ICT Silver Bullet is a time-based, liquidity-driven entry model.

  Kill zones (UTC):
    Silver Bullet 1: 03:00 – 04:00  (London open displacement)
    Silver Bullet 2: 10:00 – 11:00  (NY AM session)
    Silver Bullet 3: 14:00 – 15:00  (NY lunch reversal)

  Steps:
    1. Previous-session liquidity sweep: session high/low of the prior 20 bars
       is taken (price temporarily exceeds it then closes back inside).
    2. Market structure shift (MSS): after the sweep, a candle closes in the
       opposite direction — confirming displacement.
    3. Fair Value Gap (FVG): 3-bar pattern immediately after the MSS candle.
       Bullish FVG: bar[-3].low > bar[-1].high (gap up — buy-side imbalance)
       Bearish FVG: bar[-3].high < bar[-1].low (gap down — sell-side imbalance)
    4. Price retraces INTO the FVG zone on the current candle.

  Entry  : mid of the FVG zone (conservative)
  SL     : beyond the swing that was swept (+ 0.5×ATR buffer)
  TP     : 2× risk
  Quality: 6 base, +1 FVG size > 0.5×ATR, +1 kill-zone, +1 htf_bias match
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class ICTSilverBullet(BaseStrategy):

    name        = "ICTSilverBullet"
    description = "Time-based ICT entry: liquidity sweep + MSS + FVG retracement"
    version     = "1.0"

    SWING_LOOK = 20
    SL_BUF_ATR = 0.5
    RR         = 2.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 30:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        # Kill-zone filter
        if session not in ("london", "new_york"):
            return []

        atr_val    = float(atr_s[i])
        entry_c    = float(close[i])
        look       = min(self.SWING_LOOK, i - 3)
        swing_high = float(np.max(high[i - look: i - 1]))
        swing_low  = float(np.min(low[i - look: i - 1]))

        signals = []

        # ── Bullish setup ─────────────────────────────────────────────────────
        # 1. Liquidity sweep below swing low (wick below, closed above)
        if float(low[i - 1]) < swing_low and float(close[i - 1]) > swing_low:
            # 2. MSS: current candle is bullish (closes above previous close)
            if entry_c > float(close[i - 1]):
                # 3. FVG check in the 3 bars centred on the MSS candle
                if i >= 3 and float(low[i - 1]) > float(high[i - 3]):
                    fvg_low  = float(high[i - 3])
                    fvg_high = float(low[i - 1])
                    fvg_mid  = (fvg_low + fvg_high) / 2.0
                    # 4. Price currently in or below the FVG (retracement)
                    if entry_c <= fvg_high:
                        entry = fvg_mid
                        sl    = swing_low - self.SL_BUF_ATR * atr_val
                        risk  = abs(entry - sl)
                        if risk > 1e-10:
                            tp   = entry + self.RR * risk
                            qual = self._quality("buy", htf_bias, context,
                                                 fvg_high - fvg_low, atr_val, session)
                            signals.append({
                                "type":        "buy",
                                "entry_price": round(entry, 5),
                                "sl_price":    round(sl, 5),
                                "tp_price":    round(tp, 5),
                                "quality":     qual,
                                "zone":        {"high": round(fvg_high, 5),
                                                "low":  round(fvg_low, 5)},
                                "pattern_key": "ict_silver_bullet_buy",
                                "strategy":    self.name,
                                "notes":       f"Sweep low | MSS bull | FVG [{round(fvg_low,5)}-{round(fvg_high,5)}]",
                            })

        # ── Bearish setup ─────────────────────────────────────────────────────
        if float(high[i - 1]) > swing_high and float(close[i - 1]) < swing_high:
            if entry_c < float(close[i - 1]):
                if i >= 3 and float(high[i - 1]) < float(low[i - 3]):
                    fvg_high2 = float(low[i - 3])
                    fvg_low2  = float(high[i - 1])
                    fvg_mid2  = (fvg_low2 + fvg_high2) / 2.0
                    if entry_c >= fvg_low2:
                        entry = fvg_mid2
                        sl    = swing_high + self.SL_BUF_ATR * atr_val
                        risk  = abs(sl - entry)
                        if risk > 1e-10:
                            tp   = entry - self.RR * risk
                            qual = self._quality("sell", htf_bias, context,
                                                 fvg_high2 - fvg_low2, atr_val, session)
                            signals.append({
                                "type":        "sell",
                                "entry_price": round(entry, 5),
                                "sl_price":    round(sl, 5),
                                "tp_price":    round(tp, 5),
                                "quality":     qual,
                                "zone":        {"high": round(fvg_high2, 5),
                                                "low":  round(fvg_low2, 5)},
                                "pattern_key": "ict_silver_bullet_sell",
                                "strategy":    self.name,
                                "notes":       f"Sweep high | MSS bear | FVG [{round(fvg_low2,5)}-{round(fvg_high2,5)}]",
                            })

        return signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type, htf_bias, context, fvg_size, atr_val, session) -> float:
        score = 6.0
        if fvg_size > 0.5 * atr_val: score += 1.0
        if session == "london":       score += 1.0
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
