"""
strategies/ha_rapid_scalper.py
1-Min Rapid Scalper  (v1.0 — faithful Pine port)

Logic
-----
  Heikin Ashi candles are computed from OHLCV:
    HA_close = (O + H + L + C) / 4
    HA_open  = (prev_HA_open + prev_HA_close) / 2
    HA_high  = max(H, HA_open, HA_close)
    HA_low   = min(L, HA_open, HA_close)

  3-candle sequencer (at bars i-3..i):
    colorFlip: bar i-3 must be opposite colour (bearish for longs, bullish for shorts)
    bull1:  HA_close[i-2] > HA_open[i-2]  AND  HA_open[i-2] == HA_low[i-2]  (no lower wick)
    bull2:  HA_close[i-1] > HA_open[i-1]  AND  no lower wick  AND body > body[i-2]
    bull3:  HA_close[i]   > HA_open[i]    AND  no lower wick  AND body > body[i-1]
    (bearish sequence mirrors with upper wick = HA_open == HA_high)

  ADX filter: ADX(7) > 20

  Exits (ATR-based, useAtr=True mode):
    Long SL  = entry − ATR(14)[prev] × slMult (2.0)
    Long TP  = entry + ATR(14)[prev] × tpMult (1.0)
    Hard SL cap: entry × (1 − thePlug/100)  [thePlug = 1.0 %]

  Note: original default is 1-bar exit (useAtr=False). This port defaults to
  ATR exit (USE_ATR=True) as that gives well-defined SL/TP for signal storage.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class HARapidScalper(BaseStrategy):

    name        = "HARapidScalper"
    description = "HA 3-candle accelerating sequence + ADX filter, ATR SL/TP"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    ADX_LEN      = 7
    ADX_THRESH   = 20.0
    ATR_LEN      = 14
    TP_MULT      = 1.0      # tpMult
    SL_MULT      = 2.0      # slMult
    HARD_SL_PCT  = 1.0      # thePlug — hard stop as % of entry price

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        # Need at least 4 bars for color-flip check + 3-bar sequence
        needed = max(self.ATR_LEN + 5, self.ADX_LEN + 20)
        if len(df) < needed:
            return []

        close  = df["close"].values
        open_  = df["open"].values
        high   = df["high"].values
        low    = df["low"].values
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        # ── Heikin Ashi ───────────────────────────────────────────────────────
        ha_open, ha_high, ha_low, ha_close = self._ha_candles(open_, high, low, close)

        i = len(df) - 1
        if i < 3:
            return []

        if np.isnan(atr_s[i - 1]):
            return []

        # ── ADX filter ────────────────────────────────────────────────────────
        adx_val = context.get("adx") or self._adx(high, low, close, i)
        if adx_val < self.ADX_THRESH:
            return []

        # ── HA helpers ────────────────────────────────────────────────────────
        def is_bull(k):   return ha_close[k] > ha_open[k]
        def is_bear(k):   return ha_close[k] < ha_open[k]
        def no_lo_wick(k): return abs(ha_open[k] - ha_low[k]) < 1e-8
        def no_hi_wick(k): return abs(ha_open[k] - ha_high[k]) < 1e-8
        def body(k):       return abs(ha_close[k] - ha_open[k])

        # ── Long sequencer ────────────────────────────────────────────────────
        color_flip_bull = is_bear(i - 3)
        bull1 = is_bull(i - 2) and no_lo_wick(i - 2)
        bull2 = is_bull(i - 1) and no_lo_wick(i - 1) and body(i-1) > body(i-2)
        bull3 = is_bull(i)     and no_lo_wick(i)     and body(i)   > body(i-1)
        long_trigger = color_flip_bull and bull1 and bull2 and bull3

        # ── Short sequencer ───────────────────────────────────────────────────
        color_flip_bear = is_bull(i - 3)
        bear1 = is_bear(i - 2) and no_hi_wick(i - 2)
        bear2 = is_bear(i - 1) and no_hi_wick(i - 1) and body(i-1) > body(i-2)
        bear3 = is_bear(i)     and no_hi_wick(i)     and body(i)   > body(i-1)
        short_trigger = color_flip_bear and bear1 and bear2 and bear3

        if not long_trigger and not short_trigger:
            return []

        sig_type = "buy" if long_trigger else "sell"
        entry    = float(close[i])
        atr_val  = float(atr_s[i - 1])   # Pine uses atrValue[1] at entry bar

        # ── SL / TP (ATR mode) ────────────────────────────────────────────────
        if sig_type == "buy":
            atr_sl  = entry - atr_val * self.SL_MULT
            plug_sl = entry * (1.0 - self.HARD_SL_PCT / 100.0)
            sl      = max(atr_sl, plug_sl)   # The Plug: whichever is higher = tighter
            tp      = entry + atr_val * self.TP_MULT
        else:
            atr_sl  = entry + atr_val * self.SL_MULT
            plug_sl = entry * (1.0 + self.HARD_SL_PCT / 100.0)
            sl      = min(atr_sl, plug_sl)
            tp      = entry - atr_val * self.TP_MULT

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        quality = self._quality(sig_type, adx_val, context)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(float(ha_high[i]), 5),
                            "low":  round(float(ha_low[i]), 5)},
            "pattern_key": f"ha_scalper_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"HA 3-seq {'bull' if long_trigger else 'bear'} | "
                            f"ADX={adx_val:.1f} | "
                            f"body_sizes={body(i-2):.5f}→{body(i-1):.5f}→{body(i):.5f}"),
        }]

    # ── Heikin Ashi candles ───────────────────────────────────────────────────

    @staticmethod
    def _ha_candles(open_: np.ndarray, high: np.ndarray,
                    low: np.ndarray, close: np.ndarray):
        n        = len(close)
        ha_close = (open_ + high + low + close) / 4.0
        ha_open  = np.full(n, np.nan)
        ha_open[0] = (open_[0] + close[0]) / 2.0
        for k in range(1, n):
            ha_open[k] = (ha_open[k - 1] + ha_close[k - 1]) / 2.0
        ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
        ha_low  = np.minimum(low,  np.minimum(ha_open, ha_close))
        return ha_open, ha_high, ha_low, ha_close

    # ── ADX ───────────────────────────────────────────────────────────────────

    def _adx(self, high, low, close, i: int) -> float:
        n = self.ADX_LEN
        if i < n + 2:
            return 0.0
        sl  = slice(max(0, i - n - 5), i + 1)
        h   = high[sl];  l = low[sl];  c = close[sl]
        c1  = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        dm_p = np.where((h - np.roll(h, 1)) > (np.roll(l, 1) - l),
                         np.maximum(h - np.roll(h, 1), 0.0), 0.0)
        dm_m = np.where((np.roll(l, 1) - l) > (h - np.roll(h, 1)),
                         np.maximum(np.roll(l, 1) - l, 0.0), 0.0)
        atr14  = self._rma(tr,    n)[-1]
        di_p   = 100 * self._rma(dm_p, n)[-1] / (atr14 + 1e-10)
        di_m   = 100 * self._rma(dm_m, n)[-1] / (atr14 + 1e-10)
        return float(100 * abs(di_p - di_m) / (di_p + di_m + 1e-10))

    @staticmethod
    def _rma(arr: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        if len(arr) < n:
            return out
        out[n - 1] = arr[:n].mean()
        alpha = 1.0 / n
        for k in range(n, len(arr)):
            out[k] = alpha * arr[k] + (1 - alpha) * out[k - 1]
        return out

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values;  l = df["low"].values;  c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, adx: float, context: dict) -> float:
        score = 5.5
        if adx > 30:
            score += 1.5
        elif adx > 20:
            score += 0.5
        if context.get("volatility", "normal") == "high":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
