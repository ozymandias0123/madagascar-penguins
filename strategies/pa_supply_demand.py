"""
strategies/pa_supply_demand.py
Price Action — Supply & Demand Zones

Logic
-----
  Supply and Demand zones are the origin of strong impulsive moves.
  A fresh zone has not been retested since its creation.

  Demand Zone (for buys):
    - Find the last "base" before a strong bullish impulse:
      Base = 2-4 consecutive small-body candles (consolidation)
      Impulse = candle that breaks above the base with body > 1.5×ATR
    - Zone: base low → base high
    - Entry: when price retraces back into the zone (fresh)
    - Confirmation: bullish pin bar or engulfing inside zone

  Supply Zone (for sells):
    - Find the last base before a strong bearish impulse
    - Zone: base low → base high
    - Entry: retest of zone from below

  Freshness: a zone is only valid if price hasn't closed INSIDE the zone
  since the impulse candle.

  SL : 0.5×ATR below zone low (demand) / above zone high (supply)
  TP : 2× height of the impulse candle (minimum 2R)
  Quality: 6 base, +1 fresh zone, +1 confirmation candle,
           +1 htf_bias aligned, +1 session, +1 ADX>20
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from strategies.base_strategy import BaseStrategy


class PASupplyDemand(BaseStrategy):

    name        = "PASupplyDemand"
    description = "Supply & Demand zone retest with impulse + consolidation base"
    version     = "1.0"

    SCAN_BARS   = 50
    BASE_MAX    = 5      # max candles in the consolidation base
    IMPULSE_ATR = 1.5    # impulse body must exceed this × ATR
    SL_BUF      = 0.5
    RR_MIN      = 2.0

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

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        entry   = float(close[i])
        atr_val = float(atr_s[i])
        signals = []

        # ── Demand zone ───────────────────────────────────────────────────────
        dz = self._find_demand_zone(open_, close, high, low, atr_s, i)
        if dz is not None:
            zone_low, zone_high, impulse_size, zone_bar = dz
            if zone_low <= entry <= zone_high:
                # Freshness: no close inside zone between zone_bar and now
                if self._is_fresh(close, zone_low, zone_high, zone_bar + 1, i - 1):
                    conf = self._bull_confirmation(open_[i], close[i], high[i], low[i])
                    sl   = zone_low - self.SL_BUF * atr_val
                    risk = abs(entry - sl)
                    if risk > 1e-10:
                        tp   = entry + max(self.RR_MIN * risk, impulse_size)
                        qual = self._quality(conf, htf_bias, "buy", context, session)
                        signals.append({
                            "type":        "buy",
                            "entry_price": round(entry, 5),
                            "sl_price":    round(sl, 5),
                            "tp_price":    round(tp, 5),
                            "quality":     qual,
                            "zone":        {"high": round(zone_high, 5),
                                            "low":  round(zone_low, 5)},
                            "pattern_key": "pa_demand_zone_buy",
                            "strategy":    self.name,
                            "notes":       (f"Demand zone [{round(zone_low,5)}-{round(zone_high,5)}] | "
                                            f"fresh | impulse={round(impulse_size,5)}"),
                        })

        # ── Supply zone ───────────────────────────────────────────────────────
        sz = self._find_supply_zone(open_, close, high, low, atr_s, i)
        if sz is not None:
            zone_low2, zone_high2, impulse_size2, zone_bar2 = sz
            if zone_low2 <= entry <= zone_high2:
                if self._is_fresh(close, zone_low2, zone_high2, zone_bar2 + 1, i - 1):
                    conf2 = self._bear_confirmation(open_[i], close[i], high[i], low[i])
                    sl2   = zone_high2 + self.SL_BUF * atr_val
                    risk2 = abs(sl2 - entry)
                    if risk2 > 1e-10:
                        tp2  = entry - max(self.RR_MIN * risk2, impulse_size2)
                        qual2 = self._quality(conf2, htf_bias, "sell", context, session)
                        signals.append({
                            "type":        "sell",
                            "entry_price": round(entry, 5),
                            "sl_price":    round(sl2, 5),
                            "tp_price":    round(tp2, 5),
                            "quality":     qual2,
                            "zone":        {"high": round(zone_high2, 5),
                                            "low":  round(zone_low2, 5)},
                            "pattern_key": "pa_supply_zone_sell",
                            "strategy":    self.name,
                            "notes":       (f"Supply zone [{round(zone_low2,5)}-{round(zone_high2,5)}] | "
                                            f"fresh | impulse={round(impulse_size2,5)}"),
                        })

        return signals

    # ── Zone detection ────────────────────────────────────────────────────────

    def _find_demand_zone(self, open_, close, high, low, atr_s, i
                          ) -> Optional[Tuple[float, float, float, int]]:
        """Find most recent demand zone: base + bullish impulse."""
        start = max(1, i - self.SCAN_BARS)
        for j in range(i - 2, start, -1):
            atr_val = float(atr_s[j]) if not np.isnan(atr_s[j]) else 1e-10
            # Impulse: strong bullish candle
            imp_body = float(close[j]) - float(open_[j])
            if imp_body < self.IMPULSE_ATR * atr_val:
                continue
            # Base: preceding 2-5 small-body candles
            base_start = max(start, j - self.BASE_MAX)
            base_highs, base_lows = [], []
            for b in range(base_start, j):
                body = abs(float(close[b]) - float(open_[b]))
                rng  = float(high[b]) - float(low[b])
                if rng > 0 and body / rng < 0.6:   # small body
                    base_highs.append(float(high[b]))
                    base_lows.append(float(low[b]))
            if len(base_lows) < 2:
                continue
            zone_low  = min(base_lows)
            zone_high = max(base_highs)
            if zone_high <= zone_low:
                continue
            return (zone_low, zone_high, imp_body, j)
        return None

    def _find_supply_zone(self, open_, close, high, low, atr_s, i
                          ) -> Optional[Tuple[float, float, float, int]]:
        """Find most recent supply zone: base + bearish impulse."""
        start = max(1, i - self.SCAN_BARS)
        for j in range(i - 2, start, -1):
            atr_val = float(atr_s[j]) if not np.isnan(atr_s[j]) else 1e-10
            imp_body = float(open_[j]) - float(close[j])
            if imp_body < self.IMPULSE_ATR * atr_val:
                continue
            base_start = max(start, j - self.BASE_MAX)
            base_highs, base_lows = [], []
            for b in range(base_start, j):
                body = abs(float(close[b]) - float(open_[b]))
                rng  = float(high[b]) - float(low[b])
                if rng > 0 and body / rng < 0.6:
                    base_highs.append(float(high[b]))
                    base_lows.append(float(low[b]))
            if len(base_lows) < 2:
                continue
            zone_low  = min(base_lows)
            zone_high = max(base_highs)
            if zone_high <= zone_low:
                continue
            return (zone_low, zone_high, imp_body, j)
        return None

    @staticmethod
    def _is_fresh(close, zl, zh, start, end) -> bool:
        """True if no close was inside the zone between start and end."""
        for k in range(start, end + 1):
            if zl <= float(close[k]) <= zh:
                return False
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bull_confirmation(o, c, h, l) -> bool:
        body = c - o; rng = h - l
        return body > 0 and rng > 0 and body / rng > 0.4

    @staticmethod
    def _bear_confirmation(o, c, h, l) -> bool:
        body = o - c; rng = h - l
        return body > 0 and rng > 0 and body / rng > 0.4

    @staticmethod
    def _quality(conf, htf_bias, sig_type, context, session) -> float:
        score = 6.0
        score += 1.0   # freshness verified
        if conf: score += 1.0
        bias_ok = (sig_type == "buy"  and htf_bias == "bullish") or \
                  (sig_type == "sell" and htf_bias == "bearish")
        if bias_ok:                           score += 1.0
        if session in ("london", "new_york"): score += 0.5
        if context.get("adx", 0) > 20:       score += 0.5
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
