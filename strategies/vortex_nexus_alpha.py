"""
strategies/vortex_nexus_alpha.py
Vortex Nexus Alpha [JOAT]  (v1.0)

Logic
-----
  Five signal layers — each fires 0-or-1 contributions to bullStrength / bearStrength.
  Entry when strength >= minSignalStrength (2).

  Layer 1 — Laguerre momentum (gamma from fractal efficiency):
    price_efficiency = |close - close[N]| / sum(|Δclose|, N)
    gamma  = clamp(1 - price_efficiency, 0.1, 0.9)
    Laguerre: L0 = (1-g)*close + g*L0[-1]
               L1 = -g*L0 + L0[-1] + g*L1[-1]
               L2 = -g*L1 + L1[-1] + g*L2[-1]
               L3 = -g*L2 + L2[-1] + g*L3[-1]
    laguerre_val = (L0 + 2*L1 + 2*L2 + L3) / 6
    bull: laguerre_val > laguerre_val[-1]  (rising)
    bear: laguerre_val < laguerre_val[-1]

  Layer 2 — Fractal efficiency (momentum quality):
    fractal = price_efficiency > efficiency_threshold (0.3)
    bull/bear: same direction as close change

  Layer 3 — Temporal flow (fast/slow DEMA cross):
    DEMA(n) = 2*EMA(n) - EMA(EMA(n))
    bull: DEMA(fast=5) > DEMA(slow=13) AND crossed up in last 3 bars
    bear: DEMA(fast=5) < DEMA(slow=13) AND crossed down in last 3 bars

  Layer 4 — Volatility rank (normalized ATR):
    volRank = ATR(14) / highest(ATR, 50)
    bull: volRank > 0.3 AND close > close[-1]
    bear: volRank > 0.3 AND close < close[-1]

  Layer 5 — RSI momentum:
    bull: RSI(14) > 55 AND RSI > RSI[-1]
    bear: RSI(14) < 45 AND RSI < RSI[-1]

  SL = close ∓ volatility × slMultiplier  (volatility = ATR(14), slMultiplier=1.5)
  TP = close ± volatility × slMultiplier × tpMultiplier  (tpMultiplier=2.5)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class VortexNexusAlpha(BaseStrategy):

    name        = "VortexNexusAlpha"
    description = "Multi-layer signal scoring: Laguerre + fractal efficiency + DEMA + vol rank + RSI"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    MIN_SIGNAL_STRENGTH  = 2
    EFFICIENCY_WINDOW    = 10      # price efficiency lookback
    EFFICIENCY_THRESH    = 0.3
    GAMMA_MIN            = 0.1
    GAMMA_MAX            = 0.9
    DEMA_FAST            = 5
    DEMA_SLOW            = 13
    DEMA_CROSS_BARS      = 3
    ATR_LEN              = 14
    ATR_RANK_LEN         = 50
    ATR_RANK_THRESH      = 0.3
    RSI_LEN              = 14
    RSI_BULL             = 55.0
    RSI_BEAR             = 45.0
    SL_MULT              = 1.5
    TP_MULT              = 2.5

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = max(self.ATR_RANK_LEN + self.ATR_LEN, self.DEMA_SLOW * 4,
                     self.RSI_LEN + 5, self.EFFICIENCY_WINDOW + 5) + 10
        if len(df) < needed:
            return []

        close = df["close"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        atr_val = float(atr_s[i])

        # ── Layer 1: Laguerre momentum ────────────────────────────────────────
        laguerre_bull, laguerre_bear = self._laguerre_signal(close, i)

        # ── Layer 2: Fractal efficiency ───────────────────────────────────────
        eff_bull, eff_bear = self._fractal_efficiency_signal(close, i)

        # ── Layer 3: Temporal flow (DEMA cross) ───────────────────────────────
        dema_bull, dema_bear = self._dema_signal(close, i)

        # ── Layer 4: Volatility rank ──────────────────────────────────────────
        vol_bull, vol_bear = self._vol_rank_signal(atr_s, close, i)

        # ── Layer 5: RSI momentum ─────────────────────────────────────────────
        rsi_bull, rsi_bear = self._rsi_signal(close, i)

        bull_strength = sum([laguerre_bull, eff_bull, dema_bull, vol_bull, rsi_bull])
        bear_strength = sum([laguerre_bear, eff_bear, dema_bear, vol_bear, rsi_bear])

        long_ok  = bull_strength >= self.MIN_SIGNAL_STRENGTH and bull_strength > bear_strength
        short_ok = bear_strength >= self.MIN_SIGNAL_STRENGTH and bear_strength > bull_strength

        if not long_ok and not short_ok:
            return []

        sig_type = "buy" if long_ok else "sell"
        entry    = float(close[i])
        vol      = atr_val

        sl = (entry - vol * self.SL_MULT if sig_type == "buy"
              else entry + vol * self.SL_MULT)
        tp = (entry + vol * self.SL_MULT * self.TP_MULT if sig_type == "buy"
              else entry - vol * self.SL_MULT * self.TP_MULT)

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        quality = self._quality(sig_type, bull_strength if long_ok else bear_strength,
                                context, htf_bias)

        layers = (f"L={int(laguerre_bull if long_ok else laguerre_bear)}"
                  f" E={int(eff_bull if long_ok else eff_bear)}"
                  f" D={int(dema_bull if long_ok else dema_bear)}"
                  f" V={int(vol_bull if long_ok else vol_bear)}"
                  f" R={int(rsi_bull if long_ok else rsi_bear)}")

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(entry + vol, 5),
                            "low":  round(entry - vol, 5)},
            "pattern_key": f"vna_{sig_type}_str{int(bull_strength if long_ok else bear_strength)}",
            "strategy":    self.name,
            "notes":       (f"VNA {'bull' if long_ok else 'bear'} | "
                            f"strength={int(bull_strength if long_ok else bear_strength)}/5 | "
                            f"layers=[{layers}] | ATR={atr_val:.5f}"),
        }]

    # ── Laguerre ──────────────────────────────────────────────────────────────

    def _laguerre_signal(self, close: np.ndarray, i: int):
        n = self.EFFICIENCY_WINDOW
        if i < n + 2:
            return False, False

        # Price efficiency for gamma
        seg = close[i - n: i + 1]
        total_move = float(np.sum(np.abs(np.diff(seg))))
        net_move   = abs(float(seg[-1]) - float(seg[0]))
        eff = net_move / (total_move + 1e-10)
        gamma = float(np.clip(1.0 - eff, self.GAMMA_MIN, self.GAMMA_MAX))

        # Run Laguerre filter over a small window (need history)
        hist_len = min(50, i + 1)
        c_hist = close[i - hist_len + 1: i + 2]  # includes current
        L0 = L1 = L2 = L3 = float(c_hist[0])
        vals = []
        for c in c_hist:
            L0_new = (1 - gamma) * float(c) + gamma * L0
            L1_new = -gamma * L0_new + L0 + gamma * L1
            L2_new = -gamma * L1_new + L1 + gamma * L2
            L3_new = -gamma * L2_new + L2 + gamma * L3
            L0, L1, L2, L3 = L0_new, L1_new, L2_new, L3_new
            vals.append((L0 + 2*L1 + 2*L2 + L3) / 6.0)

        if len(vals) < 2:
            return False, False
        rising = vals[-1] > vals[-2]
        return rising, not rising

    # ── Fractal efficiency ────────────────────────────────────────────────────

    def _fractal_efficiency_signal(self, close: np.ndarray, i: int):
        n = self.EFFICIENCY_WINDOW
        if i < n:
            return False, False
        seg = close[i - n: i + 1]
        total_move = float(np.sum(np.abs(np.diff(seg))))
        net_move   = abs(float(seg[-1]) - float(seg[0]))
        eff = net_move / (total_move + 1e-10)
        if eff < self.EFFICIENCY_THRESH:
            return False, False
        up = float(close[i]) > float(close[i - 1])
        return up, not up

    # ── DEMA cross ────────────────────────────────────────────────────────────

    def _dema_signal(self, close: np.ndarray, i: int):
        if i < self.DEMA_SLOW * 4:
            return False, False
        dema_f = self._dema(close, self.DEMA_FAST)
        dema_s = self._dema(close, self.DEMA_SLOW)
        if np.isnan(dema_f[i]) or np.isnan(dema_s[i]):
            return False, False

        # Check if cross happened in last DEMA_CROSS_BARS
        bull = False; bear = False
        for lag in range(1, self.DEMA_CROSS_BARS + 1):
            k = i - lag + 1
            if k < 1 or np.isnan(dema_f[k - 1]) or np.isnan(dema_s[k - 1]):
                continue
            if dema_f[k] > dema_s[k] and dema_f[k - 1] <= dema_s[k - 1]:
                bull = True
            if dema_f[k] < dema_s[k] and dema_f[k - 1] >= dema_s[k - 1]:
                bear = True
        # Also require current alignment
        above = dema_f[i] > dema_s[i]
        return bull and above, bear and not above

    @staticmethod
    def _dema(arr: np.ndarray, n: int) -> np.ndarray:
        out  = np.full(len(arr), np.nan)
        mult = 2.0 / (n + 1)
        if len(arr) < n:
            return out
        ema1 = np.full(len(arr), np.nan)
        ema2 = np.full(len(arr), np.nan)
        ema1[n - 1] = arr[:n].mean()
        for k in range(n, len(arr)):
            ema1[k] = arr[k] * mult + ema1[k - 1] * (1 - mult)
        # EMA of EMA
        start2 = 2 * n - 2
        if start2 >= len(arr):
            return out
        ema2[start2] = ema1[n - 1: start2 + 1].mean()
        for k in range(start2 + 1, len(arr)):
            ema2[k] = ema1[k] * mult + ema2[k - 1] * (1 - mult)
        for k in range(start2, len(arr)):
            if not np.isnan(ema1[k]) and not np.isnan(ema2[k]):
                out[k] = 2 * ema1[k] - ema2[k]
        return out

    # ── Volatility rank ───────────────────────────────────────────────────────

    def _vol_rank_signal(self, atr_s: np.ndarray, close: np.ndarray, i: int):
        if i < self.ATR_RANK_LEN:
            return False, False
        seg = atr_s[i - self.ATR_RANK_LEN: i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) == 0:
            return False, False
        atr_high = float(np.max(valid))
        vol_rank = float(atr_s[i]) / (atr_high + 1e-10) if not np.isnan(atr_s[i]) else 0.0
        if vol_rank < self.ATR_RANK_THRESH:
            return False, False
        up = float(close[i]) > float(close[i - 1])
        return up, not up

    # ── RSI momentum ─────────────────────────────────────────────────────────

    def _rsi_signal(self, close: np.ndarray, i: int):
        n = self.RSI_LEN
        if i < n + 2:
            return False, False
        rsi_arr = self._rsi(close, n)
        if np.isnan(rsi_arr[i]) or np.isnan(rsi_arr[i - 1]):
            return False, False
        rv = float(rsi_arr[i]); rv_p = float(rsi_arr[i - 1])
        bull = rv > self.RSI_BULL and rv > rv_p
        bear = rv < self.RSI_BEAR and rv < rv_p
        return bull, bear

    @staticmethod
    def _rsi(close: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(close), np.nan)
        if len(close) < n + 1:
            return out
        delta = np.diff(close)
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = gain[:n].mean()
        avg_l = loss[:n].mean()
        out[n] = 100 - 100 / (1 + avg_g / (avg_l + 1e-10))
        for k in range(n + 1, len(close)):
            avg_g = (avg_g * (n - 1) + gain[k - 1]) / n
            avg_l = (avg_l * (n - 1) + loss[k - 1]) / n
            out[k] = 100 - 100 / (1 + avg_g / (avg_l + 1e-10))
        return out

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h = df["high"].values; l = df["low"].values; c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, strength: int, context: dict, htf_bias: str) -> float:
        score = 4.0 + strength * 0.8       # 2→5.6, 3→6.4, 4→7.2, 5→8.0
        if context.get("adx", 0) > 25:
            score += 0.5
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 0.5
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
