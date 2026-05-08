"""
strategies/nq_super_scalper.py
NQ/MNQ Super Scalper (Simplified Python port)

Original PineScript uses footprint/order-flow data which is not available in
standard OHLCV feeds.  This port replaces those inputs with volume-ratio proxies
and retains all structurally computable signals:

  6 entry paths (tagged in pattern_key):
    MAIN  — ORB breakout (Opening Range Breakout) + retest
    BOS   — Break of Structure: close breaks N-bar high/low with volume surge
    OB    — Order Block: bearish/bullish engulfing with volume, price returns to block
    RE    — Reentry: price retests a recent BOS level from the correct side
    TRAP  — Failed breakout / stop-hunt (sweep above/below swing, then reversal)
    OD    — Opening Drive: directional move in first 15 min with follow-through

  Additional filters applied to all paths:
    - Hurst exponent (R/S) > 0.55  → trending (enabled)   / 0.45 → skip mean-rev
    - Squeeze Momentum (LazyBear): enter only when squeeze releases
    - SMT divergence proxy: price new high/low without RSI confirmation → fade

  SL: ATR-based (1.5× ATR)
  TP: 2.5× ATR  (R:R ≥ 1.6)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from strategies.base_strategy import BaseStrategy


class NQSuperScalper(BaseStrategy):

    name        = "NQSuperScalper"
    description = "6-path ICT scalper: ORB/BOS/OB/RE/TRAP/OD + Hurst + Squeeze"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    ORB_MINUTES   = 15      # Opening Range window (bars at 1-min equiv)
    BOS_LB        = 10      # swing high/low lookback for BOS
    OB_LB         = 5       # order-block lookback
    RE_LB         = 20      # reentry level lookback
    HURST_LB      = 50      # Hurst calculation period
    HURST_MIN     = 0.52    # minimum Hurst to allow trending entries
    SQ_BB_LEN     = 20      # Squeeze BB/KC length
    SQ_BB_MULT    = 2.0
    SQ_KC_MULT    = 1.5
    VOL_RATIO_THR = 1.5     # volume must be 1.5× average for BOS/OB
    ATR_SL_MULT   = 1.5
    ATR_TP_MULT   = 2.5
    RSI_LEN       = 14

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < max(self.HURST_LB, self.SQ_BB_LEN, self.RE_LB) + 20:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns \
                 else np.ones(len(df))
        atr_s  = df["atr"].values if "atr" in df.columns \
                 else self._calc_atr_arr(df)

        i       = len(df) - 1
        entry   = float(close[i])
        atr_val = float(atr_s[i]) if not np.isnan(atr_s[i]) else entry * 0.001

        # ── Pre-filters ───────────────────────────────────────────────────────
        hurst = self._hurst(close, self.HURST_LB)
        if hurst < self.HURST_MIN:
            return []   # mean-reverting — skip directional scalp

        sq_on, sq_val = self._squeeze(close, high, low, i,
                                       self.SQ_BB_LEN, self.SQ_BB_MULT,
                                       self.SQ_KC_MULT)
        # if squeeze is ACTIVE (compressed), don't trade — wait for release
        if sq_on:
            return []

        rsi = self._rsi(close, self.RSI_LEN)
        vol_avg = float(np.mean(volume[max(0, i - 20):i])) if i > 20 else 1.0

        # ── Try each entry path ───────────────────────────────────────────────
        sig = (self._path_main(close, high, low, volume, atr_s, i, vol_avg) or
               self._path_bos(close, high, low, volume, atr_s, i, vol_avg) or
               self._path_ob(close, high, low, volume, atr_s, i, vol_avg) or
               self._path_re(close, high, low, atr_s, i) or
               self._path_trap(close, high, low, atr_s, i) or
               self._path_od(close, high, low, volume, atr_s, i, vol_avg, df))

        if sig is None:
            return []

        sig_type, sl, path_name = sig
        tp = (entry + self.ATR_TP_MULT * atr_val if sig_type == "buy"
              else entry - self.ATR_TP_MULT * atr_val)

        # SMT divergence filter (fade signal if divergence detected)
        if self._smt_divergence(close, rsi, i, sig_type):
            return []

        quality = self._quality(sig_type, hurst, sq_val, context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(float(high[i]), 5),
                            "low":  round(float(low[i]), 5)},
            "pattern_key": f"nq_scalp_{path_name}_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"path={path_name} | Hurst={hurst:.3f} | "
                            f"sq_val={sq_val:.2f}"),
        }]

    # ── Entry Path: MAIN (ORB Breakout + Retest) ──────────────────────────────

    def _path_main(self, close, high, low, volume, atr, i, vol_avg):
        lb = self.ORB_MINUTES
        if i < lb + 2:
            return None
        orb_high = float(np.max(high[i - lb - 1: i - 1]))
        orb_low  = float(np.min(low[i - lb - 1: i - 1]))
        entry    = float(close[i])
        prev_c   = float(close[i - 1])
        atr_v    = float(atr[i])

        # breakout above ORB, prev bar closed outside, current retest from above
        if prev_c > orb_high and entry > orb_high * 0.9998:
            sl = entry - self.ATR_SL_MULT * atr_v
            return "buy", sl, "MAIN"
        if prev_c < orb_low and entry < orb_low * 1.0002:
            sl = entry + self.ATR_SL_MULT * atr_v
            return "sell", sl, "MAIN"
        return None

    # ── Entry Path: BOS (Break of Structure) ─────────────────────────────────

    def _path_bos(self, close, high, low, volume, atr, i, vol_avg):
        lb  = self.BOS_LB
        if i < lb + 2:
            return None
        swing_high = float(np.max(high[i - lb: i]))
        swing_low  = float(np.min(low[i - lb: i]))
        vol_now    = float(volume[i])
        entry      = float(close[i])
        atr_v      = float(atr[i])

        if entry > swing_high and vol_now > vol_avg * self.VOL_RATIO_THR:
            sl = entry - self.ATR_SL_MULT * atr_v
            return "buy", sl, "BOS"
        if entry < swing_low and vol_now > vol_avg * self.VOL_RATIO_THR:
            sl = entry + self.ATR_SL_MULT * atr_v
            return "sell", sl, "BOS"
        return None

    # ── Entry Path: OB (Order Block) ─────────────────────────────────────────

    def _path_ob(self, close, high, low, volume, atr, i, vol_avg):
        lb    = self.OB_LB
        if i < lb + 2:
            return None
        entry = float(close[i])
        atr_v = float(atr[i])

        # Bullish OB: last big bearish candle before bullish move that broke above it
        for k in range(i - lb, i):
            c, o = float(close[k]), float(close[k - 1])
            if c < o:   # bearish candle
                ob_low  = float(low[k])
                ob_high = float(high[k])
                # price returned to inside the OB zone after breaking above
                if ob_low <= entry <= ob_high:
                    if float(volume[k]) > vol_avg * self.VOL_RATIO_THR:
                        sl = ob_low - atr_v * self.ATR_SL_MULT
                        return "buy", sl, "OB"

        # Bearish OB: last big bullish candle before bearish move
        for k in range(i - lb, i):
            c, o = float(close[k]), float(close[k - 1])
            if c > o:   # bullish candle
                ob_low  = float(low[k])
                ob_high = float(high[k])
                if ob_low <= entry <= ob_high:
                    if float(volume[k]) > vol_avg * self.VOL_RATIO_THR:
                        sl = ob_high + atr_v * self.ATR_SL_MULT
                        return "sell", sl, "OB"
        return None

    # ── Entry Path: RE (Reentry after BOS) ───────────────────────────────────

    def _path_re(self, close, high, low, atr, i):
        lb    = self.RE_LB
        if i < lb + 2:
            return None
        entry  = float(close[i])
        atr_v  = float(atr[i])
        recent_high = float(np.max(high[i - lb: i - 1]))
        recent_low  = float(np.min(low[i - lb: i - 1]))

        tol = atr_v * 0.5
        # Price pulls back to recent broken level and holds
        if abs(entry - recent_high) < tol and float(close[i - 1]) > recent_high:
            sl = entry - atr_v * self.ATR_SL_MULT
            return "buy", sl, "RE"
        if abs(entry - recent_low) < tol and float(close[i - 1]) < recent_low:
            sl = entry + atr_v * self.ATR_SL_MULT
            return "sell", sl, "RE"
        return None

    # ── Entry Path: TRAP (Stop Hunt / Failed Breakout) ───────────────────────

    def _path_trap(self, close, high, low, atr, i):
        if i < 5:
            return None
        entry  = float(close[i])
        atr_v  = float(atr[i])
        # Bullish trap: prev bar made new low (swept lows), closed back above
        prev_h = float(high[i - 1])
        prev_l = float(low[i - 1])
        pp_l   = float(low[i - 2])
        pp_h   = float(high[i - 2])
        prev_c = float(close[i - 1])

        if prev_l < pp_l and prev_c > pp_l and entry > prev_c:
            sl = prev_l - atr_v * self.ATR_SL_MULT
            return "buy", sl, "TRAP"
        if prev_h > pp_h and prev_c < pp_h and entry < prev_c:
            sl = prev_h + atr_v * self.ATR_SL_MULT
            return "sell", sl, "TRAP"
        return None

    # ── Entry Path: OD (Opening Drive) ───────────────────────────────────────

    def _path_od(self, close, high, low, volume, atr, i, vol_avg, df):
        try:
            ts = df.index[i]
            from datetime import time as dtime
            t  = ts.time()
            if not (dtime(9, 30) <= t <= dtime(9, 45)):
                return None
        except Exception:
            return None   # no time info

        if i < 3:
            return None
        entry  = float(close[i])
        atr_v  = float(atr[i])
        prev_c = float(close[i - 1])
        vol_n  = float(volume[i])

        if entry > prev_c and vol_n > vol_avg * self.VOL_RATIO_THR:
            sl = entry - atr_v * self.ATR_SL_MULT
            return "buy", sl, "OD"
        if entry < prev_c and vol_n > vol_avg * self.VOL_RATIO_THR:
            sl = entry + atr_v * self.ATR_SL_MULT
            return "sell", sl, "OD"
        return None

    # ── Hurst Exponent (R/S method) ───────────────────────────────────────────

    @staticmethod
    def _hurst(close: np.ndarray, n: int) -> float:
        """Simplified R/S Hurst estimate over last n bars."""
        seg = close[-n:]
        if len(seg) < 8:
            return 0.5
        try:
            returns = np.diff(np.log(seg + 1e-10))
            mean_r  = returns.mean()
            devs    = np.cumsum(returns - mean_r)
            R       = devs.max() - devs.min()
            S       = returns.std()
            if S < 1e-10:
                return 0.5
            rs = R / S
            h  = np.log(rs) / np.log(n)
            return float(np.clip(h, 0.0, 1.0))
        except Exception:
            return 0.5

    # ── Squeeze Momentum (LazyBear) ───────────────────────────────────────────

    def _squeeze(self, close, high, low, i, bb_len, bb_mult, kc_mult):
        """Returns (squeeze_active: bool, momentum_value: float)."""
        if i < bb_len + 2:
            return False, 0.0
        sl = slice(i - bb_len + 1, i + 1)
        c  = close[sl]
        h  = high[sl]
        l  = low[sl]

        basis   = c.mean()
        std     = c.std()
        bb_up   = basis + bb_mult * std
        bb_dn   = basis - bb_mult * std

        # True Range for KC
        c1  = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr_kc = tr.mean()
        kc_up  = basis + kc_mult * atr_kc
        kc_dn  = basis - kc_mult * atr_kc

        squeeze_on = (bb_up < kc_up) and (bb_dn > kc_dn)

        # Momentum = close - midref (linreg proxy → just use last value diff)
        mid_ref = (np.max(h) + np.min(l)) / 2.0
        mom_val = float(close[i]) - (mid_ref + basis) / 2.0

        return squeeze_on, mom_val

    # ── SMT Divergence Proxy ──────────────────────────────────────────────────

    def _smt_divergence(self, close, rsi, i, sig_type: str) -> bool:
        """
        True = divergence detected (fade the signal).
        Simplified: price makes new high but RSI doesn't (bearish div → fade buy).
        """
        lb = min(14, i)
        if lb < 5 or np.isnan(rsi[i]):
            return False
        price_max = np.max(close[i - lb: i])
        rsi_max   = np.max(rsi[i - lb: i])
        price_min = np.min(close[i - lb: i])
        rsi_min   = np.min(rsi[i - lb: i])

        if sig_type == "buy":
            # bearish divergence: price near new low but RSI not at new low
            if close[i] <= price_min * 1.001 and rsi[i] > rsi_min + 5:
                return True
        else:
            # bullish divergence: price near new high but RSI not at new high
            if close[i] >= price_max * 0.999 and rsi[i] < rsi_max - 5:
                return True
        return False

    # ── RSI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: np.ndarray, n: int) -> np.ndarray:
        out    = np.full(len(close), np.nan)
        delta  = np.diff(close, prepend=close[0])
        gains  = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        if len(close) < n + 1:
            return out
        avg_g = gains[1:n + 1].mean()
        avg_l = losses[1:n + 1].mean()
        for k in range(n, len(close)):
            if k > n:
                avg_g = (avg_g * (n - 1) + gains[k]) / n
                avg_l = (avg_l * (n - 1) + losses[k]) / n
            rs = avg_g / avg_l if avg_l > 0 else 100.0
            out[k] = 100 - 100 / (1 + rs)
        return out

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values;  l  = df["low"].values;  c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    # ── Quality score ─────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, hurst: float, sq_val: float,
                 context: dict, htf_bias: str) -> float:
        score = 5.0
        # strong trending regime
        if hurst > 0.65:
            score += 1.5
        elif hurst > 0.55:
            score += 0.5
        # momentum in right direction
        if sig_type == "buy" and sq_val > 0:
            score += 1.0
        elif sig_type == "sell" and sq_val < 0:
            score += 1.0
        # HTF bias alignment
        if sig_type == "buy" and htf_bias == "bullish":
            score += 1.0
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 1.0
        if context.get("adx", 0) > 30:
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
