"""
strategies/sovereign_execution.py
Sovereign Execution [JOAT]  (v1.0)

Logic
-----
  Regime Cipher (adaptive MA on sqrt(close)):
    sqrt_price = sqrt(close)
    adaptive_ma = EMA(sqrt_price, regimeLen=20)
    regime_slope = adaptive_ma − adaptive_ma[-1]
    regime_trending_bull: regime_slope > 0 AND sqrt_price > adaptive_ma
    regime_trending_bear: regime_slope < 0 AND sqrt_price < adaptive_ma

  Regime strength bands (vol ratio):
    atr_ratio = ATR(14) / ATR(regimeLen=20)   (short/long vol ratio)
    strong_regime: atr_ratio > ratioThresh (1.2)

  Displacement Lens (composite score):
    BB %b  = (close − BB_lower) / (BB_upper − BB_lower)    (stdDev=2, bbLen=20)
             centered: bb_score = (bb_pct_b − 0.5) × 2    → range ≈ -1..+1
    CCI    = (close − SMA(close, cciLen=20)) / (0.015 × mean_deviation)
             cci_norm = CCI / 200                           → capped ±1
    ROC    = (close / close[-rocLen(10)] − 1) × 100
             roc_std  = stdev(ROC, rocStdLen=20)
             roc_norm = ROC / (roc_std × rocStdMult(2.0) + 1e-10)   → ≈ ±1

    raw_disp = bb_score × 0.4 + cci_norm × 0.35 + roc_norm × 0.25

    Volume-directional pressure:
      vol_avg = SMA(volume, 20)
      vol_ratio = volume / vol_avg
      bar_dir = sign(close − open)
      vol_press = vol_ratio × bar_dir × 0.2

    displacement = raw_disp + vol_press

    strong_bull_disp = displacement > dispThresh  (0.3)
    strong_bear_disp = displacement < −dispThresh

  Confluence score:
    CTF contribution (40%): regime_bull → +1, regime_bear → −1
    HTF contribution (60%): htf_bias == "bullish" → +1, == "bearish" → −1
    conf_score = ctf_contrib × 0.4 + htf_contrib × 0.6
    conf_ok_long  = conf_score > confThresh  (0.3)
    conf_ok_short = conf_score < −confThresh

  FVG filter (basic):
    fvg_ok_long  = low[i]  > high[i-2]  (bullish gap — acts as support)
                   OR disabled (fvgFilter=False → always True by default here)
    (disabled by default; always True to allow signals)

  Session filter: London/NY overlap (context["session"] or always allowed)

  Entry:
    long_entry  = regime_trending_bull AND conf_ok_long  AND strong_bull_disp AND session_ok
    short_entry = regime_trending_bear AND conf_ok_short AND strong_bear_disp AND session_ok

  SL:  close ∓ ATR(14) × atrSlMult  (1.5)
  TP:  close ± sl_dist × rrRatio     (2.0)

  Exit conditions (advisory in notes):
    Regime flip (slope reversal) — primary exit
    Displacement crosses zero    — secondary exit
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SovereignExecution(BaseStrategy):

    name        = "SovereignExecution"
    description = "Regime Cipher (sqrt EMA) + Displacement Lens confluence entry"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    REGIME_LEN      = 20
    ATR_LEN         = 14
    ATR_SL_MULT     = 1.5
    RR_RATIO        = 2.0
    RATIO_THRESH    = 1.2     # vol ratio for strong regime
    BB_LEN          = 20
    BB_STD          = 2.0
    CCI_LEN         = 20
    ROC_LEN         = 10
    ROC_STD_LEN     = 20
    ROC_STD_MULT    = 2.0
    DISP_THRESH     = 0.3
    CONF_THRESH     = 0.3
    FVG_FILTER      = False   # disabled by default

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = max(self.REGIME_LEN + 5, self.BB_LEN + 5,
                     self.CCI_LEN + 5, self.ROC_LEN + self.ROC_STD_LEN + 5,
                     self.ATR_LEN + 5)
        if len(df) < needed:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        open_  = df["open"].values if "open" in df.columns else close
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        atr_val = float(atr_s[i])

        # ── Regime Cipher ─────────────────────────────────────────────────────
        sqrt_price = np.sqrt(np.abs(close)) * np.sign(close)
        regime_ema = self._ema(sqrt_price, self.REGIME_LEN)
        if np.isnan(regime_ema[i]) or np.isnan(regime_ema[i - 1]):
            return []

        regime_slope = float(regime_ema[i]) - float(regime_ema[i - 1])
        sp_now       = float(sqrt_price[i])
        ema_now      = float(regime_ema[i])

        regime_bull = regime_slope > 0 and sp_now > ema_now
        regime_bear = regime_slope < 0 and sp_now < ema_now

        if not regime_bull and not regime_bear:
            return []

        # ── Vol ratio filter ──────────────────────────────────────────────────
        atr_long = self._ema_last_val(atr_s, self.REGIME_LEN, i)
        atr_ratio = atr_val / (atr_long + 1e-10) if not np.isnan(atr_long) else 1.0

        # ── Displacement Lens ─────────────────────────────────────────────────
        displacement = self._displacement(close, open_, volume, i)

        strong_bull_disp = displacement >  self.DISP_THRESH
        strong_bear_disp = displacement < -self.DISP_THRESH

        # ── Confluence score ──────────────────────────────────────────────────
        ctf_contrib = 1.0 if regime_bull else -1.0
        if htf_bias == "bullish":
            htf_contrib = 1.0
        elif htf_bias == "bearish":
            htf_contrib = -1.0
        else:
            htf_contrib = 0.0

        conf_score    = ctf_contrib * 0.4 + htf_contrib * 0.6
        conf_ok_long  = conf_score >  self.CONF_THRESH
        conf_ok_short = conf_score < -self.CONF_THRESH

        # ── FVG filter ────────────────────────────────────────────────────────
        if self.FVG_FILTER and i >= 2:
            fvg_ok_long  = float(low[i])  > float(high[i - 2])
            fvg_ok_short = float(high[i]) < float(low[i - 2])
        else:
            fvg_ok_long = fvg_ok_short = True

        # ── Session filter ────────────────────────────────────────────────────
        sess_ok = True
        if session not in ("", None):
            sess_ok = session in ("london", "new_york", "overlap", "any")

        # ── Entry ─────────────────────────────────────────────────────────────
        long_entry  = (regime_bull and conf_ok_long  and strong_bull_disp
                       and fvg_ok_long  and sess_ok)
        short_entry = (regime_bear and conf_ok_short and strong_bear_disp
                       and fvg_ok_short and sess_ok)

        if not long_entry and not short_entry:
            return []

        sig_type = "buy" if long_entry else "sell"
        entry    = float(close[i])

        sl = (entry - atr_val * self.ATR_SL_MULT if sig_type == "buy"
              else entry + atr_val * self.ATR_SL_MULT)

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp = (entry + risk * self.RR_RATIO if sig_type == "buy"
              else entry - risk * self.RR_RATIO)

        quality = self._quality(sig_type, conf_score, displacement,
                                atr_ratio, context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(ema_now + atr_val, 5),
                            "low":  round(ema_now - atr_val, 5)},
            "pattern_key": f"sov_exec_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"Regime={'bull' if regime_bull else 'bear'} "
                            f"slope={regime_slope:.5f} | "
                            f"disp={displacement:.3f} | "
                            f"conf={conf_score:.2f} | "
                            f"vol_ratio={atr_ratio:.2f} | "
                            f"exit=regime_flip|disp_zero_cross"),
        }]

    # ── Displacement Lens ─────────────────────────────────────────────────────

    def _displacement(self, close: np.ndarray, open_: np.ndarray,
                      volume: np.ndarray, i: int) -> float:
        n   = self.BB_LEN
        if i < n + self.ROC_LEN + self.ROC_STD_LEN:
            return 0.0

        # BB %b
        seg   = close[i - n + 1: i + 1]
        sma   = float(np.mean(seg))
        std   = float(np.std(seg)) + 1e-10
        bb_u  = sma + self.BB_STD * std
        bb_l  = sma - self.BB_STD * std
        bb_pct_b = (float(close[i]) - bb_l) / (bb_u - bb_l + 1e-10)
        bb_score = (bb_pct_b - 0.5) * 2.0

        # CCI
        cci_seg  = close[i - self.CCI_LEN + 1: i + 1]
        cci_sma  = float(np.mean(cci_seg))
        mean_dev = float(np.mean(np.abs(cci_seg - cci_sma))) + 1e-10
        cci      = (float(close[i]) - cci_sma) / (0.015 * mean_dev)
        cci_norm = float(np.clip(cci / 200.0, -1.0, 1.0))

        # ROC
        roc_val = (float(close[i]) / (float(close[i - self.ROC_LEN]) + 1e-10) - 1.0) * 100.0
        roc_hist = np.array([(float(close[k]) /
                              (float(close[k - self.ROC_LEN]) + 1e-10) - 1.0) * 100.0
                             for k in range(i - self.ROC_STD_LEN, i + 1)
                             if k >= self.ROC_LEN])
        roc_std  = float(np.std(roc_hist)) * self.ROC_STD_MULT + 1e-10
        roc_norm = float(np.clip(roc_val / roc_std, -1.0, 1.0))

        raw_disp = bb_score * 0.4 + cci_norm * 0.35 + roc_norm * 0.25

        # Volume-directional pressure
        vol_seg = volume[max(0, i - 20): i]
        vol_avg = float(np.mean(vol_seg)) if len(vol_seg) > 0 else 1.0
        vol_ratio = float(volume[i]) / (vol_avg + 1e-10)
        bar_dir   = 1.0 if float(close[i]) > float(open_[i]) else -1.0
        vol_press = vol_ratio * bar_dir * 0.2

        return float(np.clip(raw_disp + vol_press, -2.0, 2.0))

    # ── EMA ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, n: int) -> np.ndarray:
        out  = np.full(len(arr), np.nan)
        mult = 2.0 / (n + 1)
        if len(arr) < n:
            return out
        out[n - 1] = arr[:n].mean()
        for k in range(n, len(arr)):
            out[k] = arr[k] * mult + out[k - 1] * (1 - mult)
        return out

    @staticmethod
    def _ema_last_val(arr: np.ndarray, n: int, i: int) -> float:
        seg = arr[max(0, i - n * 3): i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < n:
            return float(np.nanmean(arr[max(0, i - n): i + 1]))
        mult = 2.0 / (n + 1)
        val  = float(np.mean(valid[:n]))
        for v in valid[n:]:
            val = float(v) * mult + val * (1 - mult)
        return val

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
    def _quality(sig_type: str, conf_score: float, displacement: float,
                 atr_ratio: float, context: dict, htf_bias: str) -> float:
        score = 5.5
        if abs(conf_score) > 0.6:
            score += 1.0
        elif abs(conf_score) > 0.4:
            score += 0.5
        if abs(displacement) > 0.6:
            score += 0.5
        if atr_ratio > 1.3:
            score += 0.5
        if context.get("adx", 0) > 25:
            score += 0.5
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 0.5
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
