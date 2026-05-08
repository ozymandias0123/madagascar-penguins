"""
strategies/bayesian_kelly.py
Bayesian Kelly Strategy  (v1.0)

Logic
-----
  Bayesian posterior Kelly fraction:

    Prior (long-run, lookbackPrior = 252 bars):
      p_returns = log(close / close[-1])  over prior window
      p_mu      = mean(p_returns)
      p_var     = var(p_returns)

    Sample (recent, lookbackSample = 20 bars):
      s_returns = log(close / close[-1])  over sample window
      s_mu      = mean(s_returns)
      s_var     = var(s_returns)

    Blended posterior Kelly:
      f_bayes = ((1-alpha) × p_mu/p_var  +  alpha × s_mu/s_var) × kelly_frac
    where alpha = 0.3, kelly_frac = 0.25

    Clip to [min_lever, max_lever] = [0.05, 2.0]

  Signal:
    Only triggered every `interval` (7) bars.
    f_bayes > clip_min  → long  (positive leverage)
    f_bayes < -clip_min → short (negative = mean-reversion)

    (If the posterior is near zero, no signal.)

  SL / TP (advisory, ATR-based):
    SL = close ∓ ATR(14) × 2.0
    TP = close ± ATR(14) × 3.0
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class BayesianKelly(BaseStrategy):

    name        = "BayesianKelly"
    description = "Bayesian posterior Kelly fraction; rebalance every interval bars"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    LOOKBACK_PRIOR  = 252
    LOOKBACK_SAMPLE = 20
    ALPHA           = 0.3       # weight on sample vs prior
    KELLY_FRAC      = 0.25      # fractional Kelly
    MIN_LEVER       = 0.05      # minimum |f| to generate a signal
    MAX_LEVER       = 2.0       # max leverage clip
    INTERVAL        = 7         # rebalance every N bars
    ATR_LEN         = 14
    SL_MULT         = 2.0
    TP_MULT         = 3.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = self.LOOKBACK_PRIOR + self.ATR_LEN + 5
        if len(df) < needed:
            return []

        close = df["close"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1

        # ── Rebalance gate ────────────────────────────────────────────────────
        if i % self.INTERVAL != 0:
            return []

        if np.isnan(atr_s[i]):
            return []

        atr_val = float(atr_s[i])

        # ── Log returns ───────────────────────────────────────────────────────
        prior_start  = i - self.LOOKBACK_PRIOR
        sample_start = i - self.LOOKBACK_SAMPLE

        if prior_start < 1 or sample_start < 1:
            return []

        prior_prices  = close[prior_start:  i + 1]
        sample_prices = close[sample_start: i + 1]

        p_returns = np.log(prior_prices[1:] / (prior_prices[:-1] + 1e-10))
        s_returns = np.log(sample_prices[1:] / (sample_prices[:-1] + 1e-10))

        p_mu  = float(np.mean(p_returns))
        p_var = float(np.var(p_returns)) + 1e-12

        s_mu  = float(np.mean(s_returns))
        s_var = float(np.var(s_returns)) + 1e-12

        # ── Bayesian Kelly ────────────────────────────────────────────────────
        alpha = self.ALPHA
        raw_f = ((1 - alpha) * p_mu / p_var + alpha * s_mu / s_var) * self.KELLY_FRAC
        f_bayes = float(np.clip(raw_f, -self.MAX_LEVER, self.MAX_LEVER))

        if abs(f_bayes) < self.MIN_LEVER:
            return []

        long_ok  = f_bayes > 0
        short_ok = f_bayes < 0

        sig_type = "buy" if long_ok else "sell"
        entry    = float(close[i])

        sl = (entry - atr_val * self.SL_MULT if sig_type == "buy"
              else entry + atr_val * self.SL_MULT)
        tp = (entry + atr_val * self.TP_MULT  if sig_type == "buy"
              else entry - atr_val * self.TP_MULT)

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        quality = self._quality(sig_type, f_bayes, p_mu, s_mu, context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(entry + atr_val, 5),
                            "low":  round(entry - atr_val, 5)},
            "pattern_key": f"bk_{sig_type}_f{abs(f_bayes):.2f}",
            "strategy":    self.name,
            "notes":       (f"f_bayes={f_bayes:.4f} | "
                            f"p_mu={p_mu:.6f} p_var={p_var:.8f} | "
                            f"s_mu={s_mu:.6f} s_var={s_var:.8f} | "
                            f"alpha={alpha} kelly={self.KELLY_FRAC} | "
                            f"rebal_every={self.INTERVAL}bars"),
        }]

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
    def _quality(sig_type: str, f_bayes: float, p_mu: float, s_mu: float,
                 context: dict, htf_bias: str) -> float:
        score = 5.0
        lev   = abs(f_bayes)
        if lev > 1.0:
            score += 2.0
        elif lev > 0.5:
            score += 1.0
        elif lev > 0.2:
            score += 0.5
        # Agreement between prior and sample
        if (p_mu > 0 and s_mu > 0) or (p_mu < 0 and s_mu < 0):
            score += 0.5
        if context.get("adx", 0) > 25:
            score += 0.5
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 0.5
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
