"""
strategies/masu_v12.py
MASU+ v12 Conformal Edge

Advanced multi-signal strategy:
  - Kalman Filter (adaptive trend smoothing with process noise scaling)
  - Adaptive Conformal Inference (ACI) — online calibrated prediction intervals
  - Multi-timeframe EMA ribbon (3 spans) as trend proxy
  - Smart Money Concepts: FVG + OB (Order Block) detection
  - VWAP deviation bands
  - CVD proxy (cumulative delta approximation from close vs open)
  - Volatility regime (ATR percentile: low / mid / high)
  - Signal only fires when 4+ filters agree
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanFilter1D:
    """
    Scalar 1-D Kalman filter for price smoothing.
    Process noise Q is adaptive: scales with recent price variance.
    """

    def __init__(self, q: float = 0.01, r: float = 1.0):
        self.q = q      # process noise base
        self.r = r      # measurement noise
        self._x: float = 0.0    # state estimate
        self._p: float = 1.0    # error covariance
        self._initialized = False

    def update(self, measurement: float, adaptive_q: float = 0.0) -> float:
        if not self._initialized:
            self._x = measurement
            self._initialized = True
            return self._x

        q_eff = self.q + adaptive_q    # adaptive process noise

        # Predict
        p_pred = self._p + q_eff

        # Update
        k      = p_pred / (p_pred + self.r)   # Kalman gain
        self._x = self._x + k * (measurement - self._x)
        self._p = (1 - k) * p_pred

        return self._x

    def apply_series(self, series: pd.Series, vol_series: pd.Series) -> pd.Series:
        """Apply filter to a price series with volatility-driven adaptive Q."""
        self._x = 0.0
        self._p = 1.0
        self._initialized = False
        results = []
        vol_vals = vol_series.fillna(0).values
        for v, vol in zip(series.values, vol_vals):
            adaptive_q = vol * 0.001 if not np.isnan(vol) else 0.0
            results.append(self.update(float(v), adaptive_q))
        return pd.Series(results, index=series.index)


# ── ACI (Adaptive Conformal Inference) ───────────────────────────────────────

class OnlineACI:
    """
    Online conformal prediction interval.
    Maintains a rolling calibration set of residuals.
    Returns (lower, upper) interval for the next bar.
    """

    def __init__(self, alpha: float = 0.1, window: int = 50):
        self.alpha  = alpha       # miscoverage target (0.1 → 90% interval)
        self.window = window
        self._residuals: list = []

    def calibrate(self, actual: float, predicted: float) -> None:
        self._residuals.append(abs(actual - predicted))
        if len(self._residuals) > self.window:
            self._residuals.pop(0)

    def interval(self, prediction: float) -> tuple:
        if not self._residuals:
            return (prediction, prediction)
        q_level = np.quantile(self._residuals, 1 - self.alpha)
        return (prediction - q_level, prediction + q_level)

    def apply_series(self, actual: pd.Series, predicted: pd.Series
                     ) -> tuple:
        """Return (lower_series, upper_series) using growing calibration."""
        lowers, uppers = [], []
        cal = OnlineACI(self.alpha, self.window)
        for a, p in zip(actual.values, predicted.values):
            lo, hi = cal.interval(float(p))
            lowers.append(lo)
            uppers.append(hi)
            if not np.isnan(a):
                cal.calibrate(float(a), float(p))
        return (pd.Series(lowers, index=actual.index),
                pd.Series(uppers, index=actual.index))


# ── Strategy ─────────────────────────────────────────────────────────────────

class MASUv12(BaseStrategy):

    name        = "MASUv12"
    description = "Kalman + ACI + MTF ribbon + FVG/OB + VWAP + CVD + vol regime"
    version     = "1.2"

    # ── parameters ───────────────────────────────────────────────────────────
    KALMAN_Q         = 0.01
    KALMAN_R         = 1.0
    ACI_ALPHA        = 0.1       # 90% conformal interval
    ACI_WINDOW       = 50
    RIBBON_SPANS     = (8, 21, 55)
    VWAP_WINDOW      = 50
    VWAP_STD_MULT    = 1.5
    CVD_WINDOW       = 20
    ATR_PCT_WINDOW   = 50        # window for ATR percentile
    MIN_CONFIRMS     = 4         # signals need this many confirmations
    ATR_MULT_SL      = 1.2
    ATR_MULT_TP      = 3.0

    def __init__(self):
        super().__init__() if hasattr(super(), "__init__") else None
        self._kf  = KalmanFilter1D(q=self.KALMAN_Q, r=self.KALMAN_R)
        self._aci = OnlineACI(alpha=self.ACI_ALPHA, window=self.ACI_WINDOW)

    # ── main ─────────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        min_len = max(self.RIBBON_SPANS) + self.ATR_PCT_WINDOW + 10
        if len(df) < min_len:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        atr_s  = df["atr"]   if "atr"    in df.columns else self._calc_atr(df, 14)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        # ── Kalman smoothed price ─────────────────────────────────────────────
        kf_price = self._kf.apply_series(close, atr_s)
        kf_now   = float(kf_price.iloc[i])
        kf_prev  = float(kf_price.iloc[i - 1])
        kf_up    = kf_now > kf_prev
        kf_down  = kf_now < kf_prev

        # ── ACI prediction interval ───────────────────────────────────────────
        aci_lo, aci_hi = self._aci.apply_series(close, kf_price)
        in_interval    = float(aci_lo.iloc[i]) <= entry <= float(aci_hi.iloc[i])
        above_interval = entry > float(aci_hi.iloc[i])
        below_interval = entry < float(aci_lo.iloc[i])

        # ── MTF EMA ribbon ────────────────────────────────────────────────────
        emas = [close.ewm(span=s, adjust=False).mean() for s in self.RIBBON_SPANS]
        ribbon_bull = all(float(emas[j].iloc[i]) > float(emas[j + 1].iloc[i])
                          for j in range(len(emas) - 1))
        ribbon_bear = all(float(emas[j].iloc[i]) < float(emas[j + 1].iloc[i])
                          for j in range(len(emas) - 1))

        # ── FVG ───────────────────────────────────────────────────────────────
        fvg_bull = float(high.iloc[i - 3]) < float(low.iloc[i - 1])
        fvg_bear = float(low.iloc[i - 3])  > float(high.iloc[i - 1])

        # ── Order Block (OB) ──────────────────────────────────────────────────
        # Bullish OB: last bearish candle before impulsive up move
        bull_impulsive = (float(close.iloc[i]) - float(close.iloc[i - 3])) > atr_v * 1.5
        ob_bull_zone   = float(low.iloc[i - 2]) <= entry <= float(high.iloc[i - 2])
        ob_bull        = bull_impulsive and ob_bull_zone

        bear_impulsive = (float(close.iloc[i - 3]) - float(close.iloc[i])) > atr_v * 1.5
        ob_bear_zone   = float(low.iloc[i - 2]) <= entry <= float(high.iloc[i - 2])
        ob_bear        = bear_impulsive and ob_bear_zone

        # ── VWAP deviation ────────────────────────────────────────────────────
        typical  = (high + low + close) / 3
        vol_safe = volume.replace(0, 1)
        vwap     = ((typical * vol_safe).rolling(self.VWAP_WINDOW).sum()
                    / vol_safe.rolling(self.VWAP_WINDOW).sum())
        vwap_std = typical.rolling(self.VWAP_WINDOW).std()
        vwap_up  = vwap + self.VWAP_STD_MULT * vwap_std
        vwap_dn  = vwap - self.VWAP_STD_MULT * vwap_std

        vwap_v  = float(vwap.iloc[i])
        above_vwap = entry > vwap_v
        below_vwap = entry < vwap_v
        # Expansion: price beyond VWAP ± band
        vwap_exp_up = entry > float(vwap_up.iloc[i])
        vwap_exp_dn = entry < float(vwap_dn.iloc[i])

        # ── CVD proxy ─────────────────────────────────────────────────────────
        # Approximation: delta = (close - open) / (high - low + 0.001)
        delta      = (close - open_) / (high - low + 0.001)
        cvd        = delta.rolling(self.CVD_WINDOW).sum()
        cvd_rising = float(cvd.iloc[i]) > float(cvd.iloc[i - 1])
        cvd_fall   = float(cvd.iloc[i]) < float(cvd.iloc[i - 1])

        # ── Volatility regime ─────────────────────────────────────────────────
        atr_pct = atr_s.rolling(self.ATR_PCT_WINDOW).rank(pct=True)
        vol_pct = float(atr_pct.iloc[i]) if not np.isnan(atr_pct.iloc[i]) else 0.5
        # low < 0.33, mid 0.33–0.66, high > 0.66
        vol_regime = "low" if vol_pct < 0.33 else ("high" if vol_pct > 0.66 else "mid")

        # ── Confluence count ──────────────────────────────────────────────────
        # Each True confirmation adds 1 point
        bull_score = sum([
            kf_up,
            ribbon_bull,
            fvg_bull,
            ob_bull,
            above_vwap,
            cvd_rising,
            (htf_bias in ("bullish", "")),
            (vol_regime != "high"),   # prefer trading in non-extreme vol
        ])

        bear_score = sum([
            kf_down,
            ribbon_bear,
            fvg_bear,
            ob_bear,
            below_vwap,
            cvd_fall,
            (htf_bias in ("bearish", "")),
            (vol_regime != "high"),
        ])

        signals = []

        if bull_score >= self.MIN_CONFIRMS:
            sl = entry - atr_v * self.ATR_MULT_SL
            tp = entry + atr_v * self.ATR_MULT_TP
            q  = min(5.0 + bull_score * 0.5, 10.0)
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     q,
                "zone":        {"high": float(vwap_up.iloc[i]), "low": vwap_v},
                "pattern_key": f"masu_bull_{bull_score}",
                "strategy":    self.name,
                "notes": (f"MASU+ bull: score={bull_score}/8, "
                          f"kf={kf_now:.2f}, vol_regime={vol_regime}, "
                          f"vwap={vwap_v:.2f}"),
            })

        if bear_score >= self.MIN_CONFIRMS:
            sl = entry + atr_v * self.ATR_MULT_SL
            tp = entry - atr_v * self.ATR_MULT_TP
            q  = min(5.0 + bear_score * 0.5, 10.0)
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     q,
                "zone":        {"high": vwap_v, "low": float(vwap_dn.iloc[i])},
                "pattern_key": f"masu_bear_{bear_score}",
                "strategy":    self.name,
                "notes": (f"MASU+ bear: score={bear_score}/8, "
                          f"kf={kf_now:.2f}, vol_regime={vol_regime}, "
                          f"vwap={vwap_v:.2f}"),
            })

        return signals

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
