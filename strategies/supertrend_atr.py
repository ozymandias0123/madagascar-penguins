"""
strategies/supertrend_atr.py
Supertrend + ATR Strategy

Logic
-----
  - Supertrend bands computed from hl2 ± Multiplier × ATR(Period)
  - Bands are ratcheted (never retrace against trend)
  - Trend = +1 when close > lower band, -1 when close < upper band
  - Entry: trend flip (+1 → buy, -1 → sell)
  - Consolidation filter: suppress entries when ATR < SMA(ATR) × atrMult
  - Exit: trailing stop (trail_offset = ATR × trailMult)
  - Position size scales with risk % and leverage (encoded in quality)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SupertrendATR(BaseStrategy):

    name        = "SupertrendATR"
    description = "Adaptive Supertrend + ATR consolidation filter + trailing stop"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    PERIODS      = 10
    MULTIPLIER   = 3.0
    RISK_PCT     = 5.0     # % of equity at risk — used to scale quality
    LEVERAGE     = 4.0
    TRAIL_MULT   = 1.5

    # Consolidation filter
    ATR_FILT_LEN  = 14
    ATR_FILT_MULT = 0.5
    SMA_ATR_LEN   = 20

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        min_len = self.PERIODS + self.SMA_ATR_LEN + 5
        if len(df) < min_len:
            return []

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── ATR ──────────────────────────────────────────────────────────────
        atr_val = self._calc_atr(df, self.PERIODS)

        # ── Consolidation filter ──────────────────────────────────────────────
        atr_filt     = self._calc_atr(df, self.ATR_FILT_LEN)
        sma_atr_filt = atr_filt.rolling(self.SMA_ATR_LEN).mean()
        consolidating = atr_val < sma_atr_filt * self.ATR_FILT_MULT

        # ── Supertrend ────────────────────────────────────────────────────────
        src = (high + low) / 2.0   # hl2

        up_raw = src - self.MULTIPLIER * atr_val
        dn_raw = src + self.MULTIPLIER * atr_val

        # Ratchet bands
        up  = up_raw.copy()
        dn  = dn_raw.copy()
        for k in range(1, len(df)):
            up.iloc[k]  = (max(float(up_raw.iloc[k]),  float(up.iloc[k - 1]))
                           if float(close.iloc[k - 1]) > float(up.iloc[k - 1])
                           else float(up_raw.iloc[k]))
            dn.iloc[k]  = (min(float(dn_raw.iloc[k]),  float(dn.iloc[k - 1]))
                           if float(close.iloc[k - 1]) < float(dn.iloc[k - 1])
                           else float(dn_raw.iloc[k]))

        # Trend direction
        trend = pd.Series(1, index=df.index)
        for k in range(1, len(df)):
            prev_trend = int(trend.iloc[k - 1])
            if prev_trend == -1 and float(close.iloc[k]) > float(dn.iloc[k - 1]):
                trend.iloc[k] = 1
            elif prev_trend == 1 and float(close.iloc[k]) < float(up.iloc[k - 1]):
                trend.iloc[k] = -1
            else:
                trend.iloc[k] = prev_trend

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_val.iloc[i]) if not np.isnan(atr_val.iloc[i]) else 10.0

        buy_signal  = (int(trend.iloc[i]) == 1  and int(trend.iloc[i - 1]) == -1)
        sell_signal = (int(trend.iloc[i]) == -1 and int(trend.iloc[i - 1]) == 1)
        in_consol   = bool(consolidating.iloc[i])

        if in_consol:
            return []

        # Quality encodes risk×leverage for engine to scale size
        quality = min(6.0 + (self.RISK_PCT * self.LEVERAGE / 20), 10.0)
        trail   = atr_v * self.TRAIL_MULT

        signals = []

        if buy_signal:
            sl = entry - atr_v * self.MULTIPLIER
            tp = entry + atr_v * self.MULTIPLIER * 1.5
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     quality,
                "zone":        {"high": float(dn.iloc[i]), "low": float(up.iloc[i])},
                "pattern_key": "supertrend_buy",
                "strategy":    self.name,
                "trail_offset": trail,
                "notes":       (f"Supertrend flip +1, ATR={atr_v:.4f}, "
                                f"risk={self.RISK_PCT}%×{self.LEVERAGE}x lev"),
            })

        if sell_signal:
            sl = entry + atr_v * self.MULTIPLIER
            tp = entry - atr_v * self.MULTIPLIER * 1.5
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     quality,
                "zone":        {"high": float(dn.iloc[i]), "low": float(up.iloc[i])},
                "pattern_key": "supertrend_sell",
                "strategy":    self.name,
                "trail_offset": trail,
                "notes":       (f"Supertrend flip -1, ATR={atr_v:.4f}, "
                                f"risk={self.RISK_PCT}%×{self.LEVERAGE}x lev"),
            })

        return signals

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
