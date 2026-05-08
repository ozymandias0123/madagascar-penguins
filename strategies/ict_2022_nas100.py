"""
strategies/ict_2022_nas100.py
ICT 2022 Smart Trend v2.0 — NAS100

Implements core ICT 2022 concepts:
  - Killzones (London 02:00–05:00, NY 07:00–10:00 UTC)
  - Market Structure Shift (MSS) — displacement candle breaks recent pivot
  - Liquidity sweep: wick beyond prior pivot then closes back
  - Fair Value Gap (FVG) confirmation
  - Displacement candle filter (large body relative to ATR)

Signal fires when: session killzone + MSS + liquidity sweep + FVG align.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from strategies.base_strategy import BaseStrategy


class ICT2022NAS100(BaseStrategy):

    name        = "ICT2022NAS100"
    description = "ICT 2022 killzone + MSS + sweep + FVG — NAS100 scalper"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    PIVOT_LEN         = 10       # swing pivot lookback
    DISPLACE_MULT     = 1.5      # body > 1.5× ATR = displacement
    ATR_MULT_SL       = 1.0
    ATR_MULT_TP       = 2.5
    KILLZONE_SESSIONS = {"london", "new_york"}

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.PIVOT_LEN * 3 + 10:
            return []

        if session not in self.KILLZONE_SESSIONS:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        atr_s  = df["atr"] if "atr" in df.columns else self._calc_atr(df, 14)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 20.0

        # ── Swing pivots ─────────────────────────────────────────────────────
        n = self.PIVOT_LEN
        pivot_high = float(high.iloc[i - n: i].max())
        pivot_low  = float(low.iloc[i - n: i].min())

        # ── Displacement candle ───────────────────────────────────────────────
        body      = abs(float(close.iloc[i]) - float(open_.iloc[i]))
        displaced = body > atr_v * self.DISPLACE_MULT
        bull_disp = displaced and float(close.iloc[i]) > float(open_.iloc[i])
        bear_disp = displaced and float(close.iloc[i]) < float(open_.iloc[i])

        # ── Market Structure Shift ─────────────────────────────────────────────
        # Bullish MSS: price broke above recent pivot high with displacement
        # Bearish MSS: price broke below recent pivot low with displacement
        mss_bull = float(close.iloc[i]) > pivot_high and bull_disp
        mss_bear = float(close.iloc[i]) < pivot_low  and bear_disp

        # ── Liquidity sweep ───────────────────────────────────────────────────
        # Bullish sweep: wick below pivot_low then closed above it
        sweep_bull = (float(low.iloc[i - 1]) < pivot_low and
                      float(close.iloc[i - 1]) > pivot_low)
        # Bearish sweep: wick above pivot_high then closed below it
        sweep_bear = (float(high.iloc[i - 1]) > pivot_high and
                      float(close.iloc[i - 1]) < pivot_high)

        # ── Fair Value Gap ────────────────────────────────────────────────────
        fvg_bull = float(high.iloc[i - 3]) < float(low.iloc[i - 1])
        fvg_bear = float(low.iloc[i - 3])  > float(high.iloc[i - 1])

        # ── HTF bias filter ───────────────────────────────────────────────────
        htf_ok_long  = htf_bias in ("bullish", "neutral", "")
        htf_ok_short = htf_bias in ("bearish", "neutral", "")

        signals = []

        if mss_bull and sweep_bull and fvg_bull and htf_ok_long:
            sl = entry - atr_v * self.ATR_MULT_SL
            tp = entry + atr_v * self.ATR_MULT_TP
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     8.5,
                "zone":        {"high": pivot_high, "low": pivot_low},
                "pattern_key": "ict_mss_sweep_bull",
                "strategy":    self.name,
                "notes": (f"ICT bull: MSS + sweep + FVG, "
                          f"session={session}, body={body:.1f} atr={atr_v:.1f}"),
            })

        if mss_bear and sweep_bear and fvg_bear and htf_ok_short:
            sl = entry + atr_v * self.ATR_MULT_SL
            tp = entry - atr_v * self.ATR_MULT_TP
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     8.5,
                "zone":        {"high": pivot_high, "low": pivot_low},
                "pattern_key": "ict_mss_sweep_bear",
                "strategy":    self.name,
                "notes": (f"ICT bear: MSS + sweep + FVG, "
                          f"session={session}, body={body:.1f} atr={atr_v:.1f}"),
            })

        return signals

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
