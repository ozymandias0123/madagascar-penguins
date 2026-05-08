"""
strategies/martingale_sfp.py
Martingale Liquidity Sweep + HMM Regime Strategy

Swing Failure Pattern (SFP) — price sweeps pivot then reverses.
Volume anomaly confirms institutional absorption.
Simple HMM proxy: EMA50 rising + above EMA200.
Martingale: quality score increases with loss streak (engine scales size).
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class MartingaleSFP(BaseStrategy):

    name        = "MartingaleSFP"
    description = "Swing Failure Pattern + volume anomaly + HMM regime + martingale sizing"
    version     = "1.0"

    SWING_LEN        = 10
    MARTINGALE_MULT  = 2.0
    MAX_LEVELS       = 4
    CATASTROPHE_PCT  = 0.04   # close all if 4% adverse

    def __init__(self):
        super().__init__() if hasattr(super(), "__init__") else None
        self._loss_streak: int = 0

    # ── loss streak tracking ──────────────────────────────────

    def on_trade_closed(self, result: Dict) -> None:
        if result.get("pnl", 0) < 0:
            self._loss_streak = min(self._loss_streak + 1, self.MAX_LEVELS)
        else:
            self._loss_streak = 0

    # ── signal generation ─────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 210:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        atr_s  = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()

        sw = self.SWING_LEN

        # pivot highs / lows (confirmed — shifted by swing length)
        rolling_max = high.rolling(2 * sw + 1, center=True).max()
        rolling_min = low.rolling( 2 * sw + 1, center=True).min()
        pivot_h = high.where(high == rolling_max).shift(sw).ffill()
        pivot_l = low.where( low  == rolling_min).shift(sw).ffill()

        # Swing Failure Pattern
        # SFP Long:  current LOW swept below pivot_low then CLOSED back above it
        sfp_long  = (low < pivot_l.shift(1)) & (close > pivot_l.shift(1))
        # SFP Short: current HIGH swept above pivot_high then CLOSED back below it
        sfp_short = (high > pivot_h.shift(1)) & (close < pivot_h.shift(1))

        # Volume anomaly: volume > 2× its 20-bar SMA
        vol_sma     = volume.rolling(20).mean()
        vol_anomaly = volume > vol_sma * 2.0

        # HMM regime proxy: EMA50 rising for 5 bars AND price > EMA200
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        regime_bull = (
            (ema50 > ema50.shift(1)) &
            (ema50.shift(1) > ema50.shift(2)) &
            (ema50.shift(2) > ema50.shift(3)) &
            (ema50.shift(3) > ema50.shift(4)) &
            (ema50.shift(4) > ema50.shift(5)) &
            (close > ema200)
        )

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        long_ok  = sfp_long.iloc[i]  and vol_anomaly.iloc[i] and regime_bull.iloc[i]
        short_ok = sfp_short.iloc[i] and vol_anomaly.iloc[i] and (not regime_bull.iloc[i])

        # Martingale: scale quality so engine gives bigger position after losses
        streak = min(self._loss_streak, self.MAX_LEVELS)
        mart_scale = self.MARTINGALE_MULT ** streak
        base_quality = 7.0
        quality = min(base_quality * mart_scale, 10.0)

        signals = []

        if long_ok:
            sl = entry - atr_v * 1.5
            tp = entry * 1.10          # +10% target (original strategy)
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     quality,
                "zone":        {"high": float(pivot_l.iloc[i]) + atr_v * 0.3,
                                "low":  float(pivot_l.iloc[i]) - atr_v * 0.3},
                "pattern_key": f"sfp_long_L{streak}",
                "strategy":    self.name,
                "notes":       (f"SFP long, vol={volume.iloc[i]:.0f}>2×SMA, "
                                f"regime_bull, streak={streak}"),
            })

        if short_ok:
            sl = entry + atr_v * 1.5
            tp = entry * 0.90
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     quality,
                "zone":        {"high": float(pivot_h.iloc[i]) + atr_v * 0.3,
                                "low":  float(pivot_h.iloc[i]) - atr_v * 0.3},
                "pattern_key": f"sfp_short_L{streak}",
                "strategy":    self.name,
                "notes":       (f"SFP short, vol anomaly, "
                                f"no bull regime, streak={streak}"),
            })

        return signals

    def validate_signal(self, signal: Dict, balance: float) -> bool:
        # Block signal if catastrophe condition would be hit
        # (basic guard — real check happens in engine on open position)
        return signal.get("quality", 0) > 0
