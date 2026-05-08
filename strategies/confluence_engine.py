"""
strategies/confluence_engine.py
Confluence Engine Strategy  [CES — JOAT]

Logic
-----
  - Core trigger : Linear Regression line crosses its SMA signal line
  - Trend filter : Fast EMA > Slow EMA (long) / < (short)
  - HTF bias     : Approximated with longer EMA spans (no live security())
  - Volume spike : optional volume > SMA(volume) × multiplier
  - RSI cap      : RSI < 72 for longs / RSI > 28 for shorts
  - Regime filter: ATR14 / ATR50 ratio > threshold (trending, not ranging)
  - Confluence score (0-100) must exceed i_confScore to fire
  - Exit:  ATR-based TP/SL + timeout (bars) + trend-flip close
  - Cooldown: N bars after any exit before next entry
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class ConfluenceEngine(BaseStrategy):

    name        = "ConfluenceEngine"
    description = "LR cross + EMA + HTF + volume + RSI + regime + confluence score"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    LR_LEN       = 9
    SIG_LEN      = 5
    FAST_EMA     = 20
    SLOW_EMA     = 50
    ATR_LEN      = 14
    REG_LEN      = 50
    REG_THRESH   = 0.85
    MIN_SCORE    = 50
    VOL_LEN      = 20
    VOL_MULT     = 1.0
    RSI_LEN      = 14
    RSI_BULL_MAX = 72
    RSI_BEAR_MIN = 28
    SL_ATR       = 1.5
    TP_ATR       = 2.5
    TIMEOUT_BARS = 20
    COOLDOWN     = 3

    def __init__(self):
        super().__init__() if hasattr(super(), "__init__") else None
        self._cooldown_left: int = 0
        self._bars_in_trade: int = 0
        self._in_trade:      bool = False

    def on_trade_closed(self, result: Dict) -> None:
        self._cooldown_left = self.COOLDOWN
        self._in_trade      = False
        self._bars_in_trade = 0

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        warmup = max(self.SLOW_EMA * 2, self.REG_LEN * 2)
        if len(df) < warmup:
            return []

        # Cooldown counter ticks on every generate call
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return []

        close  = df["close"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        atr_s  = df["atr"]   if "atr"    in df.columns else self._calc_atr(df, self.ATR_LEN)

        # ── Linear regression line (rolling slope × position + intercept) ────
        n = self.LR_LEN
        lr_close = close.rolling(n).apply(
            lambda x: np.polyval(np.polyfit(np.arange(len(x)), x, 1),
                                 len(x) - 1), raw=True)
        sig_line = lr_close.rolling(self.SIG_LEN).mean()

        # ── EMAs ─────────────────────────────────────────────────────────────
        fast_ema = close.ewm(span=self.FAST_EMA, adjust=False).mean()
        slow_ema = close.ewm(span=self.SLOW_EMA, adjust=False).mean()

        # HTF proxy: 3× slow span
        htf_fast = close.ewm(span=self.FAST_EMA * 3, adjust=False).mean()
        htf_slow = close.ewm(span=self.SLOW_EMA * 3, adjust=False).mean()

        # ── ATR regime ────────────────────────────────────────────────────────
        atr14   = atr_s
        atr_reg = self._calc_atr(df, self.REG_LEN)

        # ── RSI & volume ─────────────────────────────────────────────────────
        rsi    = self._rsi(close, self.RSI_LEN)
        vol_sma = volume.rolling(self.VOL_LEN).mean().replace(0, 1)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) else 10.0
        atr_r = float(atr_reg.iloc[i]) if not np.isnan(atr_reg.iloc[i]) else atr_v
        r     = float(rsi.iloc[i])
        vr    = float(volume.iloc[i]) / float(vol_sma.iloc[i])

        trend_up  = float(fast_ema.iloc[i]) > float(slow_ema.iloc[i])
        trend_dn  = float(fast_ema.iloc[i]) < float(slow_ema.iloc[i])
        htf_bull  = float(htf_fast.iloc[i]) > float(htf_slow.iloc[i])
        htf_bear  = float(htf_fast.iloc[i]) < float(htf_slow.iloc[i])

        atr_ratio = atr_v / max(atr_r, 1e-9)
        is_trending = atr_ratio > self.REG_THRESH
        vol_ok      = vr > self.VOL_MULT

        # LR crossover
        lr_cross_up  = (float(lr_close.iloc[i - 1]) <= float(sig_line.iloc[i - 1]) and
                        float(lr_close.iloc[i])     >  float(sig_line.iloc[i]))
        lr_cross_dn  = (float(lr_close.iloc[i - 1]) >= float(sig_line.iloc[i - 1]) and
                        float(lr_close.iloc[i])     <  float(sig_line.iloc[i]))

        # ── Confluence scores ─────────────────────────────────────────────────
        def score_long() -> int:
            s  = 25 if trend_up  else 0
            s += 20 if htf_bull  else 0
            s += 20 if vr > 1.2  else (10 if vol_ok else 0)
            s += int(min((55.0 - r) * 0.5, 20.0)) if r < 50 else 0
            s += 15 if is_trending else 0
            return min(s, 100)

        def score_short() -> int:
            s  = 25 if trend_dn  else 0
            s += 20 if htf_bear  else 0
            s += 20 if vr > 1.2  else (10 if vol_ok else 0)
            s += int(min((r - 45.0) * 0.5, 20.0)) if r > 50 else 0
            s += 15 if is_trending else 0
            return min(s, 100)

        long_score  = score_long()  if lr_cross_up else 0
        short_score = score_short() if lr_cross_dn else 0

        bull_sig = (lr_cross_up and trend_up and htf_bull and vol_ok and
                    r < self.RSI_BULL_MAX and is_trending and
                    long_score >= self.MIN_SCORE)

        bear_sig = (lr_cross_dn and trend_dn and htf_bear and vol_ok and
                    r > self.RSI_BEAR_MIN and is_trending and
                    short_score >= self.MIN_SCORE)

        signals = []

        if bull_sig:
            sl = entry - atr_v * self.SL_ATR
            tp = entry + atr_v * self.TP_ATR
            q  = min(5.0 + long_score * 0.04, 10.0)
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     q,
                "zone":        {"high": entry + atr_v * 0.3,
                                "low":  entry - atr_v * 0.3},
                "pattern_key": f"ces_long_{long_score}",
                "strategy":    self.name,
                "timeout_bars": self.TIMEOUT_BARS,
                "notes":       (f"CES long: score={long_score}, "
                                f"trend={'up' if trend_up else 'dn'}, "
                                f"htf={'bull' if htf_bull else 'bear'}, "
                                f"atr_ratio={atr_ratio:.3f}"),
            })

        if bear_sig:
            sl = entry + atr_v * self.SL_ATR
            tp = entry - atr_v * self.TP_ATR
            q  = min(5.0 + short_score * 0.04, 10.0)
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     q,
                "zone":        {"high": entry + atr_v * 0.3,
                                "low":  entry - atr_v * 0.3},
                "pattern_key": f"ces_short_{short_score}",
                "strategy":    self.name,
                "timeout_bars": self.TIMEOUT_BARS,
                "notes":       (f"CES short: score={short_score}, "
                                f"trend={'dn' if trend_dn else 'up'}, "
                                f"htf={'bear' if htf_bear else 'bull'}, "
                                f"atr_ratio={atr_ratio:.3f}"),
            })

        return signals

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
