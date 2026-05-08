"""
core/market_structure.py — ICT market-structure detection helpers.
Ported from ozy.py / MarketStructureDetector unchanged.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import talib

from config import Config


class MarketStructureDetector:

    # ── Market Context / Regime ───────────────────────────────

    @staticmethod
    def get_market_context(df: pd.DataFrame) -> Dict:
        if not Config.REGIME_DETECTION_ENABLED or len(df) < 15:
            return {'regime': 'trending', 'volatility': 'normal', 'adx': 0, 'atr_ratio': 1.0}
        try:
            completed = df.iloc[:-1]
            adx_vals  = talib.ADX(
                completed['high'].values,
                completed['low'].values,
                completed['close'].values,
                timeperiod=14
            )
            adx_cur  = adx_vals[-1] if len(adx_vals) > 0 and not np.isnan(adx_vals[-1]) else 20
            atr_cur  = completed['atr'].iloc[-1]  if 'atr' in completed.columns else 0
            atr_mean = completed['atr'].mean()     if 'atr' in completed.columns else atr_cur
            atr_ratio = atr_cur / atr_mean if atr_mean > 0 else 1.0

            regime    = 'trending' if adx_cur > Config.ADX_TRENDING_THRESHOLD else 'ranging'
            volatility = (
                'high'   if atr_ratio > Config.ATR_HIGH_VOLATILITY_MULTIPLIER else
                'low'    if atr_ratio < 0.7 else
                'normal'
            )
            context = {
                'regime':    regime,
                'volatility': volatility,
                'adx':        round(float(adx_cur), 1),
                'atr_ratio':  round(float(atr_ratio), 2)
            }
            logging.info(
                f"[MARKET_CONTEXT] Regime={regime.upper()}, "
                f"Vol={volatility.upper()}, ADX={context['adx']}, "
                f"ATR_ratio={context['atr_ratio']}"
            )
            return context
        except Exception as exc:
            logging.error(f"[CONTEXT_ERROR] {exc}")
            return {'regime': 'trending', 'volatility': 'normal', 'adx': 0, 'atr_ratio': 1.0}

    # ── Market Structure ──────────────────────────────────────

    @staticmethod
    def detect_market_structure(df: pd.DataFrame, lookback: int = 60) -> str:
        if len(df) < 21:
            return 'no_structure'
        completed = df.iloc[:-1]
        recent    = completed.tail(lookback)
        if len(recent) < 20:
            return 'no_structure'
        sp = int(len(recent) * 0.6)
        ph = recent['high'].iloc[:sp].max()
        pl = recent['low'].iloc[:sp].min()
        ch = recent['high'].iloc[sp:].max()
        cl = recent['low'].iloc[sp:].min()
        if   ch > ph and cl > pl: return 'bullish_bos'
        elif ch < ph and cl < pl: return 'bearish_bos'
        elif ch < ph and cl > pl: return 'bullish_choch'
        elif ch > ph and cl < pl: return 'bearish_choch'
        return 'no_structure'

    # ── FVG ───────────────────────────────────────────────────

    @staticmethod
    def find_fvg(df: pd.DataFrame, atr: float) -> List[Dict]:
        if len(df) < 4:
            return []
        bullish   = (df['low'].shift(3) > df['high'].shift(1)) & \
                    (df['close'].shift(2) < df['low'].shift(3))
        bearish   = (df['high'].shift(3) < df['low'].shift(1)) & \
                    (df['close'].shift(2) > df['high'].shift(3))
        gap_b     = df['low'].shift(3) - df['high'].shift(1)
        gap_r     = df['low'].shift(1) - df['high'].shift(3)
        threshold = max(0.15 * atr, Config.FVG_SIZE_THRESHOLD)
        fvg_zones: List[Dict] = []

        for i in df.index[bullish & (gap_b > threshold)]:
            pos = df.index.get_loc(i)
            if pos >= 3:
                fvg_zones.append({
                    'type': 'bullish',
                    'high': df.iloc[pos - 1]['high'],
                    'low':  df.iloc[pos - 3]['low'],
                    'time': df.index[pos - 1]
                })
        for i in df.index[bearish & (gap_r > threshold)]:
            pos = df.index.get_loc(i)
            if pos >= 3:
                fvg_zones.append({
                    'type': 'bearish',
                    'high': df.iloc[pos - 3]['high'],
                    'low':  df.iloc[pos - 1]['low'],
                    'time': df.index[pos - 1]
                })
        return fvg_zones[-3:]

    # ── Order Block ───────────────────────────────────────────

    @staticmethod
    def find_order_block(df: pd.DataFrame, lookback: int = 50) -> Dict:
        if len(df) < 4:
            return {}
        completed = df.iloc[:-1]
        recent    = completed.tail(lookback)
        if len(recent) < 3:
            return {}
        for i in range(len(recent) - 1, 2, -1):
            ob = i - 2
            cn = i - 1
            if ob < 0 or cn < 0:
                continue
            if (recent['close'].iloc[ob] > recent['open'].iloc[ob] and
                    recent['close'].iloc[cn] < recent['open'].iloc[cn] and
                    (recent['low'].iloc[cn] - recent['low'].iloc[ob]) > recent['atr'].iloc[ob] * 0.4):
                return {'type': 'bullish_ob',
                        'high': recent['high'].iloc[ob],
                        'low':  recent['low'].iloc[ob],
                        'time': recent.index[ob], 'strength': 1.0}
            if (recent['close'].iloc[ob] < recent['open'].iloc[ob] and
                    recent['close'].iloc[cn] > recent['open'].iloc[cn] and
                    (recent['high'].iloc[ob] - recent['high'].iloc[cn]) > recent['atr'].iloc[ob] * 0.4):
                return {'type': 'bearish_ob',
                        'high': recent['high'].iloc[ob],
                        'low':  recent['low'].iloc[ob],
                        'time': recent.index[ob], 'strength': 1.0}
        return {}

    # ── Breaker Block ─────────────────────────────────────────

    @staticmethod
    def find_breaker_block(df: pd.DataFrame, lookback: int = 50) -> Dict:
        if len(df) < 11:
            return {}
        completed = df.iloc[:-1]
        recent    = completed.tail(lookback)
        if len(recent) < 10:
            return {}
        for i in range(len(recent) - 1, 5, -1):
            ob = i - 4
            if ob < 0: continue
            if (recent['close'].iloc[ob] > recent['open'].iloc[ob] and
                    recent['low'].iloc[ob + 1:i].min() < recent['low'].iloc[ob] and
                    recent['close'].iloc[i] < recent['low'].iloc[ob]):
                return {'type': 'bearish_breaker',
                        'high': recent['high'].iloc[ob],
                        'low':  recent['low'].iloc[ob],
                        'time': recent.index[ob], 'strength': 1.5}
        for i in range(len(recent) - 1, 5, -1):
            ob = i - 4
            if ob < 0: continue
            if (recent['close'].iloc[ob] < recent['open'].iloc[ob] and
                    recent['high'].iloc[ob + 1:i].max() > recent['high'].iloc[ob] and
                    recent['close'].iloc[i] > recent['high'].iloc[ob]):
                return {'type': 'bullish_breaker',
                        'high': recent['high'].iloc[ob],
                        'low':  recent['low'].iloc[ob],
                        'time': recent.index[ob], 'strength': 1.5}
        return {}

    # ── Premium / Discount ────────────────────────────────────

    @staticmethod
    def is_premium_discount(df: pd.DataFrame) -> str:
        if len(df) < 2:
            return 'equilibrium'
        completed = df.iloc[:-1]
        recent    = completed.tail(min(100, len(completed)))
        if len(recent) < 10:
            return 'equilibrium'
        rh  = recent['high'].max()
        rl  = recent['low'].min()
        cp  = recent['close'].iloc[-1]
        rng = rh - rl
        if rng == 0:
            return 'equilibrium'
        pos = (cp - rl) / rng
        return 'premium' if pos >= 0.70 else 'discount' if pos <= 0.30 else 'equilibrium'

    # ── Liquidity Sweep ───────────────────────────────────────

    @staticmethod
    def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Dict:
        if len(df) < 11:
            return {}
        completed = df.iloc[:-1]
        recent    = completed.tail(lookback)
        if len(recent) < 10:
            return {}
        older   = recent.iloc[:-4]
        recent4 = recent.iloc[-4:]
        if len(older) < 5:
            return {}
        max_high = older['high'].max()
        min_low  = older['low'].min()
        cp       = recent4['close'].iloc[-1]
        if (len(older['low'][older['low'] <= min_low * 1.001]) >= 2 and
                recent4['low'].min() < min_low and cp > recent4['close'].iloc[-2]):
            return {'type': 'bullish_sweep', 'level': min_low, 'liquidity': 'SSL'}
        if (len(older['high'][older['high'] >= max_high * 0.999]) >= 2 and
                recent4['high'].max() > max_high and cp < recent4['close'].iloc[-2]):
            return {'type': 'bearish_sweep', 'level': max_high, 'liquidity': 'BSL'}
        return {}

    # ── Power of 3 ────────────────────────────────────────────

    @staticmethod
    def detect_power_of_3(df: pd.DataFrame, lookback: int = 20) -> Dict:
        if len(df) < 16:
            return {}
        completed = df.iloc[:-1]
        recent    = completed.tail(lookback)
        if len(recent) < 15:
            return {}
        ps  = len(recent) // 3
        acc = recent.iloc[:ps]
        man = recent.iloc[ps: ps * 2]
        dis = recent.iloc[ps * 2:]
        ar  = acc['high'].max() - acc['low'].min()
        if (ar < recent['atr'].mean() * 2 and
                man['low'].min() < acc['low'].min() and
                dis['close'].iloc[-1] > dis['close'].iloc[0] and
                dis['close'].iloc[-1] > man['high'].max()):
            return {'type': 'bullish_po3', 'quality_boost': 0.4}
        if (ar < recent['atr'].mean() * 2 and
                man['high'].max() > acc['high'].max() and
                dis['close'].iloc[-1] < dis['close'].iloc[0] and
                dis['close'].iloc[-1] < man['low'].min()):
            return {'type': 'bearish_po3', 'quality_boost': 0.4}
        return {}

    # ── Inducement ────────────────────────────────────────────

    @staticmethod
    def detect_inducement(df: pd.DataFrame, lookback: int = 15) -> Dict:
        if len(df) < 9:
            return {}
        completed = df.iloc[:-1]
        recent    = completed.tail(lookback)
        if len(recent) < 8:
            return {}
        cp      = recent['close'].iloc[-1]
        older   = recent.iloc[:-4]
        recent4 = recent.iloc[-4:]
        if len(older) < 3 or len(recent4) < 2:
            return {}
        oh = older['high'].max()
        if recent4['high'].max() > oh and cp < oh * 0.998:
            return {'type': 'bullish_inducement', 'level': oh, 'quality_boost': 0.3}
        ol = older['low'].min()
        if recent4['low'].min() < ol and cp > ol * 1.002:
            return {'type': 'bearish_inducement', 'level': ol, 'quality_boost': 0.3}
        return {}

    # ── Judas Swing ───────────────────────────────────────────

    @staticmethod
    def detect_judas_swing(df: pd.DataFrame, is_silver_bullet: bool = False) -> Dict:
        if not is_silver_bullet or not Config.JUDAS_SWING_ENABLED or len(df) < 21:
            return {}
        completed = df.iloc[:-1]
        recent    = completed.tail(20)
        if len(recent) < 15:
            return {}
        cp   = recent['close'].iloc[-1]
        sess = recent.iloc[:-5]
        brk  = recent.iloc[-5:]
        if len(sess) < 5 or len(brk) < 3:
            return {}
        sh = sess['high'].max()
        sl = sess['low'].min()
        if brk['low'].min() < sl * 0.9995 and cp > sl * 1.0003:
            return {'type': 'bullish_judas', 'quality_boost': Config.JUDAS_SWING_BONUS}
        if brk['high'].max() > sh * 1.0005 and cp < sh * 0.9997:
            return {'type': 'bearish_judas', 'quality_boost': Config.JUDAS_SWING_BONUS}
        return {}

    # ── OTE ───────────────────────────────────────────────────

    @staticmethod
    def calculate_ote(df: pd.DataFrame, zone: Dict) -> float:
        if not zone:
            return df['close'].iloc[-2] if len(df) >= 2 else df['close'].iloc[-1]
        fib_prices = [
            zone['low'] + (zone['high'] - zone['low']) * lvl
            for lvl in Config.FIB_LEVELS
        ]
        return sum(fib_prices) / len(fib_prices)
