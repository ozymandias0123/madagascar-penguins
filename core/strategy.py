"""
core/strategy.py — SessionAwareICTStrategy.
Ported from ozy.py unchanged.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from config import Config
from core.market_structure import MarketStructureDetector


class SessionAwareICTStrategy:

    def __init__(self, pattern_tracker=None):
        self.sb_window_start_time: Dict = {}
        self.first_fvg_used: Dict       = {}
        self.pattern_tracker            = pattern_tracker
        self.session_params = {
            'new_york': {
                'min_confidence':      0.08,
                'max_trades':          3,
                'preferred_structures': ['bullish_bos', 'bearish_bos'],
                'kill_zone_start':     13,
                'kill_zone_end':       16,
                'best_hours':          [13, 14, 15],
                'preferred_zone':      'premium'
            }
        }

    # ── Silver Bullet ─────────────────────────────────────────

    def _is_silver_bullet_time(self, hour: int, session: str) -> bool:
        if Config.SILVER_BULLET_MODE == 'off':
            return False
        if session == 'london':
            s, e = Config.SILVER_BULLET_WINDOWS['london']
            return s <= hour < e
        elif session == 'new_york':
            s1, e1 = Config.SILVER_BULLET_WINDOWS['ny_am']
            s2, e2 = Config.SILVER_BULLET_WINDOWS['ny_pm']
            return (s1 <= hour < e1) or (s2 <= hour < e2)
        return False

    # ── Main evaluate ─────────────────────────────────────────

    def evaluate(self,
                 df: pd.DataFrame,
                 session: str,
                 structure: str,
                 fvg_zones: List[Dict],
                 ob: Dict,
                 current_price: float,
                 atr: float,
                 confidence: float,
                 htf_bias: str = 'neutral',
                 htf_structure: Optional[Dict] = None) -> Optional[Dict]:

        if len(df) < 3:
            return None

        session_config = self.session_params.get(session, {})
        current_time   = df.index[-2]
        current_hour   = current_time.hour
        context        = MarketStructureDetector.get_market_context(df)
        is_silver_bullet = self._is_silver_bullet_time(current_hour, session)

        if Config.RANGING_ONLY_SB and context['regime'] == 'ranging' and not is_silver_bullet:
            logging.debug("[RANGING_FILTER] ❌ Ranging market — only SB windows allowed")
            return None

        if is_silver_bullet:
            prev = self.sb_window_start_time.get(session)
            if prev is None or (current_time - prev).total_seconds() > 3600:
                self.sb_window_start_time[session] = current_time
                self.first_fvg_used[session]       = False
            if Config.FIRST_FVG_ONLY and self.first_fvg_used.get(session, False):
                return None

        # HTF alignment
        htf_struct_aligned = False
        if htf_structure:
            htf_type = htf_structure.get('structure', 'neutral')
            if (('bullish' in structure and 'bullish' in htf_type) or
                    ('bearish' in structure and 'bearish' in htf_type)):
                htf_struct_aligned = True

        htf_aligned = (
            (htf_bias == 'bullish' and structure in ['bullish_bos', 'bullish_choch']) or
            (htf_bias == 'bearish' and structure in ['bearish_bos', 'bearish_choch'])
        )

        if Config.SILVER_BULLET_MODE == 'filter' and not is_silver_bullet:
            return None
        if session == 'london':
            if structure not in ['bullish_bos', 'bearish_bos'] or not htf_aligned:
                return None

        in_kill_zone = current_hour in session_config.get('best_hours', [])
        if in_kill_zone:
            confidence *= 1.3

        if confidence < session_config.get('min_confidence', Config.CONFIDENCE_THRESHOLD):
            return None

        zone_type       = MarketStructureDetector.is_premium_discount(df)
        liquidity_sweep = MarketStructureDetector.detect_liquidity_sweep(df)
        preferred_structures = session_config.get('preferred_structures', [])
        if preferred_structures and structure not in preferred_structures:
            return None

        breaker    = MarketStructureDetector.find_breaker_block(df)
        po3        = MarketStructureDetector.detect_power_of_3(df)
        inducement = MarketStructureDetector.detect_inducement(df)
        judas_swing = MarketStructureDetector.detect_judas_swing(df, is_silver_bullet)

        # ── Inner signal builder ──────────────────────────────
        def _build_signal(direction: str, min_quality: float) -> Optional[Dict]:
            q         = 1.0
            is_bull   = direction == 'buy'
            struct_bos   = 'bullish_bos'   if is_bull else 'bearish_bos'
            struct_choch = 'bullish_choch' if is_bull else 'bearish_choch'
            sweep_type   = 'bullish_sweep' if is_bull else 'bearish_sweep'
            breaker_type = 'bullish_breaker' if is_bull else 'bearish_breaker'
            po3_type     = 'bullish_po3'   if is_bull else 'bearish_po3'
            ind_type     = 'bullish_inducement' if is_bull else 'bearish_inducement'
            judas_type   = 'bullish_judas' if is_bull else 'bearish_judas'
            zone_pref    = 'discount'      if is_bull else 'premium'

            if structure == struct_choch:              q -= 0.4
            elif structure == struct_bos:              q += 0.3
            if liquidity_sweep.get('type') == sweep_type: q += 0.5
            if zone_type == zone_pref:                 q += 0.3
            if in_kill_zone:                           q += 0.2
            if breaker.get('type') == breaker_type:   q += 0.6
            if po3.get('type') == po3_type:            q += po3.get('quality_boost', 0.4)
            if inducement.get('type') == ind_type:     q += inducement.get('quality_boost', 0.3)
            if judas_swing.get('type') == judas_type:
                q += judas_swing.get('quality_boost', Config.JUDAS_SWING_BONUS)
                logging.info(f"[JUDAS_SWING] 🎭 {direction.upper()} +{Config.JUDAS_SWING_BONUS}")
            if context['regime'] == 'trending' and structure == struct_bos: q += 0.3
            if context['volatility'] == 'high':                              q += 0.4
            if htf_aligned:                                                   q += 0.5
            if htf_struct_aligned:                                            q += 0.3
            if is_silver_bullet:
                q += Config.SILVER_BULLET_BONUS
                if Config.FIRST_FVG_ONLY:
                    self.first_fvg_used[session] = True

            pattern_key = None
            if self.pattern_tracker:
                temp_sig = {
                    'type': direction,
                    'zone': fvg_zones[-1] if fvg_zones else (breaker if breaker else ob),
                    'inducement_confirmed': inducement.get('type') == ind_type
                }
                pattern_key = self.pattern_tracker.get_pattern_key(
                    temp_sig, context, is_silver_bullet, htf_aligned, judas_swing)
                q *= self.pattern_tracker.get_weight(pattern_key)
                logging.info(
                    f"[PATTERN_WEIGHT] {pattern_key} | "
                    f"w={self.pattern_tracker.get_weight(pattern_key):.2f} | q={q:.2f}"
                )

            if q < min_quality:
                logging.debug(f"[LOW_QUALITY] {q:.1f} < {min_quality} — skipping")
                return None

            zone_used = (
                breaker if breaker.get('type') == breaker_type else
                (fvg_zones[-1]
                 if fvg_zones and fvg_zones[-1]['type'] == ('bullish' if is_bull else 'bearish')
                 else (ob if ob and ob.get('type') == ('bullish_ob' if is_bull else 'bearish_ob')
                       else None))
            )
            if zone_used is None:
                return None

            sig = {
                'type':        direction,
                'entry_price': MarketStructureDetector.calculate_ote(df, zone_used),
                'zone':        zone_used,
                'quality':     q,
            }
            if pattern_key:
                sig['pattern_key'] = pattern_key
            return sig

        if structure in ['bullish_bos', 'bullish_choch']:
            return _build_signal('buy', Config.MIN_QUALITY)
        elif structure in ['bearish_bos', 'bearish_choch']:
            return _build_signal('sell', Config.MIN_QUALITY_BEARISH)
        return None
