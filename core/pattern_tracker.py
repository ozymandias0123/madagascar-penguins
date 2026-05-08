"""
core/pattern_tracker.py — ICT pattern performance tracking + adaptive weights.
Ported from ozy.py / PatternPerformanceTracker unchanged.
"""

import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from typing import Dict

from config import Config


class PatternPerformanceTracker:

    def __init__(self):
        self.pattern_weights             = Config.PATTERN_WEIGHTS.copy()
        self.pattern_stats: Dict         = {}
        self.last_update_trade_count     = 0

        prefix                  = Config.get_file_prefix()
        self.save_file          = f'{prefix}_pattern_data.pkl'
        self.backup_file        = f'{prefix}_pattern_data_backup.pkl'
        self.json_report_file   = f'{prefix}_pattern_report.json'

        self.load_data()
        logging.info(
            f"[PATTERN_TRACKER] 📊 Initialised with {len(self.pattern_stats)} patterns "
            f"(mode={prefix})"
        )

    # ── Persistence ───────────────────────────────────────────

    def load_data(self):
        try:
            for f in [self.save_file, self.backup_file]:
                if os.path.exists(f):
                    with open(f, 'rb') as fh:
                        data = pickle.load(fh)
                    self.pattern_weights          = data.get('weights', Config.PATTERN_WEIGHTS.copy())
                    self.pattern_stats            = data.get('stats', {})
                    self.last_update_trade_count  = data.get('last_update', 0)
                    logging.info(f"[PATTERN_LOAD] ✅ Loaded from {f}")
                    return
            logging.info("[PATTERN_LOAD] 🆕 Starting fresh")
        except Exception as exc:
            logging.error(f"[PATTERN_LOAD] ❌ {exc}")
            self.pattern_weights = Config.PATTERN_WEIGHTS.copy()
            self.pattern_stats   = {}

    def save_data(self):
        try:
            if os.path.exists(self.save_file):
                shutil.copy2(self.save_file, self.backup_file)
            data = {
                'weights':     self.pattern_weights,
                'stats':       self.pattern_stats,
                'last_update': self.last_update_trade_count,
                'timestamp':   datetime.now().isoformat()
            }
            with open(self.save_file, 'wb') as fh:
                pickle.dump(data, fh)
            self._save_json_report()
        except Exception as exc:
            logging.error(f"[PATTERN_SAVE] ❌ {exc}")

    def _save_json_report(self):
        try:
            report = {
                'timestamp':      datetime.now().isoformat(),
                'total_patterns': len(self.pattern_stats),
                'patterns':       {}
            }
            for pattern, stats in self.pattern_stats.items():
                recent = stats['trades'][-Config.PATTERN_ANALYSIS_WINDOW:]
                if len(recent) >= Config.MIN_PATTERN_SAMPLES:
                    wins      = sum(1 for t in recent if t['pnl'] > 0)
                    total_pnl = sum(t['pnl'] for t in recent)
                    report['patterns'][pattern] = {
                        'total_trades':   len(stats['trades']),
                        'recent_trades':  len(recent),
                        'win_rate':       wins / len(recent),
                        'avg_pnl':        total_pnl / len(recent),
                        'total_pnl':      stats['total_pnl'],
                        'current_weight': self.pattern_weights.get(pattern, 1.0)
                    }
            with open(self.json_report_file, 'w', encoding='utf-8') as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logging.error(f"[JSON_REPORT] ❌ {exc}")

    # ── Recording ─────────────────────────────────────────────

    def record_trade(self, pattern_key: str, pnl: float, trade_data: Dict):
        if pattern_key not in self.pattern_stats:
            self.pattern_stats[pattern_key] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []}
        s = self.pattern_stats[pattern_key]
        s['trades'].append({'pnl': pnl, 'data': trade_data})
        s['total_pnl'] += pnl
        if pnl > 0: s['wins']   += 1
        else:        s['losses'] += 1
        total = sum(len(v['trades']) for v in self.pattern_stats.values())
        if total % 5 == 0:
            self.save_data()

    # ── Pattern Key ───────────────────────────────────────────

    def get_pattern_key(self, signal: Dict, context: Dict,
                        is_silver_bullet: bool, htf_aligned: bool,
                        judas_swing: Dict) -> str:
        c = []
        if is_silver_bullet: c.append('sb')
        st = signal.get('type', '')
        if 'bos'   in st: c.append('bos')
        elif 'choch' in st: c.append('choch')
        if signal.get('zone', {}).get('type') == 'fvg': c.append('fvg')
        if context.get('volatility') == 'high':          c.append('high_vol')
        if context.get('regime')     == 'trending':      c.append('trending')
        if htf_aligned:                                   c.append('htf_aligned')
        if judas_swing and judas_swing.get('detected'):   c.append('judas')
        if signal.get('inducement_confirmed'):            c.append('inducement')
        return '+'.join(sorted(c)) if c else 'unknown'

    # ── Weight Adaptation ─────────────────────────────────────

    def update_weights(self, total_trade_count: int) -> bool:
        if total_trade_count - self.last_update_trade_count < Config.PATTERN_UPDATE_INTERVAL:
            return False
        self.last_update_trade_count = total_trade_count
        for pattern, stats in self.pattern_stats.items():
            recent = stats['trades'][-Config.PATTERN_ANALYSIS_WINDOW:]
            if len(recent) < Config.MIN_PATTERN_SAMPLES:
                continue
            wins = sum(1 for t in recent if t['pnl'] > 0)
            wr   = wins / len(recent)
            avg  = sum(t['pnl'] for t in recent) / len(recent)
            w    = self.pattern_weights.get(pattern, 1.0)
            if wr > 0.50 and avg > 0:
                w *= Config.PATTERN_WEIGHT_BOOST
            elif wr < 0.40 or avg < -2.0:
                w *= Config.PATTERN_WEIGHT_DECAY
            self.pattern_weights[pattern] = max(0.3, min(2.0, w))
        self.save_data()
        return True

    def get_weight(self, pattern_key: str) -> float:
        return self.pattern_weights.get(pattern_key, 1.0)

    def get_stats_summary(self) -> str:
        lines = ["\n" + "=" * 80, "📊 PATTERN PERFORMANCE SUMMARY", "=" * 80]
        if not self.pattern_stats:
            lines.append("⏸️  No data yet")
            return "\n".join(lines)
        sorted_p = sorted(
            self.pattern_stats.items(),
            key=lambda x: sum(t['pnl'] for t in x[1]['trades'][-Config.PATTERN_ANALYSIS_WINDOW:]),
            reverse=True
        )
        lines.append(f"{'Pattern':<35} | {'Trades':>7} | {'WR':>6} | {'AvgPnL':>9} | {'Weight':>7}")
        lines.append("-" * 80)
        for pattern, stats in sorted_p[:15]:
            recent = stats['trades'][-Config.PATTERN_ANALYSIS_WINDOW:]
            if len(recent) < Config.MIN_PATTERN_SAMPLES:
                continue
            wins = sum(1 for t in recent if t['pnl'] > 0)
            wr   = wins / len(recent)
            avg  = sum(t['pnl'] for t in recent) / len(recent)
            w    = self.pattern_weights.get(pattern, 1.0)
            lines.append(f"{pattern:<35} | {len(recent):>7} | {wr:>5.1%} | ${avg:>7.2f} | {w:>6.2f}")
        lines.append("=" * 80)
        return "\n".join(lines)
