"""
core/performance_monitor.py — Win-rate and drawdown watcher.
Ported from ozy.py / PerformanceMonitor unchanged.
"""

import logging
from typing import List

import numpy as np

from config import Config


class PerformanceMonitor:

    def __init__(self):
        self.win_rates: List[float]  = []
        self.drawdowns: List[float]  = []
        self.alert_threshold         = Config.PERFORMANCE_ALERT_DRAWDOWN

    def check_performance(self, stats_tracker) -> bool:
        if Config.BACKTEST_MODE:
            return True

        stats = stats_tracker.get_stats()
        self.win_rates.append(stats['win_rate'])
        self.drawdowns.append(stats['max_drawdown'])

        if len(self.win_rates) > 10:
            recent_avg  = np.mean(self.win_rates[-5:])
            overall_avg = np.mean(self.win_rates)
            if recent_avg < overall_avg * 0.7:
                logging.warning(
                    f"[PERFORMANCE_DEGRADATION] Recent WR {recent_avg:.1%} < "
                    f"70% of overall {overall_avg:.1%}"
                )
                return False

        if stats['max_drawdown'] > self.alert_threshold:
            logging.critical(f"[MAX_DRAWDOWN] {stats['max_drawdown']:.1%}")
            return False

        return True
