"""
core/ml_health.py — ML model health monitor with auto-freeze.
Ported from ozy.py / MLHealthMonitor unchanged.
"""

import logging
from typing import List, Optional

import pandas as pd

from config import Config


class MLHealthMonitor:

    def __init__(self):
        self.recent_predictions: List[dict] = []
        self.is_frozen              = False
        self.freeze_time: Optional  = None
        self.bad_prediction_streak  = 0
        logging.info("[ML_HEALTH] 🧠 Initialised")

    def record_prediction(self,
                          was_correct: Optional[bool] = None,
                          confidence: float = 0.5,
                          predicted_win: Optional[bool] = None,
                          actual_win: Optional[bool] = None):
        if not Config.ML_HEALTH_CHECK_ENABLED:
            return

        if was_correct is None:
            if predicted_win is not None and actual_win is not None:
                was_correct = (predicted_win == actual_win)
            else:
                was_correct = False

        self.recent_predictions.append({
            'correct':    was_correct,
            'confidence': confidence,
            'timestamp':  pd.Timestamp.now(tz='UTC')
        })
        if len(self.recent_predictions) > Config.ML_LOOKBACK_TRADES * 2:
            self.recent_predictions = self.recent_predictions[-Config.ML_LOOKBACK_TRADES:]

        self.bad_prediction_streak = (
            0 if was_correct else self.bad_prediction_streak + 1
        )
        self._check_freeze_conditions()

    def _check_freeze_conditions(self):
        if self.is_frozen:
            return
        if len(self.recent_predictions) >= Config.ML_LOOKBACK_TRADES:
            recent   = self.recent_predictions[-Config.ML_LOOKBACK_TRADES:]
            accuracy = sum(1 for p in recent if p['correct']) / len(recent)
            if accuracy < Config.ML_MIN_WIN_RATE_THRESHOLD:
                self._freeze(f"LOW_ACCURACY ({accuracy:.1%})")
                return
        if self.bad_prediction_streak >= Config.ML_FREEZE_AFTER_BAD_STREAK:
            self._freeze(f"BAD_STREAK ({self.bad_prediction_streak})")

    def _freeze(self, reason: str):
        self.is_frozen   = True
        self.freeze_time = pd.Timestamp.now(tz='UTC')
        logging.warning(f"[ML_HEALTH] ⚠️ ML FROZEN: {reason}")

    def is_ml_healthy(self) -> bool:
        if not Config.ML_HEALTH_CHECK_ENABLED or not self.is_frozen:
            return True
        if self.freeze_time:
            elapsed = (pd.Timestamp.now(tz='UTC') - self.freeze_time).total_seconds() / 3600
            if elapsed >= Config.ML_UNFREEZE_AFTER_HOURS:
                self.is_frozen              = False
                self.freeze_time            = None
                self.bad_prediction_streak  = 0
                self.recent_predictions     = []
                logging.info("[ML_HEALTH] ✅ ML unfrozen after cooldown")
                return True
        return False

    def get_confidence_multiplier(self) -> float:
        if not self.is_ml_healthy():
            return 0.5
        if len(self.recent_predictions) < 10:
            return 1.0
        accuracy = sum(1 for p in self.recent_predictions[-10:] if p['correct']) / 10
        return 1.2 if accuracy >= 0.6 else 1.0 if accuracy >= 0.4 else 0.7

    def force_unfreeze(self):
        self.is_frozen              = False
        self.freeze_time            = None
        self.bad_prediction_streak  = 0
        logging.info("[ML_HEALTH] 🔄 Force unfrozen")
