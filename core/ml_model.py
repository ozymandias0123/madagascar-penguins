"""
core/ml_model.py — Online XGBoost model with persistent trade memory.
Ported from ozy.py / PersistentMLModel with all logic intact.
"""

import gc
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import Config
from core.market_structure import MarketStructureDetector


# ── Custom scorer ─────────────────────────────────────────────

def trading_profit_scorer(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    total_profit = (tp * 2.0) - (fp * 1.0)
    total_loss   = fn * 2.0
    profit_factor = (total_profit / (total_loss + 1e-10)) if total_loss > 0 else (total_profit if total_profit > 0 else 0)
    return min(0.9, max(0.0, profit_factor / 3.0))


class PersistentMLModel:

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            gamma=0.2, reg_alpha=0.1, reg_lambda=2.0,
            scale_pos_weight=1.5, random_state=42, n_jobs=-1,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
        )
        self.scaler              = StandardScaler()
        self.trade_history: List[Tuple] = []
        self.is_trained          = False
        self.feature_importance: Dict   = {}
        self.selected_features: Optional[List[str]] = None
        self.min_training_samples = Config.MIN_TRAINING_SAMPLES
        self.performance_history: List[Dict] = []
        self.use_cv              = True

        prefix = Config.get_file_prefix()
        self.save_file   = f'{prefix}_ml_training.pkl'
        self.backup_file = f'{prefix}_ml_training_backup.pkl'
        self.scaler_file = 'ml_scaler_locked.pkl'
        self.model_file  = 'ml_model_pretrained.pkl'

        self.load_training_data()
        self.load_locked_scaler_and_model()

        if self.selected_features is None:
            self.selected_features = Config.LOCKED_FEATURE_LIST.copy()
            logging.info(f"[LOCKED_FEATURES] 🔒 {len(self.selected_features)} features")

        logging.info(
            f"[ML_MODEL] Initialised — {len(self.trade_history)} samples "
            f"(mode={Config.get_file_prefix()}, weight={Config.get_mode_weight()})"
        )

    # ── Persistence ───────────────────────────────────────────

    def load_training_data(self):
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        all_samples = []
        for mode_prefix, weight in [('backtest', 1.0), ('demo', 2.0), ('live', 3.0)]:
            filepath = os.path.join(os.path.dirname(script_dir),
                                    f'{mode_prefix}_ml_training.pkl')
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as fh:
                        data = pickle.load(fh)
                    if isinstance(data, list) and data:
                        for sample in data:
                            if len(sample) == 3:
                                all_samples.append((*sample, weight))
                            elif len(sample) == 4:
                                all_samples.append(sample)
                        logging.info(f"[ML_LOAD] {mode_prefix}: {len(data)} samples (w={weight})")
                except Exception as exc:
                    logging.error(f"[ML_LOAD_ERROR] {mode_prefix}: {exc}")

        if all_samples:
            self.trade_history = all_samples
            logging.info(f"[ML_COMBINED] Total: {len(self.trade_history)} samples")
        else:
            logging.info("[ML_LOAD] No previous data — starting fresh")

    def load_locked_scaler_and_model(self):
        try:
            if os.path.exists(self.scaler_file):
                with open(self.scaler_file, 'rb') as fh:
                    saved = pickle.load(fh)
                    self.scaler            = saved['scaler']
                    self.selected_features = saved['features']
                    logging.info(f"[SCALER_LOADED] 🔒 {len(self.selected_features)} features")
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as fh:
                    self.model     = pickle.load(fh)
                    self.is_trained = True
                    logging.info("[MODEL_LOADED] 🚀 Pre-trained model loaded")
        except Exception as exc:
            logging.warning(f"[LOCK_LOAD_ERROR] {exc} — will create new")

    def save_locked_scaler_and_model(self):
        try:
            with open(self.scaler_file, 'wb') as fh:
                pickle.dump({'scaler': self.scaler, 'features': self.selected_features}, fh)
            with open(self.model_file, 'wb') as fh:
                pickle.dump(self.model, fh)
            logging.info("[LOCK_SAVED] 🔒 Scaler + model saved")
        except Exception as exc:
            logging.error(f"[LOCK_SAVE_ERROR] {exc}")

    def save_training_data(self):
        try:
            current_weight   = Config.get_mode_weight()
            current_samples  = [t for t in self.trade_history
                                 if len(t) == 4 and t[3] == current_weight]
            if os.path.exists(self.save_file):
                os.replace(self.save_file, self.backup_file)
            with open(self.save_file, 'wb') as fh:
                pickle.dump(current_samples, fh)
            logging.info(f"[SAVE_SUCCESS] {len(current_samples)} {Config.get_file_prefix()} samples")
            if self.is_trained:
                self.save_locked_scaler_and_model()
        except Exception as exc:
            logging.error(f"[SAVE_ERROR] {exc}")

    def auto_save_check(self, trade_count: int):
        if trade_count % Config.SAVE_INTERVAL_TRADES == 0:
            self.save_training_data()

    # ── Feature Engineering ───────────────────────────────────

    def prepare_features(self, df: pd.DataFrame, session: str,
                         current_price: float, zone: Dict) -> Dict:
        zone = zone or {}
        if len(df) < 3:
            return {}
        idx = len(df) - 2  # Last completed bar

        atr_current = df['atr'].iloc[idx] if len(df) > 1 else 0
        atr_ma5     = df['atr'].iloc[max(0, idx - 4):idx + 1].mean() if idx >= 4 else atr_current
        atr_ma20    = df['atr'].iloc[max(0, idx - 19):idx + 1].mean() if idx >= 19 else atr_current

        try:
            ote_distance = (
                abs(current_price - MarketStructureDetector.calculate_ote(df, zone)) / atr_current
                if zone and atr_current > 0 else 0
            )
        except Exception:
            ote_distance = 0

        all_features = {
            'atr':            atr_current,
            'volatility_ratio': atr_current / atr_ma20 if atr_ma20 > 0 else 1.0,
            'bos_bullish':    1 if MarketStructureDetector.detect_market_structure(df) == 'bullish_bos' else 0,
            'bos_bearish':    1 if MarketStructureDetector.detect_market_structure(df) == 'bearish_bos' else 0,
            'rsi':            df['rsi'].iloc[idx] if idx >= 0 else 50,
            'rsi_change':     df['rsi'].iloc[idx] - df['rsi'].iloc[max(0, idx - 5)] if idx >= 5 else 0,
            'volume_ratio':   (
                df['tick_volume'].iloc[idx] /
                df['tick_volume'].iloc[max(0, idx - 19):idx + 1].mean()
                if idx >= 19 else 1.0
            ),
            'price_momentum': (
                (df['close'].iloc[idx] - df['close'].iloc[max(0, idx - 5)]) /
                df['close'].iloc[max(0, idx - 5)]
                if idx >= 5 and df['close'].iloc[max(0, idx - 5)] > 0 else 0
            ),
            'ema_distance':   (
                (df['close'].iloc[idx] - df['ema_20'].iloc[idx]) / df['ema_20'].iloc[idx]
                if 'ema_20' in df.columns and df['ema_20'].iloc[idx] > 0 else 0
            ),
            'ema_slope':      (
                (df['ema_20'].iloc[idx] - df['ema_20'].iloc[max(0, idx - 5)]) /
                df['ema_20'].iloc[max(0, idx - 5)]
                if 'ema_20' in df.columns and idx >= 5 and df['ema_20'].iloc[max(0, idx - 5)] > 0 else 0
            ),
            'is_silver_bullet': 1 if idx >= 0 and self._is_silver_bullet_time(df.index[idx].hour) else 0,
            'hour_of_day':    df.index[idx].hour if idx >= 0 else 0,
            'candle_body_ratio': (
                abs(df['close'].iloc[idx] - df['open'].iloc[idx]) /
                (df['high'].iloc[idx] - df['low'].iloc[idx])
                if idx >= 0 and (df['high'].iloc[idx] - df['low'].iloc[idx]) > 0 else 0
            ),
            'ote_distance':   ote_distance,
            # New 6 features (boti9)
            'adx':            float(df['adx_indicator'].iloc[idx])
                              if 'adx_indicator' in df.columns and idx >= 0
                              and not pd.isna(df['adx_indicator'].iloc[idx]) else 20.0,
            'atr_trend':      (
                (atr_current - df['atr'].iloc[max(0, idx - 5)]) /
                df['atr'].iloc[max(0, idx - 5)]
                if idx >= 5 and df['atr'].iloc[max(0, idx - 5)] > 0 else 0.0
            ),
            'prev_day_direction': (
                1.0 if idx >= 4 and df['close'].iloc[idx] > df['open'].iloc[max(0, idx - 4)]
                else -1.0
            ),
            'price_vs_ema50': (
                (df['close'].iloc[idx] - df['ema_50'].iloc[idx]) / df['ema_50'].iloc[idx]
                if 'ema_50' in df.columns and idx >= 0 and df['ema_50'].iloc[idx] > 0 else 0.0
            ),
            'price_vs_ema200': (
                (df['close'].iloc[idx] - df['ema_200'].iloc[idx]) / df['ema_200'].iloc[idx]
                if 'ema_200' in df.columns and idx >= 0 and df['ema_200'].iloc[idx] > 0 else 0.0
            ),
            'session_phase':  (
                0.0 if idx >= 0 and df.index[idx].hour in [8, 9]    else
                1.0 if idx >= 0 and df.index[idx].hour in [13, 14]  else
                2.0 if idx >= 0 and df.index[idx].hour in [15, 16]  else
                3.0 if idx >= 0 and df.index[idx].hour in [19, 20]  else
                4.0
            ),
        }

        return {k: all_features.get(k, 0.0) for k in Config.LOCKED_FEATURE_LIST}

    def _is_silver_bullet_time(self, hour: int) -> bool:
        return any(s <= hour < e for _, (s, e) in Config.SILVER_BULLET_WINDOWS.items())

    # ── Update / Train ────────────────────────────────────────

    def update(self, df: pd.DataFrame, session: str, current_price: float,
               zone: Dict, trade_outcome: Optional[int] = None,
               profit_r: float = 0.0) -> float:
        if not Config.ML_ENABLED:
            return 0.5

        zone     = zone or {}
        features = self.prepare_features(df, session, current_price, zone)

        if trade_outcome is not None:
            strict   = (1 if profit_r >= Config.PROFIT_R_THRESHOLD else 0) \
                       if Config.USE_PROFIT_THRESHOLD else trade_outcome
            weight   = Config.get_mode_weight()
            self.trade_history.append((features, strict, profit_r, weight))
            self.auto_save_check(len(self.trade_history))
            self._trim_history()

            if len(self.trade_history) >= self.min_training_samples:
                self._train()

        if not self.is_trained:
            return 0.5

        X       = pd.DataFrame([features])
        X       = X[self.selected_features]
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        return float(self.model.predict_proba(X_scaled)[0][1])

    def update_from_features(self, features: Dict, session: str,
                             trade_outcome: int, profit_r: float):
        """Used in live mode when position closes."""
        if not Config.ML_ENABLED or features is None:
            return
        strict = (1 if profit_r >= Config.PROFIT_R_THRESHOLD else 0) \
                 if Config.USE_PROFIT_THRESHOLD else trade_outcome
        self.trade_history.append((features, strict, profit_r, Config.get_mode_weight()))
        self.auto_save_check(len(self.trade_history))
        self._trim_history()
        if len(self.trade_history) >= self.min_training_samples:
            self._fast_retrain()

    def optimize_parameters(self, df: pd.DataFrame, session: str,
                            win_rate: float, current_price: float, zone: Dict):
        zone     = zone or {}
        features = self.prepare_features(df, session, current_price, zone)
        X = pd.DataFrame([features])
        if self.is_trained:
            X       = X[self.selected_features]
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            confidence = self.model.predict_proba(X_scaled)[0][1]
            if win_rate < 0.4:
                Config.FVG_SIZE_THRESHOLD  = min(0.0001, Config.FVG_SIZE_THRESHOLD + Config.LEARNING_RATE * 0.1)
                Config.SL_ATR_MULTIPLIER   = max(2.5,   Config.SL_ATR_MULTIPLIER  + Config.LEARNING_RATE * 0.05)
                Config.CONFIDENCE_THRESHOLD = max(0.2,  Config.CONFIDENCE_THRESHOLD - Config.LEARNING_RATE * 0.05)
            elif confidence > 0.7:
                Config.FVG_SIZE_THRESHOLD  = max(0.00003, Config.FVG_SIZE_THRESHOLD - Config.LEARNING_RATE * 0.05)

    # ── Private training helpers ──────────────────────────────

    def _trim_history(self):
        if len(self.trade_history) > 2000:
            wins   = [t for t in self.trade_history if t[1] == 1]
            losses = [t for t in self.trade_history if t[1] == 0]
            self.trade_history = (wins[-1000:] + losses[-1000:])[-2000:]

    def _train(self):
        X_train   = pd.DataFrame([t[0] for t in self.trade_history])
        y_train   = [t[1] for t in self.trade_history]
        profits_r = [t[2] for t in self.trade_history]

        if len(np.unique(y_train)) < 2:
            return
        try:
            avail = [f for f in self.selected_features if f in X_train.columns]
            X_sel = X_train[avail]

            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                X_scaled = self.scaler.fit_transform(X_sel)
            else:
                X_scaled = self.scaler.transform(X_sel)
            X_scaled = pd.DataFrame(X_scaled, columns=avail)

            X_tr, X_val, y_tr, y_val, sw_tr, _ = train_test_split(
                X_scaled, y_train,
                [t[3] for t in self.trade_history],
                test_size=0.2, random_state=42, stratify=y_train
            )
            sw_tr = np.array(sw_tr)
            sw_tr = sw_tr / sw_tr.sum() * len(sw_tr)

            # CV score
            cv_mean, cv_std = 0.5, 0.0
            if len(self.trade_history) > 150:
                try:
                    scorer   = make_scorer(trading_profit_scorer, greater_is_better=True)
                    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_model = XGBClassifier(n_estimators=80, max_depth=4,
                                             learning_rate=0.05, random_state=77)
                    scores   = cross_val_score(cv_model, X_scaled.values, y_train,
                                               cv=cv, scoring=scorer, n_jobs=-1)
                    cv_mean, cv_std = scores.mean(), scores.std()
                    del cv_model
                    gc.collect()
                    logging.info(f"[CV_SCORES] CV={cv_mean:.3f}±{cv_std:.3f}")
                except Exception as cv_err:
                    logging.warning(f"[CV_ERROR] {cv_err}")

            self.model = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                gamma=0.2, reg_alpha=0.1, reg_lambda=2.0,
                scale_pos_weight=1.5, random_state=42, n_jobs=-1,
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
            )
            self.model.fit(X_tr, y_tr, sample_weight=sw_tr,
                           eval_set=[(X_val, y_val)], verbose=False)

            self.is_trained          = True
            self.feature_importance  = dict(zip(avail, self.model.feature_importances_))
            self.performance_history.append({
                'cv_mean': cv_mean, 'cv_std': cv_std,
                'n_samples': len(y_train),
                'win_rate':  sum(y_train) / len(y_train),
                'avg_profit_r': np.mean(profits_r),
            })
            self.save_locked_scaler_and_model()
            logging.info(
                f"[MODEL_UPDATE] ✅ CV={cv_mean:.3f}±{cv_std:.3f}, "
                f"N={len(y_train)}, WR={sum(y_train)/len(y_train):.2%}"
            )
        except Exception as exc:
            logging.error(f"[TRAINING_ERROR] {exc}")
            self._fallback_train()

    def _fallback_train(self):
        try:
            X_train = pd.DataFrame([t[0] for t in self.trade_history])
            y_train = [t[1] for t in self.trade_history]
            avail   = [f for f in (self.selected_features or Config.LOCKED_FEATURE_LIST)
                       if f in X_train.columns]
            X_sel   = X_train[avail]
            self.scaler  = StandardScaler()
            X_scaled     = pd.DataFrame(self.scaler.fit_transform(X_sel), columns=avail)
            self.model   = XGBClassifier(n_estimators=100, max_depth=4,
                                          learning_rate=0.05, random_state=42)
            self.model.fit(X_scaled, y_train)
            self.is_trained = True
            logging.info("[FALLBACK_TRAIN] ⚠️ Fallback training succeeded")
        except Exception as fe:
            logging.error(f"[FALLBACK_ERROR] {fe}")

    def _fast_retrain(self):
        try:
            X_train = pd.DataFrame([t[0] for t in self.trade_history])
            y_train = [t[1] for t in self.trade_history]
            if len(np.unique(y_train)) < 2:
                return
            avail   = [f for f in (self.selected_features or Config.LOCKED_FEATURE_LIST)
                       if f in X_train.columns]
            X_sel   = X_train[avail]
            if not hasattr(self.scaler, 'mean_'):
                X_scaled = pd.DataFrame(self.scaler.fit_transform(X_sel), columns=avail)
            else:
                X_scaled = pd.DataFrame(self.scaler.transform(X_sel), columns=avail)
            sw = np.array([t[3] for t in self.trade_history])
            sw = sw / sw.sum() * len(sw)
            self.model = XGBClassifier(n_estimators=100, max_depth=4,
                                        learning_rate=0.05, random_state=42, n_jobs=-1)
            self.model.fit(X_scaled, y_train, sample_weight=sw)
            self.is_trained = True
            self.save_locked_scaler_and_model()
            logging.info(f"[LIVE_ML_RETRAIN] ✅ {len(y_train)} samples, "
                         f"WR={sum(y_train)/len(y_train):.2%}")
        except Exception as exc:
            logging.error(f"[LIVE_TRAIN_ERROR] {exc}")
