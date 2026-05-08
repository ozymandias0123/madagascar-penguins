"""
core/engine.py — PersistentTradingEngine (multi-agent orchestrator edition).

Key changes vs ozy.py:
  • _evaluate_signals() now calls the LangGraph orchestrator when
    ORCHESTRATOR_ENABLED=True and signal quality >= AGENT_MIN_QUALITY.
  • All original classes, persistence, backtest, demo and live logic
    are 100% preserved.
  • RicoExecution agent handles the actual MT5 order placement / simulation.
"""

import logging
import os
import pickle
import shutil
import time
from typing import Dict, List, Optional

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    _MT5_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import Config
from core.circuit_breaker    import CircuitBreaker
from core.data_manager       import DataManager
from core.market_structure   import MarketStructureDetector
from core.ml_health          import MLHealthMonitor
from core.ml_model           import PersistentMLModel
from core.pattern_tracker    import PatternPerformanceTracker
from core.performance_monitor import PerformanceMonitor
from core.risk_manager       import ImprovedRiskManager
from core.stats_tracker      import PersistentStatsTracker
from core.strategy           import SessionAwareICTStrategy


class PersistentTradingEngine:

    def __init__(self, symbol: str, user_email: Optional[str] = None):
        self.symbol               = symbol
        self.user_email           = user_email
        self.stats_tracker        = PersistentStatsTracker(user_email=user_email)
        self.pattern_tracker      = (PatternPerformanceTracker()
                                     if Config.PERFORMANCE_TRACKING_ENABLED else None)
        self.last_candle_time     = None
        self.current_df           = None
        self.online_model         = PersistentMLModel()
        self.strategy             = SessionAwareICTStrategy(pattern_tracker=self.pattern_tracker)
        self.data_manager         = DataManager()
        self.performance_monitor  = PerformanceMonitor()
        self.session_trade_counts = {'london': 0, 'new_york': 0}
        self.used_fvg_zones: set  = set()
        self.open_positions: Dict = {}

        self.circuit_breaker  = (CircuitBreaker(Config.INITIAL_BALANCE)
                                  if Config.CIRCUIT_BREAKER_ENABLED else None)
        self.ml_health_monitor = (MLHealthMonitor()
                                   if Config.ML_HEALTH_CHECK_ENABLED else None)

        # Lazy-load the orchestrator graph (avoids circular import at module level)
        self._graph = None

        logging.info(f"[ENGINE_INIT] symbol={symbol}, mode={Config.get_file_prefix()}")
        logging.info(f"[PREVIOUS_TRADES] {len(self.stats_tracker.trades)}")
        logging.info(f"[TRAINING_SAMPLES] {len(self.online_model.trade_history)}")

        try:
            self._load_open_positions_from_mt5()
        except Exception as exc:
            logging.warning(f"[ENGINE_INIT] Position recovery skipped: {exc}")


    # ── Orchestrator (lazy init) ──────────────────────────────

    def _get_graph(self):
        """Return (and cache) the compiled LangGraph."""
        if self._graph is None:
            from orchestrator.graph import build_graph
            self._graph = build_graph()
            logging.info("[ENGINE] 🧠 LangGraph orchestrator compiled")
        return self._graph

    # ── Session helper ────────────────────────────────────────

    def _get_current_session(self, current_time: pd.Timestamp) -> str:
        if current_time is None:
            return 'new_york'
        try:
            current_time = (current_time.tz_localize('UTC')
                            if current_time.tzinfo is None
                            else current_time.tz_convert('UTC'))
        except Exception:
            pass
        hour = int(current_time.hour)
        for session, start, end in Config.TRADING_SESSIONS:
            if start <= hour < end:
                return session
        return 'new_york'

    # ── Main loop ─────────────────────────────────────────────

    def process_symbol(self, symbol: str) -> bool:
        df = self.data_manager.get_candles(symbol, Config.TIMEFRAME, Config.MIN_CANDLES)
        if df is not None and not df.empty and not df.index.isna().any():
            t = df.index[-1]
            if self.last_candle_time is not None and t <= self.last_candle_time:
                return False
            self.current_df       = df
            self.last_candle_time = t
            logging.info(f"[NEW_CANDLE] {t}, Price={df['close'].iloc[-1]:.2f}")
            self._evaluate_signals()
            self.analyze_trade_performance()
            return True
        return False

    def analyze_trade_performance(self):
        recent = [t for t in self.stats_tracker.trades
                  if pd.Timestamp.now(tz='UTC') -
                  pd.Timestamp(t['time']) < pd.Timedelta(days=7)]
        if not recent:
            return
        df_r = pd.DataFrame(recent)
        for session in ['london', 'new_york']:
            st = df_r[df_r['session'] == session]
            if not st.empty:
                wr = (len([t for t in st.to_dict('records')
                           if t['pnl'] - t['trade_cost'] > 0]) / len(st))
                self.online_model.optimize_parameters(
                    self.current_df, session, wr,
                    self.current_df['close'].iloc[-1], {}
                )

    def run(self):
        # ── Block until Telegram user chooses a mode ──────────
        from core.bot_controller import get_controller as _gc
        _ctrl = _gc()
        if _ctrl.is_paused() and not _ctrl.is_stopped():
            logging.info("[ENGINE] ⏸ Waiting for Telegram command (/demo /live /signals /analysis)...")
            while _ctrl.is_paused() and not _ctrl.is_stopped():
                _ctrl.wait_if_paused(timeout=30.0)
        if _ctrl.is_stopped():
            logging.info("[ENGINE] 🛑 Stopped before start")
            return

        # ── Sync mode from BotController (set by Telegram) ────
        _chosen = _ctrl.get_mode()
        Config.MODE          = _chosen
        Config.PAPER_TRADING = (_chosen == 'demo')
        Config.LIVE_MODE     = (_chosen == 'live')
        Config.BACKTEST_MODE = (_chosen == 'backtest')

        # ── Connect to exchange NOW (after user chose mode) ────
        _active_exchange = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5').lower()
        if _active_exchange == 'mt5':
            logging.info("[ENGINE] Connecting to MetaTrader 5...")
            try:
                from utils.mt5_manager import MT5Manager
                _new_sym = MT5Manager.initialize()
                if _new_sym and _new_sym != self.symbol:
                    self.symbol = _new_sym
                    logging.info(f"[ENGINE] MT5 connected — symbol={self.symbol}")
                # Sync real balance
                _info = mt5.account_info()
                if _info:
                    self.stats_tracker.balance    = _info.balance
                    Config.INITIAL_BALANCE        = _info.balance
                    logging.info(f"[ENGINE] Balance synced: ${_info.balance:.2f}")
            except Exception as _exc:
                logging.error(f"[ENGINE] MT5 connection failed: {_exc}")
                try:
                    from utils.telegram_notifier import get_notifier as _gn
                    _gn().send(f"⚠️ <b>MT5 connection failed</b>\n<code>{str(_exc)[:200]}</code>")
                except Exception:
                    pass
        else:
            logging.info(f"[ENGINE] Connecting to {_active_exchange.upper()} via ccxt...")
            try:
                from core.exchange import get_exchange_from_config
                _ex = get_exchange_from_config()
                if _ex.connect():
                    _bal = _ex.get_balance()
                    Config.INITIAL_BALANCE     = _bal
                    self.stats_tracker.balance = _bal
                    logging.info(f"[ENGINE] {_active_exchange.upper()} connected — balance=${_bal:.2f}")
                    _ex.disconnect()
                else:
                    logging.error(f"[ENGINE] {_active_exchange.upper()} connection refused — check API keys")
            except Exception as _exc:
                logging.error(f"[ENGINE] Exchange init failed: {_exc}")

        mode = Config.get_file_prefix().upper()
        logging.info(
            f"[BOT_START] Mode={mode}, "
            f"Trades={len(self.stats_tracker.trades)}, "
            f"ML={len(self.online_model.trade_history)}, "
            f"Orchestrator={'ON' if Config.ORCHESTRATOR_ENABLED else 'OFF'}"
        )

        if Config.BACKTEST_MODE:
            self.run_persistent_backtest()
            self._finalize()
            return

        if Config.PAPER_TRADING or _chosen in ('signals', 'analysis'):
            self.run_persistent_demo()
            self._finalize()
            return

        # ── Live trading ──────────────────────────────────────
        end_date = pd.Timestamp.now(tz='UTC')
        init_df  = self.data_manager.get_candles(
            self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES,
            (end_date - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        if init_df is not None and not init_df.empty:
            self.last_candle_time = init_df.index[-1]

        while True:
            try:
                # ── wait if Telegram sent /pause ──────────────
                _ctrl = _gc()
                while _ctrl.is_paused() and not _ctrl.is_stopped():
                    _ctrl.wait_if_paused(timeout=30.0)
                if _ctrl.is_stopped():
                    break
                if not mt5.initialize():
                    from utils.mt5_manager import MT5Manager
                    MT5Manager.initialize()
                self._monitor_open_positions()
                self.process_symbol(self.symbol)
                time.sleep(10)
            except KeyboardInterrupt:
                logging.info("[SHUTDOWN] Keyboard interrupt")
                self.shutdown()
                break
            except SystemExit as se:
                self.shutdown()
                raise
            except Exception as exc:
                logging.error(f"[MAIN_LOOP_ERROR] {exc}")
                time.sleep(60)

    # ── Signal Evaluation ─────────────────────────────────────

    def _evaluate_signals(self):
        df = self.current_df
        if df is None or df.empty or len(df) < 3:
            return

        if (self.stats_tracker.trades and
                self.stats_tracker.trades[-1]['time'] == self.last_candle_time.isoformat()):
            return

        current_time = (self.last_candle_time
                        if (Config.BACKTEST_MODE or Config.PAPER_TRADING)
                        else pd.Timestamp.now(tz='UTC'))
        if current_time is None:
            return

        session = self._get_current_session(current_time)
        if self.session_trade_counts[session] >= Config.SESSION_MAX_TRADES[session]:
            return

        current_price = df['close'].iloc[-2]
        atr           = df['atr'].iloc[-2]
        structure     = MarketStructureDetector.detect_market_structure(df)
        fvg_zones     = MarketStructureDetector.find_fvg(df, atr)
        ob            = MarketStructureDetector.find_order_block(df)
        breaker       = MarketStructureDetector.find_breaker_block(df)
        zone          = breaker or ob or (fvg_zones[-1] if fvg_zones else {})
        confidence    = self.online_model.update(df, session, current_price, zone)

        # Volatility filter
        atr_mean = df['atr'].iloc[:-1].mean() if len(df) > 2 else atr
        if Config.VOLATILITY_FILTER and atr > atr_mean * Config.ATR_VOLATILITY_THRESHOLD:
            logging.warning(f"[HIGH_VOLATILITY] ATR={atr:.2f} — skipping")
            return

        # ADX filter
        context    = MarketStructureDetector.get_market_context(df)
        is_sb_time = self.strategy._is_silver_bullet_time(self.last_candle_time.hour, session)
        if context['adx'] < 25:
            if is_sb_time and context['adx'] >= 15:
                logging.info(f"[ADX_SB_PASS] ADX={context['adx']} — SB exception")
            else:
                logging.info(f"[ADX_FILTER] ❌ ADX={context['adx']} < 25 — skipping")
                return
        logging.info(f"[ADX_PASS] ADX={context['adx']:.1f} ✅")

        htf_structure = self.data_manager.get_htf_structure(self.symbol)
        htf_bias      = self.data_manager.get_h1_bias(self.symbol)

        signal = self.strategy.evaluate(
            df, session, structure, fvg_zones, ob,
            current_price, atr, confidence, htf_bias, htf_structure
        )
        if not signal:
            return

        if not isinstance(signal, dict) or 'zone' not in signal:
            logging.error(f"[SIGNAL_ERROR] Invalid signal: {signal}")
            return

        # Deduplicate by FVG zone
        if 'time' in signal['zone']:
            fvg_key = (
                signal['zone']['time'].isoformat()
                if hasattr(signal['zone']['time'], 'isoformat')
                else str(signal['zone']['time']),
                signal['zone'].get('type', 'unknown'),
                round(signal['zone'].get('high', 0), 2),
                round(signal['zone'].get('low', 0), 2)
            )
            if fvg_key in self.used_fvg_zones:
                return
            self.used_fvg_zones.add(fvg_key)

        quality_score = signal.get('quality', 1.0)

        # ── Multi-agent orchestrator ──────────────────────────
        if (Config.ORCHESTRATOR_ENABLED
                and quality_score >= Config.AGENT_MIN_QUALITY
                and not Config.BACKTEST_MODE):

            final_signal = self._run_orchestrator(
                signal, context, htf_bias, htf_structure,
                fvg_zones, structure, session, atr, confidence
            )
            if final_signal is None:
                return          # orchestrator vetoed the trade
            signal = final_signal
        # ─────────────────────────────────────────────────────

        self._execute_trade(
            signal['type'], signal['zone'], signal['entry_price'],
            atr, session, structure, quality_score, signal
        )

    # ── Orchestrator bridge ───────────────────────────────────

    def _run_orchestrator(self, signal: Dict, context: Dict,
                          htf_bias: str, htf_structure: Dict,
                          fvg_zones: List, structure: str,
                          session: str, atr: float,
                          confidence: float) -> Optional[Dict]:
        """
        Build TradingState, invoke the LangGraph graph, and return the
        (possibly direction-corrected) signal — or None to skip.
        """
        from orchestrator.state import TradingState

        df            = self.current_df
        current_price = df['close'].iloc[-2]
        quality_score = signal.get('quality', 1.0)

        symbol_info = mt5.symbol_info(self.symbol)
        point       = symbol_info.point if symbol_info else 0.01

        sl_dist = min(max(atr * Config.SL_ATR_MULTIPLIER, Config.MIN_SL_PRICE), Config.MAX_SL_PRICE)
        tp_dist = sl_dist * Config.RR_RATIO

        zone_type  = MarketStructureDetector.is_premium_discount(df)
        liq_sweep  = MarketStructureDetector.detect_liquidity_sweep(df)

        state: TradingState = {
            # ── Market data ────────────────────────────────
            'symbol':        self.symbol,
            'df':            df,
            'session':       session,
            'current_price': current_price,
            'atr':           float(atr),
            'structure':     structure,
            'fvg_zones':     fvg_zones,
            'ob':            MarketStructureDetector.find_order_block(df),
            'context':       context,
            'htf_bias':      htf_bias,
            'htf_structure': htf_structure,
            'signal':        signal,
            'confidence':    float(confidence),
            'quality_score': float(quality_score),
            # ── Derived trade levels ───────────────────────
            'entry_price':   signal['entry_price'],
            'sl_price':      (signal['entry_price'] - sl_dist
                              if signal['type'] == 'buy'
                              else signal['entry_price'] + sl_dist),
            'tp_price':      (signal['entry_price'] + tp_dist
                              if signal['type'] == 'buy'
                              else signal['entry_price'] - tp_dist),
            'sl_distance':   sl_dist,
            'tp_distance':   tp_dist,
            'zone_type':     zone_type,
            'liquidity_swept': bool(liq_sweep),
            'is_silver_bullet': self.strategy._is_silver_bullet_time(
                                    self.last_candle_time.hour, session),
            'hour':          int(self.last_candle_time.hour),
            'balance':       self.stats_tracker.balance,
            # ── Agent outputs (filled by graph nodes) ──────
            'market_analysis':  None,
            'news_analysis':    None,
            'risk_assessment':  None,
            'validation':       None,
            # ── Decision ───────────────────────────────────
            'approved':         False,
            'final_action':     'skip',
            'rejection_reasons': [],
            'execution_result': None,
            # ── Meta ───────────────────────────────────────
            'timestamp':  self.last_candle_time.isoformat(),
            'iteration':  len(self.stats_tracker.trades),
        }

        try:
            graph        = self._get_graph()
            final_state  = graph.invoke(state)

            if not final_state.get('approved', False):
                reasons = final_state.get('rejection_reasons', [])
                logging.info(f"[ORCHESTRATOR] ⏭️ Trade vetoed — {reasons}")
                return None

            final_action = final_state.get('final_action', 'skip')
            if final_action == 'skip':
                return None

            # Possibly direction-corrected signal
            updated = dict(signal)
            updated['type'] = final_action
            logging.info(f"[ORCHESTRATOR] ✅ Approved: {final_action.upper()}")
            return updated

        except Exception as exc:
            logging.error(f"[ORCHESTRATOR_ERROR] {exc} — falling back to rule-based")
            return signal   # graceful fallback: trust the rule-based signal

    # ── Execute Trade ─────────────────────────────────────────

    def _execute_trade(self, order_type: str, zone: Dict,
                       entry_price: float, atr: float,
                       session: str, structure: str,
                       quality_score: float = 1.0,
                       signal: Optional[Dict] = None):

        if self.circuit_breaker and not Config.BACKTEST_MODE:
            if not self.circuit_breaker.check_and_trigger(self.stats_tracker):
                return
        if not self.performance_monitor.check_performance(self.stats_tracker):
            return
        if not ImprovedRiskManager.check_drawdown_limit(self.stats_tracker):
            return
        if self.stats_tracker.balance < Config.MIN_BALANCE:
            return
        if self.session_trade_counts[session] >= Config.SESSION_MAX_TRADES[session]:
            return

        trades_today = sum(
            1 for t in self.stats_tracker.trades
            if pd.Timestamp(t['time']).date() == self.last_candle_time.date()
        )
        if trades_today >= Config.MAX_DAILY_TRADES:
            return

        recent_7d  = [t for t in self.stats_tracker.trades
                      if pd.Timestamp.now(tz='UTC') -
                      pd.Timestamp(t['time']) < pd.Timedelta(days=7)]
        win_rate   = (len([t for t in recent_7d if t['pnl'] - t['trade_cost'] > 0]) /
                      len(recent_7d) if recent_7d else 0.5)
        dynamic_risk = ImprovedRiskManager.calculate_dynamic_position_size(
            self.stats_tracker.balance, win_rate, atr, atr)
        self.online_model.optimize_parameters(
            self.current_df, session, win_rate,
            self.current_df['close'].iloc[-1], zone
        )

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return
        point = symbol_info.point

        context = MarketStructureDetector.get_market_context(self.current_df)
        is_sb   = self.strategy._is_silver_bullet_time(self.last_candle_time.hour, session)

        if context['volatility'] == 'high':
            sl_mult, rr = Config.HIGH_VOL_SL_MULTIPLIER, Config.HIGH_VOL_RR_RATIO
        elif is_sb and context['regime'] == 'trending':
            sl_mult, rr = 1.8, 4.0
        else:
            sl_mult, rr = Config.SL_ATR_MULTIPLIER, Config.RR_RATIO

        sl_distance = atr * sl_mult
        sl_distance = max(sl_distance, Config.MIN_SL_PRICE)
        sl_distance = min(sl_distance, Config.MAX_SL_PRICE)

        if order_type == 'buy':
            sl = entry_price - sl_distance
            tp = entry_price + sl_distance * rr
        else:
            sl = entry_price + sl_distance
            tp = entry_price - sl_distance * rr

        sl_pts = sl_distance / point

        logging.info(
            f"[SL_TP_FINAL] SL={sl:.2f} ({sl_distance:.1f} u), "
            f"TP={tp:.2f} ({sl_distance*rr:.1f} u), ATR={atr:.1f}, RR={rr:.1f}"
        )

        tick = mt5.symbol_info_tick(self.symbol)
        if tick and (tick.ask - tick.bid) > Config.MAX_SPREAD:
            return

        if abs(tp - entry_price) / point < Config.MIN_SB_TARGET_POINTS:
            logging.warning("[MIN_TARGET_FAIL] Target too close — skipping")
            return

        lot = ImprovedRiskManager.calculate_kelly_lot_size(
            self.symbol, sl_pts, self.stats_tracker.balance,
            atr, win_rate, dynamic_risk, quality_score, zone
        )
        if lot <= 0:
            return

        logging.info(
            f"[TRADE_OPENING] {order_type.upper()} {self.symbol} "
            f"Entry={entry_price:.2f} SL={sl:.2f} TP={tp:.2f} "
            f"Lot={lot:.2f} Q={quality_score:.1f} Session={session}"
        )
        self.session_trade_counts[session] += 1

        if Config.VISUAL_DEMO and Config.PAPER_TRADING:
            self._place_order(order_type, lot, sl, tp, atr, session, structure,
                              entry_price, quality_score=quality_score,
                              instant_execution=True)
        elif Config.PAPER_TRADING or Config.BACKTEST_MODE:
            self._simulate_trade(order_type, lot, sl, tp, entry_price,
                                 atr, session, structure, zone, signal)
        else:
            self._place_order(order_type, lot, sl, tp, atr, session,
                              structure, entry_price)

    # ── Simulate Trade ────────────────────────────────────────

    def _simulate_trade(self, order_type: str, lot: float,
                        sl: float, tp: float, entry_price: float,
                        atr: float, session: str, structure: str,
                        zone: Dict, signal: Optional[Dict] = None):
        df           = self.current_df
        current_time = self.last_candle_time
        if current_time is None or current_time not in df.index:
            return

        spread   = 2.0
        slippage = atr * 0.1
        entry_price = (entry_price + spread + slippage if order_type == 'buy'
                       else entry_price - spread - slippage)

        if Config.BACKTEST_MODE:
            extended = self.data_manager.get_candles(
                self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES,
                Config.BACKTEST_START_DATE, Config.BACKTEST_END_DATE
            )
            if extended is not None and not extended.empty:
                df = extended

        try:
            current_pos = df.index.get_loc(current_time)
        except KeyError:
            return

        exit_idx = exit_reason = exit_price = None
        current_sl            = sl
        breakeven_triggered   = False

        for i in range(current_pos + 1, min(current_pos + 100, len(df))):
            candle     = df.iloc[i]
            current_sl = self._update_trailing_sl(candle['close'], current_sl, atr, order_type == 'buy')

            if not breakeven_triggered:
                bp = (entry_price + (tp - entry_price) * 0.5 if order_type == 'buy'
                      else entry_price - (entry_price - tp) * 0.5)
                if ((order_type == 'buy'  and candle['close'] >= bp) or
                        (order_type == 'sell' and candle['close'] <= bp)):
                    current_sl        = entry_price
                    breakeven_triggered = True

            if order_type == 'buy':
                if candle['low'] <= current_sl:
                    exit_idx, exit_reason, exit_price = i, 'sl_hit', current_sl - slippage; break
                elif candle['high'] >= tp:
                    exit_idx, exit_reason, exit_price = i, 'tp_hit', tp - slippage; break
            else:
                if candle['high'] >= current_sl:
                    exit_idx, exit_reason, exit_price = i, 'sl_hit', current_sl + slippage; break
                elif candle['low'] <= tp:
                    exit_idx, exit_reason, exit_price = i, 'tp_hit', tp + slippage; break

        if exit_idx is None:
            exit_price  = df['close'].iloc[-1]
            exit_reason = 'open'
            exit_idx    = len(df) - 1

        duration    = (df.index[exit_idx] - current_time).total_seconds() / 60.0
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return
        tv    = symbol_info.trade_tick_value
        point = symbol_info.point

        if pd.isna(entry_price) or pd.isna(exit_price):
            return

        pnl = ((exit_price - entry_price) if order_type == 'buy'
               else (entry_price - exit_price)) * lot * tv / point
        trade_cost  = ImprovedRiskManager.calculate_trade_cost(self.symbol, lot, order_type)
        net_pnl     = pnl - trade_cost
        exit_reason = 'win' if net_pnl > 0 else 'loss'

        risk     = abs(entry_price - sl) * lot * tv / point if tv and point else 1
        profit_r = net_pnl / risk if risk > 0 else 0


        pattern_key = signal.get('pattern_key') if isinstance(signal, dict) else None
        if pattern_key and self.pattern_tracker:
            self.pattern_tracker.record_trade(
                pattern_key, net_pnl,
                {'type': order_type, 'session': session, 'structure': structure,
                 'entry_price': entry_price, 'exit_price': exit_price, 'duration': duration}
            )

        self.stats_tracker.add_trade(
            order_type, lot, sl, tp, exit_reason, pnl, trade_cost,
            df.index[exit_idx], session, atr, structure, duration, pattern_key
        )

        if self.pattern_tracker:
            n = len(self.stats_tracker.trades)
            if self.pattern_tracker.update_weights(n):
                logging.info(self.pattern_tracker.get_stats_summary())
            if n % 10 == 0:
                self.pattern_tracker.save_data()

        trade_outcome = 1 if exit_reason == 'win' else 0
        if self.circuit_breaker and not Config.BACKTEST_MODE:
            self.circuit_breaker.record_trade_result(trade_outcome == 1)
        if self.ml_health_monitor and not Config.BACKTEST_MODE:
            self.ml_health_monitor.record_prediction(
                was_correct=(trade_outcome == 1),
                confidence=signal.get('ml_confidence', 0.5) if signal else 0.5
            )

        risk_amount = abs(entry_price - sl) * lot * tv / point
        profit_r_ml = net_pnl / risk_amount if risk_amount > 0 else 0
        self.online_model.update(
            df.iloc[:current_pos + 1], session, entry_price,
            zone or {}, trade_outcome, profit_r_ml
        )
        self.stats_tracker.export_detailed_stats()

    def _update_trailing_sl(self, current_price: float, sl: float,
                            atr: float, is_buy: bool) -> float:
        info  = mt5.symbol_info(self.symbol)
        min_d = (getattr(info, 'trade_stops_level', 0) * info.point if info else 0)
        trail = (current_price - Config.TRAILING_STOP_ATR * atr if is_buy
                 else current_price + Config.TRAILING_STOP_ATR * atr)
        return (max(trail, sl + min_d) if is_buy else min(trail, sl - min_d))

    # ── Backtest ──────────────────────────────────────────────

    def run_persistent_backtest(self, n_folds: int = 5):
        self.stats_tracker.trades     = []
        self.stats_tracker.balance    = 180.0
        self.stats_tracker.equity     = [180.0]
        self.stats_tracker.max_equity = 180.0
        self.stats_tracker.max_drawdown = 0.0
        logging.info("[BACKTEST_START] Fresh backtest")

        _orig = Config.MAX_DAILY_TRADES
        Config.MAX_DAILY_TRADES = 999999

        start     = pd.Timestamp(Config.BACKTEST_START_DATE, tz='UTC')
        end       = pd.Timestamp(Config.BACKTEST_END_DATE,   tz='UTC')
        fold_days = max((end - start).days // n_folds, 10)

        for fold in range(n_folds):
            if len(self.stats_tracker.trades) >= Config.PER_BACKTEST_MAX_TRADES:
                break
            ts = start + pd.offsets.Day(fold * fold_days)
            te = ts    + pd.offsets.Day(int(fold_days * 0.7))
            xs = te    + pd.offsets.Day(1)
            xe = ts    + pd.offsets.Day(fold_days)

            logging.info(f"[FOLD {fold+1}/{n_folds}] Train={ts.date()}→{te.date()}, "
                         f"Test={xs.date()}→{xe.date()}")

            train_df = self.data_manager.get_candles(
                self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES,
                ts.strftime('%Y-%m-%d'), te.strftime('%Y-%m-%d')
            )
            if train_df is not None and not train_df.empty:
                self.current_df = train_df
                self._train_ml_model(train_df)
            else:
                continue

            test_df = self.data_manager.get_candles(
                self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES,
                xs.strftime('%Y-%m-%d'), xe.strftime('%Y-%m-%d')
            )
            if test_df is not None and not test_df.empty:
                for i in range(Config.MIN_CANDLES, len(test_df)):
                    if len(self.stats_tracker.trades) >= Config.PER_BACKTEST_MAX_TRADES:
                        self.stats_tracker.print_stats(force_print=True)
                        Config.MAX_DAILY_TRADES = _orig
                        self._export_backtest()
                        return
                    self.current_df       = test_df.iloc[:i + 1]
                    self.last_candle_time = test_df.index[i]
                    self._evaluate_signals()

            s = self.stats_tracker.get_stats()
            logging.info(
                f"[FOLD_RESULTS] Fold {fold+1}: "
                f"Trades={s['total_trades']}, WR={s['win_rate']:.1%}, P&L=${s['net_pnl']:.2f}"
            )

        Config.MAX_DAILY_TRADES = _orig
        self._export_backtest()

    def _export_backtest(self):
        try:
            with open('backtest_trading_stats.pkl', 'wb') as fh:
                pickle.dump({
                    'trades':       self.stats_tracker.trades,
                    'equity':       self.stats_tracker.equity,
                    'max_drawdown': self.stats_tracker.max_drawdown,
                    'start_date':   Config.BACKTEST_START_DATE,
                    'end_date':     Config.BACKTEST_END_DATE
                }, fh)
            self.online_model.save_locked_scaler_and_model()
            if os.path.exists(self.online_model.model_file):
                shutil.copy2(self.online_model.model_file, 'pretrained_from_backtest_model.pkl')
            if os.path.exists(self.online_model.scaler_file):
                shutil.copy2(self.online_model.scaler_file, 'pretrained_from_backtest_scaler.pkl')
            logging.info("[BACKTEST_EXPORT] ✅")
        except Exception as exc:
            logging.error(f"[BACKTEST_EXPORT_ERROR] {exc}")

    # ── Demo ──────────────────────────────────────────────────

    def run_persistent_demo(self):
        self.stats_tracker.trades     = []
        self.stats_tracker.balance    = 180.0
        self.stats_tracker.equity     = [180.0]
        self.stats_tracker.max_equity = 180.0
        self.stats_tracker.max_drawdown = 0.0
        logging.info("[DEMO_START] Starting demo session")

        end   = pd.Timestamp.now(tz='UTC')
        start = end - pd.Timedelta(days=7)

        train_df = self.data_manager.get_candles(
            self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES,
            start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        )
        if train_df is not None and not train_df.empty:
            self._train_ml_model(train_df)

        init_df = self.data_manager.get_candles(
            self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES,
            start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        )
        last_t    = init_df.index[-1] if init_df is not None and not init_df.empty else None
        sleep_map = {15: 900, 5: 300, 1: 60}   # M15=15, M5=5, M1=1
        sleep_s   = sleep_map.get(Config.TIMEFRAME, 60)
        iteration = 0

        while (len(self.stats_tracker.trades) < Config.PER_DEMO_MAX_TRADES
               and iteration < 1000):
            # ── wait while paused / stopped by Telegram ───────
            from core.bot_controller import get_controller as _gc
            _ctrl = _gc()
            while _ctrl.is_paused() and not _ctrl.is_stopped():
                _ctrl.wait_if_paused(timeout=30.0)
            if _ctrl.is_stopped():
                break
            # ── handle live mode switch without restart ────────
            _cur_mode = _ctrl.get_mode()
            if _cur_mode == 'live':
                logging.info("[ENGINE] Mode switched to LIVE — restarting loop")
                break
            iteration += 1
            end   = pd.Timestamp.now(tz='UTC')
            start = end - pd.Timedelta(days=7)
            live_df = self.data_manager.get_candles(
                self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES,
                start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
            )
            if live_df is None or live_df.empty:
                time.sleep(60)
                continue
            ct = live_df.index[-1]
            if last_t is not None and ct <= last_t:
                time.sleep(sleep_s)
                continue
            self.current_df       = live_df
            self.last_candle_time = ct
            last_t                = ct
            self._monitor_open_positions()
            self._evaluate_signals()
            # ── Periodic Telegram heartbeat (every 4 candles) ─
            if iteration % 4 == 0:
                try:
                    from utils.telegram_notifier import get_notifier as _gn
                    _s   = self.stats_tracker
                    _pr  = live_df['close'].iloc[-1]
                    _mod = _ctrl.get_mode().upper()
                    _gn().send(
                        f"📊 <b>CYCLE #{iteration}</b>  [{_mod}]\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💰 Price:   <code>{_pr:.2f}</code>\n"
                        f"📈 Trades:  <code>{len(_s.trades)}</code>\n"
                        f"💵 Balance: <code>${_s.balance:.2f}</code>\n"
                        f"🕐 Candle:  <code>{ct.strftime('%H:%M')}</code>",
                        silent=True
                    )
                except Exception:
                    pass
            time.sleep(sleep_s)

    # ── ML Initial Training ───────────────────────────────────

    def _train_ml_model(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        X, y = [], []
        for i in range(1, len(df)):
            feats = self.online_model.prepare_features(
                df.iloc[:i],
                self._get_current_session(df.index[i - 1]),
                df['close'].iloc[i - 1],
                {}
            )
            X.append(feats)
            pc = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
            y.append(1 if abs(pc) > 0.002 else 0)

        X_df = pd.DataFrame(X)
        if len(np.unique(y)) < 2 or len(y) < 50:
            return
        try:
            from sklearn.utils.class_weight import compute_class_weight
            if self.online_model.selected_features is None:
                ts  = StandardScaler()
                Xs  = ts.fit_transform(X_df)
                tm  = XGBClassifier(n_estimators=50, max_depth=3, random_state=99, n_jobs=-1)
                tm.fit(Xs, y)
                top = np.argsort(tm.feature_importances_)[-15:]
                self.online_model.selected_features = X_df.columns[top].tolist()
                import gc; del tm, ts, Xs; gc.collect()

            X_sel = X_df[self.online_model.selected_features]
            if not hasattr(self.online_model.scaler, 'mean_'):
                X_scaled = pd.DataFrame(
                    self.online_model.scaler.fit_transform(X_sel),
                    columns=self.online_model.selected_features
                )
            else:
                X_scaled = pd.DataFrame(
                    self.online_model.scaler.transform(X_sel),
                    columns=self.online_model.selected_features
                )

            classes = np.unique(y)
            cw  = compute_class_weight('balanced', classes=classes, y=y)
            spw = cw[1] / cw[0] if len(cw) > 1 else 1.0

            if not self.online_model.is_trained:
                self.online_model.model = XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                    gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
                    scale_pos_weight=spw, random_state=42, n_jobs=-1
                )
            else:
                try:
                    self.online_model.model.set_params(scale_pos_weight=spw)
                except Exception:
                    pass

            if (self.online_model.is_trained and
                    hasattr(self.online_model.model, 'get_booster')):
                try:
                    self.online_model.model.fit(
                        X_scaled, y,
                        xgb_model=self.online_model.model.get_booster()
                    )
                except Exception:
                    self.online_model.model.fit(X_scaled, y)
            else:
                self.online_model.model.fit(X_scaled, y)

            self.online_model.is_trained = True
            yp = self.online_model.model.predict(X_scaled)
            logging.info(
                f"[MODEL_TRAINED] ✅ Acc={accuracy_score(y, yp):.2%}, "
                f"F1={f1_score(y, yp, average='weighted'):.2%}, N={len(y)}"
            )
        except Exception as exc:
            logging.error(f"[TRAINING_ERROR] {exc}")

    # ── Live position tracking ────────────────────────────────

    def _load_open_positions_from_mt5(self):
        # Skip entirely if MT5 is not the active exchange or not yet connected
        if not _MT5_AVAILABLE or mt5 is None:
            return
        active_ex = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5').lower()
        if active_ex != 'mt5':
            return
        # account_info() returns None when MT5 is NOT initialized — safe check
        if mt5.account_info() is None:
            return
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                for pos in positions:
                    if pos.magic == 234000 or 'ICT_' in (pos.comment or ''):
                        parts     = (pos.comment or 'ICT_new_york_unknown').split('_')
                        session   = parts[1] if len(parts) > 1 else 'new_york'
                        structure = parts[2] if len(parts) > 2 else 'unknown'
                        df        = self.data_manager.get_candles(
                            self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES)
                        atr = df['atr'].iloc[-1] if df is not None and not df.empty else 10.0
                        self.open_positions[pos.ticket] = {
                            'type':        'buy' if pos.type == 0 else 'sell',
                            'entry_price': pos.price_open,
                            'sl': pos.sl, 'tp': pos.tp, 'lot': pos.volume,
                            'open_time':   pd.Timestamp(pos.time, unit='s', tz='UTC'),
                            'session': session, 'structure': structure, 'atr': atr
                        }
                logging.info(f"[RESTART_RECOVERY] ✅ {len(self.open_positions)} positions recovered")
        except Exception as exc:
            logging.warning(f"[RESTART_RECOVERY] {exc}")

    def _monitor_open_positions(self):
        if not self.open_positions:
            return
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return
        existing = {p.ticket for p in positions}
        closed   = []

        for ticket, pd_ in list(self.open_positions.items()):
            if ticket not in existing:
                deals = mt5.history_deals_get(position=ticket)
                if deals:
                    exit_deal = (
                        next((d for d in deals if d.entry == 1), None) or
                        max(deals, key=lambda d: abs(d.profit))
                    )
                    if exit_deal:
                        pnl        = exit_deal.profit
                        exit_price = exit_deal.price
                        lot        = pd_['lot']
                        otype      = pd_['type']
                        trade_cost = ImprovedRiskManager.calculate_trade_cost(
                            self.symbol, lot, otype)
                        result     = 'win' if pnl > 0 else 'loss'
                        exit_time  = pd.Timestamp(exit_deal.time, unit='s', tz='UTC')
                        duration   = (exit_time - pd_['open_time']).total_seconds() / 60.0

                        self.stats_tracker.add_trade(
                            otype, lot, pd_['sl'], pd_['tp'], result, pnl, trade_cost,
                            exit_time, pd_['session'], pd_['atr'], pd_['structure'], duration
                        )

                        net      = pnl
                        risk_r   = abs(pd_['entry_price'] - pd_['sl']) * lot
                        profit_r = net / risk_r if risk_r > 0 else 0

                        if self.circuit_breaker and not Config.BACKTEST_MODE:
                            self.circuit_breaker.record_trade_result(pnl > 0)
                        if self.ml_health_monitor and not Config.BACKTEST_MODE:
                            self.ml_health_monitor.record_prediction(was_correct=(pnl > 0))

                        self.stats_tracker.save_stats()

                        if pd_.get('entry_features'):
                            risk_amt    = abs(pd_['entry_price'] - pd_['sl']) * lot
                            profit_r_ml = pnl / risk_amt if risk_amt > 0 else 0
                            self.online_model.update_from_features(
                                pd_['entry_features'], pd_['session'],
                                1 if pnl > 0 else 0, profit_r_ml
                            )

                closed.append(ticket)
        for t in closed:
            del self.open_positions[t]

    def _place_order(self, order_type: str, lot: float,
                     sl: float, tp: float, atr: float,
                     session: str, structure: str, entry_price: float,
                     quality_score: float = 1.0, instant_execution: bool = False):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        price         = tick.ask if order_type == 'buy' else tick.bid
        order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL
        dist          = abs(price - entry_price)

        if instant_execution:
            MAX_DRIFT = 150.0
            if dist > MAX_DRIFT:
                logging.warning(f"[INSTANT_SKIP] Drifted {dist:.1f} > {MAX_DRIFT}")
                return
        else:
            if dist > atr * 0.2:
                logging.warning(f"[OTE_MISS] {dist:.1f} from OTE > {atr*0.2:.1f}")
                return
            SPREAD_BUFFER = 5.0
            if order_type == 'buy'  and price > entry_price + SPREAD_BUFFER:
                logging.warning(f"[CHASING_BUY] Price {price:.1f} above OTE {entry_price:.1f}")
                return
            if order_type == 'sell' and price < entry_price - SPREAD_BUFFER:
                logging.warning(f"[CHASING_SELL] Price {price:.1f} below OTE {entry_price:.1f}")
                return

        point        = symbol_info.point
        sl_dist_live = atr * Config.SL_ATR_MULTIPLIER
        sl_dist_live = max(sl_dist_live, Config.MIN_SL_PRICE)
        sl_dist_live = min(sl_dist_live, Config.MAX_SL_PRICE)

        if order_type == 'buy':
            sl = price - sl_dist_live
            tp = price + sl_dist_live * Config.RR_RATIO
        else:
            sl = price + sl_dist_live
            tp = price - sl_dist_live * Config.RR_RATIO

        sl_pts_live = sl_dist_live / point

        if order_type == 'buy'  and (price > tp or sl >= price): return
        if order_type == 'sell' and (price < tp or sl <= price): return

        ts  = symbol_info.trade_tick_size
        sl  = round(sl / ts) * ts
        tp  = round(tp / ts) * ts
        msl = symbol_info.trade_stops_level
        if msl > 0:
            md = msl * point
            if order_type == 'buy':
                if abs(price - sl) < md: sl = price - md
                if abs(tp - price) < md: tp = price + md
            else:
                if abs(sl - price) < md: sl = price + md
                if abs(price - tp) < md: tp = price - md

        fm = symbol_info.filling_mode
        tf = (mt5.ORDER_FILLING_FOK    if fm & 1 else
              mt5.ORDER_FILLING_IOC    if fm & 2 else
              mt5.ORDER_FILLING_RETURN)

        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
            "volume": lot, "type": order_type_mt5, "price": price,
            "sl": sl, "tp": tp, "deviation": 5, "magic": 234000,
            "comment": f"ICT_{session}_{structure}",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": tf
        }
        result = mt5.order_send(req)
        if result is None:
            logging.error(f"[ORDER_ERROR] {mt5.last_error()}")
            return
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode == 10027:
                raise SystemExit("[CRITICAL] AutoTrading DISABLED in MT5!")
            logging.error(f"[ORDER_FAILED] retcode={result.retcode}, {result.comment}")
            return

        logging.info(
            f"[ORDER_SUCCESS] ✅ Ticket={result.order}, "
            f"Price={result.price:.2f}, SL={sl:.2f}, TP={tp:.2f}"
        )

        entry_features = None
        try:
            df_fresh = self.data_manager.get_candles(
                self.symbol, Config.TIMEFRAME, Config.MIN_CANDLES)
            if df_fresh is not None and not df_fresh.empty:
                entry_features = self.online_model.prepare_features(
                    df_fresh, session, result.price, {})
        except Exception as exc:
            logging.warning(f"[FEATURES_ERROR] {exc}")

        self.open_positions[result.order] = {
            'type': order_type, 'entry_price': result.price,
            'sl': sl, 'tp': tp, 'lot': lot,
            'open_time': pd.Timestamp.now(tz='UTC'),
            'session': session, 'structure': structure, 'atr': atr,
            'entry_features': entry_features
        }

    # ── Finalize / Shutdown ───────────────────────────────────

    def _finalize(self):
        self.stats_tracker.print_stats(force_print=True)
        self.stats_tracker.export_detailed_stats()
        self.online_model.save_training_data()
        self.stats_tracker.save_stats()
        if self.pattern_tracker:
            self.pattern_tracker.save_data()
            logging.info(self.pattern_tracker.get_stats_summary())

    def shutdown(self):
        try:
            self.online_model.save_training_data()
            self.stats_tracker.save_stats()
            if self.pattern_tracker:
                self.pattern_tracker.save_data()
                logging.info(self.pattern_tracker.get_stats_summary())
            mt5.shutdown()
            logging.info("[SHUTDOWN_SUCCESS] ✅")
        except Exception as exc:
            logging.error(f"[SHUTDOWN_ERROR] {exc}")
