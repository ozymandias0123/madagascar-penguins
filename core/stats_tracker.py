"""
core/stats_tracker.py — Persistent trade statistics tracker.
Ported from ozy.py / PersistentStatsTracker unchanged.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd

from config import Config


class PersistentStatsTracker:

    def __init__(self, user_email: Optional[str] = None):
        self.trades: List[Dict]  = []
        self.user_email          = user_email
        self.print_interval      = 999

        prefix          = Config.get_file_prefix()
        self.stats_file = f'{prefix}_trading_stats.pkl'
        self.csv_file   = f'{prefix}_detailed_stats.csv'

        # Sync balance from MT5 — only if MT5 is ALREADY connected (no auto-launch)
        self.balance = Config.INITIAL_BALANCE
        try:
            acct = mt5.account_info()   # returns None if not connected — safe
            if acct is not None:
                self.balance = acct.balance
                logging.info(f"[BALANCE_SYNC] MT5 balance: ${self.balance:.2f}")
        except Exception:
            pass   # MT5 not available / not connected yet — use default balance

        self.equity       = [self.balance]
        self.max_equity   = self.balance
        self.max_drawdown = 0.0

        self.load_stats()
        logging.info(
            f"[STATS_INIT] Mode={prefix}, Balance=${self.balance:.2f}, "
            f"Trades={len(self.trades)}"
        )

    # ── Persistence ───────────────────────────────────────────

    def load_stats(self):
        if Config.BACKTEST_MODE:
            logging.info("[BACKTEST] Starting fresh stats")
            return

        pkl_trades, csv_trades = [], []

        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'rb') as fh:
                    data       = pickle.load(fh)
                pkl_trades = data.get('trades', [])
                logging.info(f"[STATS_PKL] {len(pkl_trades)} trades")
            except Exception as exc:
                logging.error(f"[STATS_LOAD_ERROR] PKL: {exc}")

        if os.path.exists(self.csv_file):
            try:
                csv_df = pd.read_csv(self.csv_file)
                if not csv_df.empty:
                    csv_trades = csv_df.to_dict('records')
                    logging.info(f"[STATS_CSV] {len(csv_trades)} trades")
            except Exception as exc:
                logging.error(f"[STATS_LOAD_ERROR] CSV: {exc}")

        if len(csv_trades) > len(pkl_trades):
            self.trades = csv_trades
            self.save_stats()
        elif pkl_trades:
            self.trades = pkl_trades
        elif csv_trades:
            self.trades = csv_trades
            self.save_stats()

        logging.info(f"[STATS_LOAD] ✅ {len(self.trades)} trades loaded")

    def save_stats(self):
        if Config.BACKTEST_MODE:
            return
        try:
            with open(self.stats_file, 'wb') as fh:
                pickle.dump({'trades':       self.trades,
                             'equity':       self.equity,
                             'max_drawdown': self.max_drawdown}, fh)
            logging.info(f"[STATS_SAVED] {len(self.trades)} trades → {self.stats_file}")
        except Exception as exc:
            logging.error(f"[STATS_SAVE_ERROR] {exc}")

    def auto_save_check(self, trade_count: int):
        if trade_count % Config.SAVE_INTERVAL_TRADES == 0:
            self.save_stats()
            self.export_detailed_stats()

    # ── Trade recording ───────────────────────────────────────

    def add_trade(self, order_type: str, lot: float, sl: float, tp: float,
                  result: str, pnl: float, trade_cost: float,
                  trade_time, session: str, atr: float,
                  structure: str, duration: float,
                  pattern_key: Optional[str] = None):
        trade = {
            'type': order_type, 'lot': lot, 'sl': sl, 'tp': tp,
            'result': result, 'pnl': pnl, 'trade_cost': trade_cost,
            'time': trade_time.isoformat() if isinstance(trade_time, pd.Timestamp) else str(trade_time),
            'session': session, 'atr': atr, 'structure': structure,
            'duration': duration, 'pattern_key': pattern_key
        }
        self.trades.append(trade)
        net             = pnl - trade_cost
        self.balance   += net
        self.equity.append(self.balance)
        self.max_equity = max(self.max_equity, self.balance)
        dd              = (self.max_equity - self.balance) / self.max_equity if self.max_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, dd)
        self._print_trade_result(trade, net)
        self.auto_save_check(len(self.trades))
        self.export_detailed_stats()

    def _print_trade_result(self, trade: Dict, net_pnl: float):
        emoji  = "✅" if net_pnl > 0 else "❌" if net_pnl < 0 else "➖"
        label  = "WIN" if net_pnl > 0 else "LOSS" if net_pnl < 0 else "BREAK-EVEN"
        wins   = len([t for t in self.trades if t['pnl'] - t['trade_cost'] > 0])
        n      = len(self.trades)
        msg    = (
            f"\n{'─'*60}\n{emoji} TRADE #{n}: {label}\n{'─'*60}\n"
            f"📍 {trade['type'].upper()} | {trade['session']}\n"
            f"💰 P&L: ${net_pnl:+.2f}  Balance: ${self.balance:.2f}\n"
            f"🎯 WR: {wins}/{n} ({wins/n:.1%})\n{'─'*60}"
        )
        print(msg)
        logging.info(msg)

    # ── Stats query ───────────────────────────────────────────

    def get_stats(self) -> Dict:
        n = len(self.trades)
        if n == 0:
            return {
                'total_trades': 0, 'wins': 0, 'win_rate': 0.0,
                'total_pnl': 0.0, 'total_cost': 0.0, 'net_pnl': 0.0,
                'current_balance': self.balance, 'max_drawdown': 0.0,
                'avg_trade_duration': 0.0
            }
        wins = len([t for t in self.trades if t['pnl'] - t['trade_cost'] > 0])
        return {
            'total_trades': n,
            'wins':         wins,
            'win_rate':     wins / n,
            'total_pnl':    sum(t['pnl'] for t in self.trades),
            'total_cost':   sum(t['trade_cost'] for t in self.trades),
            'net_pnl':      sum(t['pnl'] - t['trade_cost'] for t in self.trades),
            'current_balance': self.balance,
            'max_drawdown': self.max_drawdown,
            'avg_trade_duration': sum(t['duration'] for t in self.trades) / n
        }

    def print_stats(self, force_print: bool = False):
        if not force_print and len(self.trades) % self.print_interval != 0:
            return
        stats = self.get_stats()
        out   = (
            f"\n=== Trading Summary ({Config.get_file_prefix().upper()}) ===\n"
            f"Trades: {stats['total_trades']} | Wins: {stats['wins']} ({stats['win_rate']:.1%})\n"
            f"Net P&L: ${stats['net_pnl']:.2f} | Balance: ${stats['current_balance']:.2f}\n"
            f"Max DD: {stats['max_drawdown']:.1%}\n"
        )
        print(out)
        logging.info(out)

    def export_detailed_stats(self):
        try:
            df = pd.DataFrame(self.trades)
            if df.empty:
                return
            if os.path.exists(self.csv_file):
                try:
                    existing = pd.read_csv(self.csv_file)
                    if len(df) < len(existing):
                        return
                except Exception:
                    pass
            df.to_csv(self.csv_file, index=False)
        except PermissionError:
            logging.warning("[STATS_EXPORT] File open elsewhere")
        except Exception as exc:
            logging.error(f"[STATS_EXPORT_ERROR] {exc}")
