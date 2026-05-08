"""
core/risk_manager.py — Position sizing and drawdown checks.
Ported from ozy.py / ImprovedRiskManager unchanged.
"""

import logging
from typing import Dict, Optional

import MetaTrader5 as mt5

from config import Config


class ImprovedRiskManager:

    @staticmethod
    def get_dynamic_limits(balance: float) -> Dict:
        if balance < 50:
            return {'max_lot': 0.01, 'min_lot': 0.01, 'risk_percent': 0.005, 'max_daily_trades': 3}
        elif balance < 200:
            return {'max_lot': 0.05, 'min_lot': 0.01, 'risk_percent': 0.005, 'max_daily_trades': 5}
        elif balance < 1000:
            return {'max_lot': 0.05, 'min_lot': 0.01, 'risk_percent': 0.01,  'max_daily_trades': 5}
        return {'max_lot': 0.5,  'min_lot': 0.01, 'risk_percent': 0.02,  'max_daily_trades': 10}

    @staticmethod
    def calculate_dynamic_position_size(balance: float, win_rate: float,
                                        atr: float, current_volatility: float) -> float:
        base_risk = Config.RISK_PERCENT
        if win_rate < 0.45:
            base_risk *= 0.5
        elif win_rate > 0.65:
            base_risk *= 1.2
        volatility_factor = max(0.3, min(2.0, 10.0 / (atr + 0.1)))
        return min(base_risk * volatility_factor, 0.02)

    @staticmethod
    def calculate_kelly_lot_size(symbol: str, sl_points: float,
                                 balance: float, atr: float,
                                 win_rate: float, dynamic_risk: float,
                                 quality_score: float = 1.0,
                                 zone: Optional[Dict] = None) -> float:
        try:
            limits       = ImprovedRiskManager.get_dynamic_limits(balance)
            risk_percent = min(dynamic_risk, limits['risk_percent'])

            if Config.ORDER_BLOCK_SL and zone and 'high' in zone and 'low' in zone:
                zone_height      = zone['high'] - zone['low']
                sl_points_ob     = zone_height * 1.15
                sl_points_atr    = atr * 1.4
                sl_points_hybrid = min(sl_points_atr, sl_points_ob)
                logging.info(
                    f"[OB_SL] Hybrid SL: min(ATR×1.4={sl_points_atr:.1f}, "
                    f"OB×1.15={sl_points_ob:.1f}) = {sl_points_hybrid:.1f}"
                )
                sl_points = max(Config.MIN_SL_POINTS, sl_points_hybrid)

            quality_multiplier = (
                1.5 if quality_score >= 2.5 else
                1.2 if quality_score >= 2.2 else
                1.0
            )
            risk_percent *= quality_multiplier
            if win_rate < 0.4:
                risk_percent /= 2

            kelly_fraction = min(
                risk_percent * (1 - atr / (sl_points + 0.1)),
                limits['risk_percent']
            )
            risk_amount = balance * kelly_fraction

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return limits['min_lot']

            tick_value = symbol_info.trade_tick_value
            point      = symbol_info.point
            sl_value   = sl_points * point * tick_value
            if sl_value <= 0:
                return limits['min_lot']

            lot_size = risk_amount / sl_value
            lot_size = min(round(lot_size, 2), symbol_info.volume_max, limits['max_lot'])
            lot_size = max(symbol_info.volume_min, lot_size, limits['min_lot'])

            max_loss = sl_points * point * tick_value * lot_size
            if max_loss > balance * limits['risk_percent']:
                lot_size = (balance * limits['risk_percent']) / (sl_points * point * tick_value)

            return lot_size
        except Exception as exc:
            logging.error(f"[LOT_ERROR] {exc}")
            return Config.MIN_TRADE_COST

    @staticmethod
    def calculate_trade_cost(symbol: str, lot_size: float, order_type: str) -> float:
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return Config.MIN_TRADE_COST
            tick        = mt5.symbol_info_tick(symbol)
            spread      = (tick.ask - tick.bid) / symbol_info.point if tick else 10
            spread_cost = spread * symbol_info.point * symbol_info.trade_tick_value * lot_size
            commission  = Config.COMMISSION_PER_LOT * lot_size
            return max(Config.MIN_TRADE_COST, min(spread_cost + commission, lot_size * 50))
        except Exception as exc:
            logging.error(f"[COST_ERROR] {exc}")
            return Config.MIN_TRADE_COST

    @staticmethod
    def check_drawdown_limit(stats_tracker) -> bool:
        stats = stats_tracker.get_stats()
        if stats['max_drawdown'] > Config.MAX_DRAWDOWN:
            logging.warning(f"[DRAWDOWN_LIMIT] {stats['max_drawdown']:.1%}")
            return False
        if stats['current_balance'] < Config.MIN_BALANCE:
            logging.warning(f"[BALANCE_LOW] ${stats['current_balance']:.2f}")
            return False
        return True
