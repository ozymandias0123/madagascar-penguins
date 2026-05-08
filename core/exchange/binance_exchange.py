"""
core/exchange/binance_exchange.py — Binance adapter (Futures USD-M).

Supports both testnet and mainnet.
Symbol format: 'BTC/USDT:USDT', 'ETH/USDT:USDT', 'NQ/USD' (not available — use indices via MT5)
"""

from core.exchange.ccxt_exchange import CCXTExchange


class BinanceExchange(CCXTExchange):
    """Binance USD-M Perpetual Futures."""

    name             = "Binance"
    exchange_id      = "binance"
    market_type      = "future"
    default_settle   = "USDT"
    supports_futures = True
    supports_spot    = True


class BinanceSpotExchange(CCXTExchange):
    """Binance Spot market."""

    name             = "BinanceSpot"
    exchange_id      = "binance"
    market_type      = "spot"
    default_settle   = "USDT"
    supports_futures = False
    supports_spot    = True
