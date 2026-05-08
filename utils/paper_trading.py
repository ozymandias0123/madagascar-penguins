"""
utils/paper_trading.py
Paper Trading Simulator

Simulates order execution, position tracking, and P&L calculation
without connecting to a real broker. Useful for strategy back-testing
and forward-testing in dry-run mode.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING   = "pending"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Order:
    symbol:     str
    side:       OrderSide
    quantity:   float
    order_type: OrderType        = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price:  Optional[float] = None
    strategy:   str              = ""
    pattern_key: str             = ""

    # filled in by simulator
    order_id:   str              = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status:     OrderStatus      = OrderStatus.PENDING
    fill_price: Optional[float]  = None
    fill_time:  Optional[datetime] = None
    commission: float            = 0.0
    notes:      str              = ""


@dataclass
class Position:
    symbol:    str
    side:      OrderSide
    quantity:  float
    avg_entry: float
    strategy:  str    = ""
    pattern_key: str  = ""
    open_time: datetime = field(default_factory=datetime.utcnow)

    # set on close
    close_price: Optional[float]  = None
    close_time:  Optional[datetime] = None
    realised_pnl: float           = 0.0
    commission:   float           = 0.0

    @property
    def is_open(self) -> bool:
        return self.close_price is None

    def unrealised_pnl(self, current_price: float) -> float:
        if self.side == OrderSide.BUY:
            return (current_price - self.avg_entry) * self.quantity
        return (self.avg_entry - current_price) * self.quantity

    def close(self, price: float, commission: float = 0.0) -> None:
        self.close_price  = price
        self.close_time   = datetime.utcnow()
        self.commission  += commission
        if self.side == OrderSide.BUY:
            self.realised_pnl = (price - self.avg_entry) * self.quantity - self.commission
        else:
            self.realised_pnl = (self.avg_entry - price) * self.quantity - self.commission


@dataclass
class TradeRecord:
    symbol:      str
    side:        str
    quantity:    float
    entry_price: float
    exit_price:  float
    pnl:         float
    commission:  float
    duration_s:  float
    strategy:    str
    pattern_key: str
    entry_time:  datetime
    exit_time:   datetime


# ── Simulator ────────────────────────────────────────────────────────────────

class PaperTradingSimulator:
    """
    Paper trading simulator.

    Usage
    -----
    sim = PaperTradingSimulator(initial_balance=10_000, commission_per_lot=7.0)
    order = sim.place_order("EURUSD", OrderSide.BUY, quantity=1.0, strategy="MyStrat")
    sim.tick("EURUSD", bid=1.0950, ask=1.0952)    # price update
    sim.close_position(order.order_id)
    print(sim.summary())
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        commission_per_lot: float = 7.0,   # per standard lot
        slippage_pct: float = 0.0001,      # 0.01% slippage
    ):
        self.balance           = initial_balance
        self._initial_balance  = initial_balance
        self.commission_per_lot = commission_per_lot
        self.slippage_pct      = slippage_pct

        self._orders:    Dict[str, Order]    = {}
        self._positions: Dict[str, Position] = {}   # order_id → Position
        self._history:   List[TradeRecord]   = []
        self._prices:    Dict[str, float]    = {}   # symbol → mid price

    # ── market data ──────────────────────────────────────────────────────────

    def tick(self, symbol: str, bid: float, ask: float) -> None:
        """Update current market price."""
        self._prices[symbol] = (bid + ask) / 2.0
        self._check_pending_orders(symbol, bid, ask)

    def set_price(self, symbol: str, price: float) -> None:
        """Set mid price directly (for back-test loops)."""
        self._prices[symbol] = price

    # ── orders ───────────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price:  Optional[float] = None,
        strategy: str = "",
        pattern_key: str = "",
    ) -> Order:
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            strategy=strategy,
            pattern_key=pattern_key,
        )
        self._orders[order.order_id] = order

        if order_type == OrderType.MARKET:
            price = self._prices.get(symbol)
            if price is None:
                order.status = OrderStatus.REJECTED
                order.notes  = "No price available for market order"
                return order
            fill = self._apply_slippage(price, side)
            self._fill_order(order, fill)

        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _fill_order(self, order: Order, fill_price: float) -> None:
        commission = order.quantity * self.commission_per_lot
        cost = fill_price * order.quantity + (commission if order.side == OrderSide.BUY else 0)

        if order.side == OrderSide.BUY and self.balance < cost:
            order.status = OrderStatus.REJECTED
            order.notes  = f"Insufficient balance ({self.balance:.2f} < {cost:.2f})"
            return

        order.fill_price = fill_price
        order.fill_time  = datetime.utcnow()
        order.commission = commission
        order.status     = OrderStatus.FILLED

        pos = Position(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            avg_entry=fill_price,
            strategy=order.strategy,
            pattern_key=order.pattern_key,
            commission=commission,
        )
        self._positions[order.order_id] = pos

        if order.side == OrderSide.BUY:
            self.balance -= cost
        else:
            self.balance -= commission   # short: commission upfront

    def _check_pending_orders(self, symbol: str, bid: float, ask: float) -> None:
        for order in self._orders.values():
            if order.status != OrderStatus.PENDING or order.symbol != symbol:
                continue
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and ask <= (order.limit_price or 0):
                    self._fill_order(order, ask)
                elif order.side == OrderSide.SELL and bid >= (order.limit_price or 0):
                    self._fill_order(order, bid)
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and ask >= (order.stop_price or 0):
                    self._fill_order(order, ask)
                elif order.side == OrderSide.SELL and bid <= (order.stop_price or 0):
                    self._fill_order(order, bid)

    # ── positions ────────────────────────────────────────────────────────────

    def close_position(self, order_id: str, price: Optional[float] = None) -> Optional[TradeRecord]:
        pos = self._positions.get(order_id)
        if pos is None or not pos.is_open:
            return None

        if price is None:
            price = self._prices.get(pos.symbol)
        if price is None:
            return None

        fill = self._apply_slippage(price, OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY)
        commission = pos.quantity * self.commission_per_lot
        pos.close(fill, commission)

        if pos.side == OrderSide.BUY:
            self.balance += fill * pos.quantity - commission
        else:
            gross = (pos.avg_entry - fill) * pos.quantity
            self.balance += gross - commission

        duration = (pos.close_time - pos.open_time).total_seconds()
        rec = TradeRecord(
            symbol=pos.symbol,
            side=pos.side.value,
            quantity=pos.quantity,
            entry_price=pos.avg_entry,
            exit_price=fill,
            pnl=pos.realised_pnl,
            commission=pos.commission,
            duration_s=duration,
            strategy=pos.strategy,
            pattern_key=pos.pattern_key,
            entry_time=pos.open_time,
            exit_time=pos.close_time,
        )
        self._history.append(rec)
        return rec

    def close_all(self) -> List[TradeRecord]:
        records = []
        for oid in list(self._positions.keys()):
            r = self.close_position(oid)
            if r:
                records.append(r)
        return records

    # ── portfolio snapshot ────────────────────────────────────────────────────

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.is_open]

    def equity(self) -> float:
        upnl = sum(
            p.unrealised_pnl(self._prices[p.symbol])
            for p in self.open_positions
            if p.symbol in self._prices
        )
        return self.balance + upnl

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._history:
            return {"trades": 0, "win_rate": 0.0, "avg_pnl": 0.0,
                    "total_pnl": 0.0, "max_dd": 0.0, "profit_factor": 0.0}
        pnls  = [r.pnl for r in self._history]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses)) or 1e-9

        # drawdown
        eq_curve = []
        bal = self._initial_balance
        for r in self._history:
            bal += r.pnl
            eq_curve.append(bal)
        peak = self._initial_balance
        max_dd = 0.0
        for e in eq_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "trades":        len(pnls),
            "win_rate":      len(wins) / len(pnls),
            "avg_pnl":       sum(pnls) / len(pnls),
            "total_pnl":     sum(pnls),
            "max_dd":        max_dd,
            "profit_factor": gross_profit / gross_loss,
            "best_trade":    max(pnls),
            "worst_trade":   min(pnls),
        }

    def summary(self) -> str:
        s = self.stats()
        lines = [
            "═══ Paper Trading Summary ═══",
            f"  Initial Balance : {self._initial_balance:,.2f}",
            f"  Current Balance : {self.balance:,.2f}",
            f"  Equity          : {self.equity():,.2f}",
            f"  Total P&L       : {s['total_pnl']:+,.2f}",
            f"  Trades          : {s['trades']}",
            f"  Win Rate        : {s['win_rate']*100:.1f}%",
            f"  Avg P&L/Trade   : {s['avg_pnl']:+,.2f}",
            f"  Profit Factor   : {s['profit_factor']:.2f}",
            f"  Max Drawdown    : {s['max_dd']*100:.1f}%",
        ]
        if s["trades"] > 0:
            lines += [
                f"  Best Trade      : {s['best_trade']:+,.2f}",
                f"  Worst Trade     : {s['worst_trade']:+,.2f}",
            ]
        if self.open_positions:
            lines.append(f"\n  Open Positions  : {len(self.open_positions)}")
            for p in self.open_positions:
                mp = self._prices.get(p.symbol, p.avg_entry)
                upnl = p.unrealised_pnl(mp)
                lines.append(
                    f"    {p.symbol} {p.side.value} {p.quantity} "
                    f"@ {p.avg_entry:.5f}  uPnL={upnl:+.2f}"
                )
        return "\n".join(lines)

    def trade_log(self) -> List[dict]:
        return [
            {
                "symbol":      r.symbol,
                "side":        r.side,
                "qty":         r.quantity,
                "entry":       r.entry_price,
                "exit":        r.exit_price,
                "pnl":         round(r.pnl, 2),
                "commission":  round(r.commission, 2),
                "duration_s":  round(r.duration_s, 0),
                "strategy":    r.strategy,
                "pattern_key": r.pattern_key,
                "entry_time":  r.entry_time.isoformat(),
                "exit_time":   r.exit_time.isoformat() if r.exit_time else "",
            }
            for r in self._history
        ]
