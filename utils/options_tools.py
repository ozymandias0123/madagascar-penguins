"""
utils/options_tools.py
Options Spread Builder + Earnings Tools

Classes / functions
-------------------
OptionSpread          — generic multi-leg spread container
bull_call_spread()    — long lower call, short upper call
box_spread()          — riskless arbitrage box
butterfly_spread()    — long body, short wings
put_credit_spread()   — short lower put, long even-lower put
calculate_expected_move() — IV-based EM for earnings
earnings_strangle()   — OTM strangle around earnings EM
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Black-Scholes helpers ────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    d1, _ = _d1_d2(S, K, r, sigma, T)
    if option_type.lower() == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, sigma, T)
    nd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return nd1 / (S * sigma * math.sqrt(T))


def bs_theta(S: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    """Theta per calendar day."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    nd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    if option_type.lower() == "call":
        theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * _norm_cdf(d2))
    else:
        theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2))
    return theta / 365.0


def bs_vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Vega per 1% IV move."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, sigma, T)
    nd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return S * math.sqrt(T) * nd1 * 0.01


# ── Leg & Spread ─────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """Single option leg."""
    option_type:  str        # "call" | "put"
    strike:       float
    expiry_days:  float
    quantity:     int        # positive = long, negative = short
    iv:           float
    premium:      float = 0.0   # will be computed if 0

    def compute_premium(self, S: float, r: float = 0.04) -> None:
        T = self.expiry_days / 365.0
        if self.option_type.lower() == "call":
            self.premium = bs_call(S, self.strike, r, self.iv, T)
        else:
            self.premium = bs_put(S, self.strike, r, self.iv, T)

    def greeks(self, S: float, r: float = 0.04) -> dict:
        T = self.expiry_days / 365.0
        sign = self.quantity
        return {
            "delta": sign * bs_delta(S, self.strike, r, self.iv, T, self.option_type),
            "gamma": sign * bs_gamma(S, self.strike, r, self.iv, T),
            "theta": sign * bs_theta(S, self.strike, r, self.iv, T, self.option_type),
            "vega":  sign * bs_vega(S, self.strike, r, self.iv, T),
        }


@dataclass
class OptionSpread:
    """
    Generic multi-leg spread.

    Parameters
    ----------
    name : str
        Spread type label (e.g. "Bull Call Spread")
    legs : list[Leg]
        All legs of the spread
    spot : float
        Underlying price at construction
    contracts : int
        Number of contracts (multiplies net credit/debit by 100)
    """
    name:       str
    legs:       List[Leg]
    spot:       float
    contracts:  int = 1
    risk_free:  float = 0.04

    def __post_init__(self):
        for leg in self.legs:
            if leg.premium == 0.0:
                leg.compute_premium(self.spot, self.risk_free)

    # ── P&L ──────────────────────────────────────────────────────────────────

    @property
    def net_debit(self) -> float:
        """Positive = debit (paid), negative = credit (received)."""
        return sum(leg.quantity * leg.premium for leg in self.legs)

    @property
    def net_credit(self) -> float:
        return -self.net_debit

    @property
    def max_profit(self) -> Optional[float]:
        """Max profit per spread (not per contract)."""
        # computed via payoff scan
        strikes = sorted(set(l.strike for l in self.legs))
        if not strikes:
            return None
        lo, hi = strikes[0] * 0.7, strikes[-1] * 1.3
        prices = np.linspace(lo, hi, 500)
        payoffs = [self._payoff(p) for p in prices]
        return float(max(payoffs))

    @property
    def max_loss(self) -> Optional[float]:
        lo = min(l.strike for l in self.legs) * 0.5
        hi = max(l.strike for l in self.legs) * 1.5
        prices = np.linspace(lo, hi, 500)
        payoffs = [self._payoff(p) for p in prices]
        return float(min(payoffs))

    @property
    def breakevens(self) -> List[float]:
        lo = min(l.strike for l in self.legs) * 0.5
        hi = max(l.strike for l in self.legs) * 1.5
        prices = np.linspace(lo, hi, 2000)
        payoffs = np.array([self._payoff(p) for p in prices])
        beps = []
        for i in range(len(payoffs) - 1):
            if payoffs[i] * payoffs[i + 1] <= 0:
                beps.append(float((prices[i] + prices[i + 1]) / 2))
        return beps

    def _payoff(self, S_exp: float) -> float:
        """Total payoff at expiry for a given underlying price."""
        pnl = -self.net_debit   # initial debit/credit
        for leg in self.legs:
            if leg.option_type.lower() == "call":
                intrinsic = max(S_exp - leg.strike, 0)
            else:
                intrinsic = max(leg.strike - S_exp, 0)
            pnl += leg.quantity * intrinsic
        return pnl

    def payoff_table(self, steps: int = 20) -> list:
        lo = min(l.strike for l in self.legs) * 0.8
        hi = max(l.strike for l in self.legs) * 1.2
        prices = np.linspace(lo, hi, steps)
        return [(round(p, 2), round(self._payoff(p) * self.contracts * 100, 2))
                for p in prices]

    # ── Greeks ───────────────────────────────────────────────────────────────

    def net_greeks(self) -> dict:
        totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for leg in self.legs:
            g = leg.greeks(self.spot, self.risk_free)
            for k in totals:
                totals[k] += g[k]
        return {k: v * self.contracts * 100 for k, v in totals.items()}

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        g = self.net_greeks()
        be_str = ", ".join(f"{b:.2f}" for b in self.breakevens) or "N/A"
        lines = [
            f"═══ {self.name} ═══",
            f"  Spot          : {self.spot:.2f}",
            f"  Contracts     : {self.contracts}",
            f"  Net Debit     : {self.net_debit * self.contracts * 100:+.2f}",
            f"  Max Profit    : {(self.max_profit or 0) * self.contracts * 100:+.2f}",
            f"  Max Loss      : {(self.max_loss  or 0) * self.contracts * 100:+.2f}",
            f"  Breakeven(s)  : {be_str}",
            f"  Net Delta     : {g['delta']:+.4f}",
            f"  Net Gamma     : {g['gamma']:+.6f}",
            f"  Net Theta/day : {g['theta']:+.4f}",
            f"  Net Vega/1%   : {g['vega']:+.4f}",
            "",
            "  Legs:",
        ]
        for leg in self.legs:
            sign = "+" if leg.quantity > 0 else ""
            lines.append(
                f"    {sign}{leg.quantity:+d} {leg.option_type.upper()} "
                f"K={leg.strike:.2f}  "
                f"exp={leg.expiry_days:.0f}d  "
                f"IV={leg.iv*100:.1f}%  "
                f"prem={leg.premium:.2f}"
            )
        return "\n".join(lines)


# ── Spread constructors ───────────────────────────────────────────────────────

def bull_call_spread(
    spot: float,
    lower_strike: float,
    upper_strike: float,
    expiry_days: float,
    iv: float,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """Long lower call, short upper call — limited risk, limited profit."""
    legs = [
        Leg("call", lower_strike, expiry_days, +1, iv),
        Leg("call", upper_strike, expiry_days, -1, iv),
    ]
    return OptionSpread("Bull Call Spread", legs, spot, contracts, risk_free)


def bear_put_spread(
    spot: float,
    upper_strike: float,
    lower_strike: float,
    expiry_days: float,
    iv: float,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """Long upper put, short lower put."""
    legs = [
        Leg("put", upper_strike, expiry_days, +1, iv),
        Leg("put", lower_strike, expiry_days, -1, iv),
    ]
    return OptionSpread("Bear Put Spread", legs, spot, contracts, risk_free)


def put_credit_spread(
    spot: float,
    short_strike: float,
    long_strike: float,
    expiry_days: float,
    iv: float,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """Short put at short_strike, long put at long_strike (lower)."""
    legs = [
        Leg("put", short_strike, expiry_days, -1, iv),
        Leg("put", long_strike,  expiry_days, +1, iv),
    ]
    return OptionSpread("Put Credit Spread", legs, spot, contracts, risk_free)


def call_credit_spread(
    spot: float,
    short_strike: float,
    long_strike: float,
    expiry_days: float,
    iv: float,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """Short call at short_strike, long call at long_strike (higher)."""
    legs = [
        Leg("call", short_strike, expiry_days, -1, iv),
        Leg("call", long_strike,  expiry_days, +1, iv),
    ]
    return OptionSpread("Call Credit Spread", legs, spot, contracts, risk_free)


def butterfly_spread(
    spot: float,
    lower: float,
    middle: float,
    upper: float,
    expiry_days: float,
    iv: float,
    option_type: str = "call",
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """Standard long butterfly: +1 lower, -2 middle, +1 upper."""
    ot = option_type.lower()
    legs = [
        Leg(ot, lower,  expiry_days, +1, iv),
        Leg(ot, middle, expiry_days, -2, iv),
        Leg(ot, upper,  expiry_days, +1, iv),
    ]
    return OptionSpread(f"Butterfly ({ot.title()})", legs, spot, contracts, risk_free)


def box_spread(
    spot: float,
    lower_strike: float,
    upper_strike: float,
    expiry_days: float,
    iv: float,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """
    Riskless box: bull call spread + bear put spread at same strikes.
    Net value at expiry = upper - lower.
    Used to identify mispricing / financing arbitrage.
    """
    legs = [
        Leg("call", lower_strike, expiry_days, +1, iv),
        Leg("call", upper_strike, expiry_days, -1, iv),
        Leg("put",  upper_strike, expiry_days, +1, iv),
        Leg("put",  lower_strike, expiry_days, -1, iv),
    ]
    return OptionSpread("Box Spread", legs, spot, contracts, risk_free)


def iron_condor(
    spot: float,
    put_long: float,
    put_short: float,
    call_short: float,
    call_long: float,
    expiry_days: float,
    iv: float,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """Short strangle hedged with wings."""
    legs = [
        Leg("put",  put_long,   expiry_days, +1, iv),
        Leg("put",  put_short,  expiry_days, -1, iv),
        Leg("call", call_short, expiry_days, -1, iv),
        Leg("call", call_long,  expiry_days, +1, iv),
    ]
    return OptionSpread("Iron Condor", legs, spot, contracts, risk_free)


# ── Earnings helpers ──────────────────────────────────────────────────────────

def calculate_expected_move(
    spot: float,
    iv: float,
    days_to_expiry: float,
    method: str = "1sd",
) -> Tuple[float, float, float]:
    """
    Calculate expected move around an event (earnings).

    Parameters
    ----------
    spot : float
        Current underlying price
    iv : float
        Implied volatility (annualised decimal, e.g. 0.30 for 30%)
    days_to_expiry : float
        Calendar days to expiry
    method : str
        "1sd"  → ±1 standard deviation (68.2% probability range)
        "atm"  → ATM straddle approximation (iv × sqrt(T) × 0.8)

    Returns
    -------
    (em, lower, upper) : tuple[float, float, float]
    """
    T = days_to_expiry / 365.0
    if method == "atm":
        em = spot * iv * math.sqrt(T) * 0.8
    else:  # 1sd
        em = spot * iv * math.sqrt(T)
    lower = spot - em
    upper = spot + em
    return round(em, 2), round(lower, 2), round(upper, 2)


def earnings_strangle(
    spot: float,
    iv: float,
    expiry_days: float,
    em_multiplier: float = 1.0,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """
    OTM strangle sized to the 1-SD expected move.
    Strikes are placed at ± em_multiplier × EM from spot.
    """
    em, lower, upper = calculate_expected_move(spot, iv, expiry_days)
    call_strike = round(spot + em * em_multiplier, 0)
    put_strike  = round(spot - em * em_multiplier, 0)
    legs = [
        Leg("call", call_strike, expiry_days, +1, iv),
        Leg("put",  put_strike,  expiry_days, +1, iv),
    ]
    return OptionSpread(
        f"Earnings Strangle (EM={em:.2f})", legs, spot, contracts, risk_free
    )


def earnings_iron_fly(
    spot: float,
    iv: float,
    expiry_days: float,
    wing_width: float = 5.0,
    contracts: int = 1,
    risk_free: float = 0.04,
) -> OptionSpread:
    """
    Short iron butterfly at ATM, hedged with wings `wing_width` away.
    Typical earnings play when expecting a contained move.
    """
    atm = round(spot, 0)
    call_long = atm + wing_width
    put_long  = atm - wing_width
    legs = [
        Leg("put",  put_long,  expiry_days, +1, iv),
        Leg("put",  atm,       expiry_days, -1, iv),
        Leg("call", atm,       expiry_days, -1, iv),
        Leg("call", call_long, expiry_days, +1, iv),
    ]
    return OptionSpread("Earnings Iron Fly", legs, spot, contracts, risk_free)


# ── Quick demo ───────────────────────────────────────────────────────────────

def demo():
    S   = 450.0
    iv  = 0.25
    exp = 30

    print("─── Bull Call Spread ───")
    s1 = bull_call_spread(S, 450, 460, exp, iv, contracts=5)
    print(s1.summary())

    print("\n─── Put Credit Spread ───")
    s2 = put_credit_spread(S, 440, 430, exp, iv, contracts=3)
    print(s2.summary())

    print("\n─── Butterfly ───")
    s3 = butterfly_spread(S, 440, 450, 460, exp, iv, contracts=2)
    print(s3.summary())

    print("\n─── Earnings Strangle ───")
    em, lo, hi = calculate_expected_move(S, iv, 7)
    print(f"Expected move ±{em:.2f}  [{lo:.2f} – {hi:.2f}]")
    s4 = earnings_strangle(S, iv, 7, contracts=1)
    print(s4.summary())

    print("\n─── Iron Condor ───")
    s5 = iron_condor(S, 425, 435, 465, 475, exp, iv, contracts=2)
    print(s5.summary())


if __name__ == "__main__":
    demo()
