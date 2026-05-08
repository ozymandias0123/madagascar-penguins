"""
utils/gamma_scanner.py
Gamma Punch Scanner

Scans options chain data for significant gamma exposure levels.
Identifies gamma walls, flip points, and high-OI strike clusters.
Designed for equity/index options (SPY, QQQ, SPX, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Black-Scholes helpers ────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes gamma per 1 contract (100 shares)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d = _d1(S, K, r, sigma, T)
    nd1 = math.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
    return nd1 / (S * sigma * math.sqrt(T))


def bs_delta(S: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d = _d1(S, K, r, sigma, T)
    if option_type.lower() == "call":
        return _norm_cdf(d)
    return _norm_cdf(d) - 1.0


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    strike:      float
    expiry_days: float           # calendar days to expiry
    option_type: str             # "call" or "put"
    open_interest: int
    volume:      int
    iv:          float           # implied volatility (annualised decimal)

    # computed
    gamma:  float = 0.0
    delta:  float = 0.0
    dex:    float = 0.0          # dollar delta exposure
    gex:    float = 0.0          # dollar gamma exposure


@dataclass
class GammaLevel:
    strike:    float
    net_gex:   float             # positive = dealer long gamma (stabilising)
    net_dex:   float
    total_oi:  int
    call_oi:   int
    put_oi:    int
    call_vol:  int
    put_vol:   int
    label:     str = ""          # "WALL", "FLIP", "PIN", ""


@dataclass
class ScanResult:
    underlying:     str
    spot:           float
    gamma_flip:     Optional[float]
    largest_wall:   Optional[float]
    call_wall:      Optional[float]
    put_wall:       Optional[float]
    net_gex_total:  float
    regime:         str                      # "positive" | "negative"
    levels:         List[GammaLevel] = field(default_factory=list)
    top_strikes:    List[GammaLevel] = field(default_factory=list)   # top 5 by |gex|

    def summary(self) -> str:
        lines = [
            f"═══ Gamma Punch Scanner — {self.underlying} ═══",
            f"  Spot          : {self.spot:.2f}",
            f"  Regime        : {self.regime.upper()} GAMMA",
            f"  Net GEX Total : {self.net_gex_total:+,.0f}",
            f"  Gamma Flip    : {self.gamma_flip or 'N/A'}",
            f"  Largest Wall  : {self.largest_wall or 'N/A'}",
            f"  Call Wall     : {self.call_wall or 'N/A'}",
            f"  Put Wall      : {self.put_wall or 'N/A'}",
            "",
            "  Top GEX Strikes:",
        ]
        for lvl in self.top_strikes:
            tag = f"[{lvl.label}]" if lvl.label else ""
            lines.append(
                f"    {lvl.strike:>8.2f}  GEX={lvl.net_gex:+12,.0f}  "
                f"OI={lvl.total_oi:>7,}  {tag}"
            )
        return "\n".join(lines)


# ── Scanner ──────────────────────────────────────────────────────────────────

class GammaPunchScanner:
    """
    Gamma exposure scanner for equity/index options.

    Usage
    -----
    scanner = GammaPunchScanner(spot=450.0, risk_free=0.05)
    scanner.load_chain(contracts)          # list[OptionContract]
    result = scanner.scan(underlying="SPY")
    print(result.summary())
    """

    SHARES_PER_CONTRACT = 100

    def __init__(self, spot: float, risk_free: float = 0.04):
        self.spot = spot
        self.risk_free = risk_free
        self._contracts: List[OptionContract] = []
        self._levels: Dict[float, GammaLevel] = {}

    # ── loading ──────────────────────────────────────────────────────────────

    def load_chain(self, contracts: List[OptionContract]) -> None:
        """Load a list of OptionContract objects and compute greeks."""
        self._contracts = []
        self._levels = {}
        S = self.spot
        r = self.risk_free

        for c in contracts:
            T = c.expiry_days / 365.0
            c.gamma = bs_gamma(S, c.strike, r, c.iv, T)
            c.delta = bs_delta(S, c.strike, r, c.iv, T, c.option_type)
            mult = self.SHARES_PER_CONTRACT * c.open_interest
            c.gex = c.gamma * S * S * mult * 0.01   # GEX in $ per 1% move
            c.dex = c.delta * S * mult
            self._contracts.append(c)
            self._aggregate(c)

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load from a DataFrame with columns:
        strike, expiry_days, option_type, open_interest, volume, iv
        """
        contracts = [
            OptionContract(
                strike=float(row["strike"]),
                expiry_days=float(row["expiry_days"]),
                option_type=str(row["option_type"]).lower(),
                open_interest=int(row["open_interest"]),
                volume=int(row.get("volume", 0)),
                iv=float(row["iv"]),
            )
            for _, row in df.iterrows()
        ]
        self.load_chain(contracts)

    def _aggregate(self, c: OptionContract) -> None:
        k = c.strike
        if k not in self._levels:
            self._levels[k] = GammaLevel(
                strike=k, net_gex=0.0, net_dex=0.0,
                total_oi=0, call_oi=0, put_oi=0,
                call_vol=0, put_vol=0,
            )
        lvl = self._levels[k]
        lvl.total_oi += c.open_interest
        lvl.net_dex  += c.dex
        if c.option_type == "call":
            lvl.net_gex += c.gex          # dealer short call → long gamma
            lvl.call_oi += c.open_interest
            lvl.call_vol += c.volume
        else:
            lvl.net_gex -= c.gex          # dealer short put → short gamma
            lvl.put_oi  += c.open_interest
            lvl.put_vol += c.volume

    # ── scanning ─────────────────────────────────────────────────────────────

    def scan(self, underlying: str = "UNKNOWN") -> ScanResult:
        if not self._levels:
            raise ValueError("No chain loaded. Call load_chain() first.")

        levels = sorted(self._levels.values(), key=lambda l: l.strike)
        self._label_levels(levels)

        net_gex_total = sum(l.net_gex for l in levels)
        regime = "positive" if net_gex_total >= 0 else "negative"

        gamma_flip   = self._find_gamma_flip(levels)
        call_wall    = self._find_wall(levels, side="call")
        put_wall     = self._find_wall(levels, side="put")
        largest_wall = self._find_largest_wall(levels)

        top_strikes = sorted(levels, key=lambda l: abs(l.net_gex), reverse=True)[:5]

        return ScanResult(
            underlying=underlying,
            spot=self.spot,
            gamma_flip=gamma_flip,
            largest_wall=largest_wall,
            call_wall=call_wall,
            put_wall=put_wall,
            net_gex_total=net_gex_total,
            regime=regime,
            levels=levels,
            top_strikes=top_strikes,
        )

    # ── labelling ────────────────────────────────────────────────────────────

    def _label_levels(self, levels: List[GammaLevel]) -> None:
        """Assign WALL / FLIP / PIN labels."""
        if not levels:
            return

        gex_vals = [abs(l.net_gex) for l in levels]
        max_gex  = max(gex_vals) if gex_vals else 1

        # Gamma flip: sign change in cumulative GEX near spot
        cum = 0.0
        prev_sign = None
        for lvl in levels:
            cum += lvl.net_gex
            sign = 1 if cum >= 0 else -1
            if prev_sign is not None and sign != prev_sign:
                lvl.label = "FLIP"
            prev_sign = sign

        # Walls: strikes where |gex| > 60% of max and above/below spot
        for lvl in levels:
            if abs(lvl.net_gex) >= 0.6 * max_gex and not lvl.label:
                lvl.label = "WALL"

        # PIN: highest OI strike closest to spot
        closest = min(levels, key=lambda l: abs(l.strike - self.spot))
        if not closest.label:
            closest.label = "PIN"

    def _find_gamma_flip(self, levels: List[GammaLevel]) -> Optional[float]:
        """Lowest strike above spot where cumulative GEX flips sign."""
        cum = 0.0
        prev_sign = None
        for lvl in sorted(levels, key=lambda l: l.strike):
            cum += lvl.net_gex
            sign = 1 if cum >= 0 else -1
            if prev_sign is not None and sign != prev_sign and lvl.strike >= self.spot:
                return lvl.strike
            prev_sign = sign
        return None

    def _find_wall(self, levels: List[GammaLevel], side: str) -> Optional[float]:
        """Largest GEX strike above spot (call wall) or below spot (put wall)."""
        if side == "call":
            candidates = [l for l in levels if l.strike > self.spot and l.net_gex > 0]
        else:
            candidates = [l for l in levels if l.strike < self.spot and l.net_gex < 0]
        if not candidates:
            return None
        return max(candidates, key=lambda l: abs(l.net_gex)).strike

    def _find_largest_wall(self, levels: List[GammaLevel]) -> Optional[float]:
        if not levels:
            return None
        return max(levels, key=lambda l: abs(l.net_gex)).strike

    # ── convenience ──────────────────────────────────────────────────────────

    def gex_by_strike(self) -> pd.DataFrame:
        """Return a sorted DataFrame of GEX by strike."""
        rows = [
            {
                "strike":    l.strike,
                "net_gex":   l.net_gex,
                "net_dex":   l.net_dex,
                "total_oi":  l.total_oi,
                "call_oi":   l.call_oi,
                "put_oi":    l.put_oi,
                "label":     l.label,
            }
            for l in sorted(self._levels.values(), key=lambda l: l.strike)
        ]
        return pd.DataFrame(rows)

    def key_levels(self, n: int = 5) -> List[float]:
        """Return the n most significant strike levels by |gex|."""
        return [
            l.strike
            for l in sorted(
                self._levels.values(),
                key=lambda l: abs(l.net_gex),
                reverse=True,
            )[:n]
        ]


# ── Quick demo ───────────────────────────────────────────────────────────────

def demo():
    """Quick demo with synthetic chain data."""
    import random
    random.seed(42)

    spot = 450.0
    scanner = GammaPunchScanner(spot=spot, risk_free=0.04)

    contracts = []
    for strike in range(420, 481, 5):
        for days in [7, 14, 30]:
            for opt in ["call", "put"]:
                oi = random.randint(500, 15_000)
                iv = 0.18 + random.uniform(-0.04, 0.04)
                contracts.append(OptionContract(
                    strike=float(strike),
                    expiry_days=float(days),
                    option_type=opt,
                    open_interest=oi,
                    volume=random.randint(0, oi // 3),
                    iv=iv,
                ))

    scanner.load_chain(contracts)
    result = scanner.scan("SPY")
    print(result.summary())


if __name__ == "__main__":
    demo()
