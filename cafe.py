"""cafe.py - The Smart Cafe perishable-inventory simulator (backward compatible).

Assignment interface kept exactly:
    env = CafeSim(); s = env.reset(); profit, next_state, done, info = env.step(order)
    info keeps the original keys: order, demand, sold, spoiled, stockout, carry, profit

Non-breaking additions:
    * optional ``seed`` (None = legacy module-level generator, identical behaviour)
    * ``day_economics()`` - the ONE canonical day calculation, reused by the simulator,
      ledger, KPIs, scoreboard and You-vs-AI game
    * extra financial keys in ``info`` (revenue, each cost line, available, leftover ...)
    * helpers for fair evaluation on common demand scenarios (CRN) and diagnostics
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

rng = np.random.default_rng()  # legacy module-level generator (used when seed is None)

PARAM_KEYS = ("price", "order_cost", "holding_cost", "spoilage_cost", "stockout_penalty",
              "spoil_frac", "mean_demand", "busy_chance", "busy_extra", "max_stock", "days",
              "start_stock")
SUM_KEYS = ("order", "demand", "sold", "stockout", "leftover", "spoiled", "revenue",
            "order_cost", "holding_cost", "spoilage_cost", "stockout_penalty", "profit",
            "clipped_units")
Policy = Callable[[int, int], int]  # policy(stock, day_index) -> order


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Division with an explicit default when the denominator is zero."""
    return float(num) / float(den) if den else default


def day_economics(stock: int, order: Any, demand: int, p: Dict[str, float]) -> Dict[str, float]:
    """Canonical one-day transition + economics (assignment steps 1-10).

    Single source of truth for every profit figure in the project.
    """
    max_stock = int(p["max_stock"])
    order = int(np.clip(order, 0, max_stock))                 # 1. integer order 0..max_stock
    available = min(int(stock) + order, max_stock)            # 2. fridge capacity cap
    demand = int(demand)                                      # 3. demand drawn by the caller
    sold = min(available, demand)                             # 4. can only sell what you have
    stockout = max(demand - available, 0)                     # 5. unmet demand, never negative
    leftover = available - sold                               # 6.
    spoiled = int(round(p["spoil_frac"] * leftover))          # 7. Python round() = banker's rounding
    carry = leftover - spoiled                                # 8. next morning's stock
    revenue = sold * p["price"]                               # 9.
    order_cost = order * p["order_cost"]                      # charged on the full order (spec)
    holding = leftover * p["holding_cost"]                    # on all leftover, incl. units that spoil
    spoil_cost = spoiled * p["spoilage_cost"]
    penalty = stockout * p["stockout_penalty"]
    profit = revenue - (order_cost + holding + spoil_cost + penalty)   # 10.
    return dict(start_stock=int(stock), order=order, available=available, demand=demand,
                sold=sold, stockout=stockout, leftover=leftover, spoiled=spoiled, carry=carry,
                revenue=revenue, order_cost=order_cost, holding_cost=holding,
                spoilage_cost=spoil_cost, stockout_penalty=penalty, profit=profit,
                clipped_units=int(stock) + order - available)


class CafeSim:
    """The cafe as an RL environment. State = stock on hand each morning (0..max_stock)."""

    def __init__(self, price: float = 5.0, order_cost: float = 1.0, holding_cost: float = 0.3,
                 spoilage_cost: float = 1.5, stockout_penalty: float = 6.0,
                 spoil_frac: float = 0.25, mean_demand: float = 7, busy_chance: float = 0.20,
                 max_stock: int = 15, days: int = 30, *, start_stock: int = 5,
                 busy_extra: float = 5, seed: Optional[Any] = None) -> None:
        if mean_demand < 0 or busy_extra < 0 or not 0 <= busy_chance <= 1:
            raise ValueError("demand parameters out of range")
        if not 0 <= spoil_frac <= 1 or max_stock < 1 or days < 1:
            raise ValueError("spoil_frac, max_stock or days out of range")
        self.price, self.order_cost, self.holding_cost = price, order_cost, holding_cost
        self.spoilage_cost, self.stockout_penalty = spoilage_cost, stockout_penalty
        self.spoil_frac = spoil_frac
        self.mean_demand, self.busy_chance, self.busy_extra = mean_demand, busy_chance, busy_extra
        self.max_stock, self.days = int(max_stock), int(days)
        self.start_stock = int(min(max(start_stock, 0), max_stock))
        self.rng = rng if seed is None else np.random.default_rng(seed)
        self.stock, self.day = self.start_stock, 0

    def params(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in PARAM_KEYS}

    def reset(self) -> int:
        self.stock, self.day = self.start_stock, 0   # start each month with 5 units on hand
        return self.stock

    def _demand(self, gen: Optional[np.random.Generator] = None) -> int:
        """Poisson(mean_demand) customers, plus Poisson(busy_extra) on a busy day."""
        g = self.rng if gen is None else gen
        d = g.poisson(self.mean_demand)
        if g.random() < self.busy_chance:
            d += g.poisson(self.busy_extra)
        return int(d)

    def sample_demands(self, n_months: int, days: Optional[int] = None,
                       seed: Optional[Any] = None) -> np.ndarray:
        """Pre-generate demand scenarios with the simulator's OWN generator (for fair CRN tests)."""
        g = np.random.default_rng(seed)
        n_days = int(days or self.days)
        return np.array([[self._demand(g) for _ in range(n_days)] for _ in range(n_months)],
                        dtype=int)

    def step(self, order: Any, demand: Optional[int] = None):
        d = self._demand() if demand is None else int(demand)
        r = day_economics(self.stock, order, d, self.params())
        self.stock = int(r["carry"])
        self.day += 1
        done = self.day >= self.days
        info = dict(order=r["order"], demand=r["demand"], sold=r["sold"], spoiled=r["spoiled"],
                    stockout=r["stockout"], carry=r["carry"], profit=round(r["profit"], 2))
        info.update(day=self.day, start_stock=r["start_stock"], available=r["available"],
                    leftover=r["leftover"], revenue=r["revenue"], order_cost=r["order_cost"],
                    holding_cost=r["holding_cost"], spoilage_cost=r["spoilage_cost"],
                    stockout_penalty=r["stockout_penalty"], clipped_units=r["clipped_units"],
                    profit_raw=r["profit"])
        return r["profit"], self.stock, done, info


# ---------------------------------------------------------------- evaluation helpers
def run_month(policy: Policy, demands, p: Dict[str, float],
              start_stock: Optional[int] = None) -> List[Dict]:
    """Play one month on a GIVEN demand sequence. Returns one ledger row per day."""
    stock = int(p["start_stock"] if start_stock is None else start_stock)
    rows = []
    for t, d in enumerate(demands):
        r = day_economics(stock, policy(stock, t), int(d), p)
        r["day"] = t + 1
        rows.append(r)
        stock = int(r["carry"])
    return rows


def totals(rows: List[Dict]) -> Dict[str, float]:
    return {k: float(sum(r[k] for r in rows)) for k in SUM_KEYS}


def kpis(t: Dict[str, float], n_days: int) -> Dict[str, float]:
    """KPI definitions (numerator / denominator):
    service = sold / demand;  stockout rate = unmet / demand;
    spoilage KPI = spoiled / units ordered;  spoil-of-leftover (a DIFFERENT idea) = spoiled / leftover.
    """
    return dict(service=safe_div(t["sold"], t["demand"], 1.0),
                stockout_rate=safe_div(t["stockout"], t["demand"], 0.0),
                spoilage_kpi=safe_div(t["spoiled"], t["order"], 0.0),
                spoil_of_leftover=safe_div(t["spoiled"], t["leftover"], 0.0),
                net_margin=safe_div(t["profit"], t["revenue"], float("nan")),
                profit_per_day=safe_div(t["profit"], n_days, 0.0))


def evaluate(policy: Policy, D: np.ndarray, p: Dict[str, float]) -> Dict[str, Any]:
    """Evaluate a policy on a common demand matrix D (months x days). Ratios use pooled sums."""
    months = [totals(run_month(policy, row, p)) for row in D]
    n = len(months)
    pooled = {k: float(sum(m[k] for m in months)) for k in SUM_KEYS}
    profits = np.array([m["profit"] for m in months])
    out: Dict[str, Any] = kpis(pooled, D.shape[1] * n)
    out.update({f"mean_{k}": pooled[k] / n for k in SUM_KEYS})
    out.update(mean_profit=float(profits.mean()),
               std_profit=float(profits.std(ddof=1)) if n > 1 else 0.0,
               q25=float(np.percentile(profits, 25)), q75=float(np.percentile(profits, 75)),
               p5=float(np.percentile(profits, 5)), loss_share=float((profits < 0).mean()),
               profits=profits, months=n)
    return out


def reachable_states(p: Dict[str, float]) -> Set[int]:
    """Morning stocks that can ever occur: the start stock plus carry(leftover), leftover 0..max."""
    m = int(p["max_stock"])
    return {int(p["start_stock"])} | {l - int(round(p["spoil_frac"] * l)) for l in range(m + 1)}


def state_features(stock: int, p: Dict[str, float], order: Optional[int] = None) -> Dict[str, float]:
    """DIAGNOSTIC ONLY - never fed to the agent. Each is a deterministic alias of stock."""
    mean = float(p["mean_demand"])
    exp_d = mean + p["busy_chance"] * p["busy_extra"]          # model-implied expected demand
    f = dict(utilisation=stock / p["max_stock"],
             base_coverage=safe_div(stock, mean, float("nan")),
             expected_demand=exp_d,
             expected_coverage=safe_div(stock, exp_d, float("nan")),
             remaining_capacity=p["max_stock"] - stock)
    if order is not None:
        f["post_order_inventory"] = min(stock + int(order), int(p["max_stock"]))
    return f
