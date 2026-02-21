"""
Rebalancing Engine
==================
The core algorithm. This module is deliberately isolated from all I/O —
no file access, no network calls, no UI framework. It's a pure function:
give it a DataFrame and a cash amount, get back a DataFrame with orders.

This isolation makes it trivially testable:

    >>> import pandas as pd
    >>> from rebalancer.engine import calculate_rebalance
    >>> df = pd.DataFrame([
    ...     {"ticker": "A.NS", "name": "A", "weight": 60, "held": 10,
    ...      "price": 100.0, "restricted": False},
    ...     {"ticker": "B.NS", "name": "B", "weight": 40, "held": 5,
    ...      "price": 200.0, "restricted": False},
    ... ])
    >>> result = calculate_rebalance(df, 10_000)
    >>> result[["ticker", "buy_qty", "buy_cost"]].to_dict("records")

Stock Classification
--------------------
Every stock is placed into exactly one of four categories:

    RESTRICTED   — restricted=True (any weight)  → frozen, no trade
    LIQUIDATION  — weight=0, held>0, not restricted → sell all shares
    ACTIVE       — weight>0, not restricted       → buy to close deficit
    INERT        — weight=0, held=0               → no action

Cash Pool
---------
    available_cash = new_monthly_sip + liquidation_proceeds

Restricted stocks are frozen but their current value IS included in the
total portfolio valuation. This ensures active stocks' targets are computed
in the context of the full portfolio, so when restrictions lift next month,
the deficit algorithm naturally corrects any accumulated drift.
"""

import math
import pandas as pd


def calculate_rebalance(
    portfolio: pd.DataFrame,
    new_cash: float,
) -> pd.DataFrame:
    """
    Compute buy/sell orders to push a portfolio toward its target weights.

    Parameters
    ----------
    portfolio : DataFrame
        Must contain columns: ticker, name, weight, held, price.
        May contain: restricted (defaults to False if missing).
    new_cash : float
        Fresh money to invest this month (e.g., monthly SIP amount).

    Returns
    -------
    DataFrame with all original columns plus:
        category, sell_qty, sell_value, buy_qty, buy_cost,
        new_held, new_value, current_weight, new_weight, drift,
        target_value, deficit, allocation
    """
    df = portfolio.copy()
    df["current_value"] = df["held"] * df["price"]

    # ── Ensure restricted column exists (backward compat) ──
    if "restricted" not in df.columns:
        df["restricted"] = False
    df["restricted"] = df["restricted"].fillna(False).astype(bool)

    # ── Classify each stock ──
    def _classify(row):
        if row["restricted"]:
            return "RESTRICTED"
        if row["weight"] == 0 and row["held"] > 0:
            return "LIQUIDATION"
        if row["weight"] > 0:
            return "ACTIVE"
        return "INERT"

    df["category"] = df.apply(_classify, axis=1)

    # ── Initialize order columns ──
    df["sell_qty"] = 0
    df["sell_value"] = 0.0
    df["buy_qty"] = 0
    df["buy_cost"] = 0.0

    # ── LIQUIDATION: sell all shares ──
    is_liq = df["category"] == "LIQUIDATION"
    df.loc[is_liq, "sell_qty"] = df.loc[is_liq, "held"]
    df.loc[is_liq, "sell_value"] = df.loc[is_liq, "sell_qty"] * df.loc[is_liq, "price"]
    liq_proceeds = df["sell_value"].sum()

    # ── Available cash = SIP + liquidation proceeds ──
    available_cash = new_cash + liq_proceeds

    # ── ACTIVE stocks: deficit-proportional allocation ──
    is_active = df["category"] == "ACTIVE"
    is_restricted = df["category"] == "RESTRICTED"

    # Include restricted value in the total so active targets are accurate
    restricted_value = df.loc[is_restricted, "current_value"].sum()
    active_value = df.loc[is_active, "current_value"].sum()
    total_target = restricted_value + active_value + available_cash

    df["target_value"] = 0.0
    df.loc[is_active, "target_value"] = (
        total_target * (df.loc[is_active, "weight"] / 100.0)
    )

    # Deficit = how much each active stock is below target (clipped to 0)
    df["deficit"] = 0.0
    df.loc[is_active, "deficit"] = (
        df.loc[is_active, "target_value"] - df.loc[is_active, "current_value"]
    ).clip(lower=0)
    total_deficit = df.loc[is_active, "deficit"].sum()

    # Distribute cash proportionally to deficits
    df["allocation"] = 0.0
    if total_deficit > 0:
        df.loc[is_active, "allocation"] = (
            df.loc[is_active, "deficit"] / total_deficit
        ) * available_cash
    else:
        # All at/above target — fall back to weight-proportional split
        ws = df.loc[is_active, "weight"].sum()
        if ws > 0:
            df.loc[is_active, "allocation"] = (
                available_cash * (df.loc[is_active, "weight"] / ws)
            )

    # ── Whole-share buy orders ──
    ok = is_active & (df["price"] > 0)
    df.loc[ok, "buy_qty"] = (
        df.loc[ok, "allocation"] / df.loc[ok, "price"]
    ).apply(math.floor).astype(int)
    df["buy_cost"] = df["buy_qty"] * df["price"]

    # ── Post-rebalance state ──
    df["new_held"] = df["held"] + df["buy_qty"] - df["sell_qty"]
    df["new_value"] = df["new_held"] * df["price"]

    # ── Weight analysis ──
    tc = df["current_value"].sum()
    nt = df["new_value"].sum()
    df["current_weight"] = (df["current_value"] / tc * 100).round(1) if tc > 0 else 0.0
    df["new_weight"] = (df["new_value"] / nt * 100).round(1) if nt > 0 else 0.0
    df["drift"] = (df["current_weight"] - df["weight"]).round(1)

    return df
