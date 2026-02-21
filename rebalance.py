"""
Portfolio Rebalancer — Panel + yFinance
========================================
A SIP-style portfolio rebalancer that fetches live NSE prices.
Supports buy-only rebalancing AND liquidation (set weight to 0%).

Setup:
    pip install panel yfinance pandas

Run:
    panel serve rebalancer.py --show
    # or just:  python rebalancer.py
"""

import json
import math
from datetime import datetime
from pathlib import Path

import panel as pn
import yfinance as yf
import pandas as pd

pn.extension("tabulator", sizing_mode="stretch_width")

# =============================================================================
# 1. TICKER DISPLAY NAMES
#
#    Names are fetched from yFinance the first time we see a ticker, then
#    cached both in-memory (_name_cache) and on-disk (state file) so that
#    subsequent startups don't require any API calls just for display names.
# =============================================================================

_name_cache: dict[str, str] = {}


def name_for_ticker(ticker: str) -> str:
    """
    Get display name for a ticker. Check the in-memory cache first, then
    call yFinance if needed. The result is cached so each ticker only ever
    triggers one API call across the lifetime of the process.
    """
    if ticker in _name_cache:
        return _name_cache[ticker]
    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName")
        if name:
            _name_cache[ticker] = name
            return name
    except Exception:
        pass
    # Fallback: strip the exchange suffix and use the raw symbol
    fallback = ticker.replace(".NS", "").replace(".BO", "")
    _name_cache[ticker] = fallback
    return fallback

# =============================================================================
# 2. PRICE FETCHING
#
#    Uses yf.download() for batch fetching (one HTTP call for all tickers)
#    which is dramatically faster than individual Ticker().info calls.
#    Falls back to per-ticker fetching only if the batch approach fails.
# =============================================================================


def fetch_all_prices(tickers: list[str]) -> dict[str, float | None]:
    """
    Fetch prices for multiple tickers in a single batch call via yf.download().
    Returns {symbol: price_or_None}.

    The batch approach is ~5-10x faster than sequential per-ticker calls because
    yFinance sends one HTTP request to Yahoo's API for all symbols at once.
    """
    prices: dict[str, float | None] = {t: None for t in tickers}
    if not tickers:
        return prices

    try:
        # yf.download returns a DataFrame with MultiIndex columns when
        # multiple tickers are requested: (field, ticker).
        # For a single ticker it returns simple columns, so we handle both.
        df = yf.download(tickers, period="5d", progress=False, threads=True)

        if df.empty:
            return prices

        for t in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker: columns are just ["Open","High",...,"Close",...]
                    close_series = df["Close"]
                else:
                    # Multiple tickers: columns are MultiIndex (field, ticker)
                    close_series = df["Close"][t]

                # Drop NaN and take the most recent valid close
                valid = close_series.dropna()
                if not valid.empty:
                    prices[t] = round(float(valid.iloc[-1]), 2)
            except (KeyError, IndexError):
                continue

    except Exception as e:
        print(
            f"[WARN] Batch download failed: {e}. Falling back to per-ticker fetch.")
        # Fallback: fetch one at a time (slower but more resilient)
        for t in tickers:
            prices[t] = _fetch_price_single(t)

    return prices


def _fetch_price_single(ticker_symbol: str) -> float | None:
    """
    Fallback: fetch price for a single ticker via Ticker.info, then history.
    Only used if the batch yf.download() fails entirely.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price and price > 0:
            return round(price, 2)
        hist = tk.history(period="5d")
        if not hist.empty:
            return round(hist["Close"].iloc[-1], 2)
    except Exception as e:
        print(f"[WARN] Could not fetch price for {ticker_symbol}: {e}")
    return None

# =============================================================================
# 3. REBALANCING LOGIC (with liquidation support)
#
#    Two cases handled:
#
#    ACTIVE stocks (weight > 0):
#      The algorithm calculates how far each is from its target allocation
#      and directs new money toward the most underweight ones. Buy-only for
#      active stocks — we never sell them, just stop buying overweight ones.
#
#    LIQUIDATION stocks (weight == 0 AND held > 0):
#      Setting weight to 0% while holding shares is the signal to exit.
#      All held shares become a SELL order, and the proceeds join the cash
#      pool available for buying active stocks.
#
#    Cash pool = new_monthly_sip + sum(liquidation_proceeds)
# =============================================================================


def calculate_rebalance(
    portfolio: pd.DataFrame,   # columns: ticker, name, weight, held, price
    new_cash: float,
) -> pd.DataFrame:
    """
    Given a portfolio and new cash to invest, return a DataFrame with buy AND
    sell orders that push allocations toward target weights.
    """
    df = portfolio.copy()
    df["current_value"] = df["held"] * df["price"]

    # ── Step 1: Identify liquidations (weight=0 AND actually holding shares) ──
    # Stocks with weight=0 and held=0 are inert — no action needed on them.
    is_liquidation = (df["weight"] == 0) & (df["held"] > 0)
    df["sell_qty"] = 0
    df.loc[is_liquidation, "sell_qty"] = df.loc[is_liquidation, "held"]
    df["sell_value"] = df["sell_qty"] * df["price"]
    liquidation_proceeds = df["sell_value"].sum()

    # ── Step 2: Available cash = fresh SIP + recovered liquidation money ──
    available_cash = new_cash + liquidation_proceeds

    # ── Step 3: Rebalance only active stocks (weight > 0) ──
    active = df["weight"] > 0
    active_current_value = df.loc[active, "current_value"].sum()
    total_target = active_current_value + available_cash

    df["target_value"] = 0.0
    df.loc[active, "target_value"] = total_target * \
        (df.loc[active, "weight"] / 100.0)

    # Deficit: how much each active stock is underweight. Overweight stocks
    # get deficit=0 — we never sell active stocks, we just don't buy more.
    df["deficit"] = 0.0
    df.loc[active, "deficit"] = (
        df.loc[active, "target_value"] - df.loc[active, "current_value"]
    ).clip(lower=0)
    total_deficit = df.loc[active, "deficit"].sum()

    # Distribute available_cash proportionally to deficits
    df["allocation"] = 0.0
    if total_deficit > 0:
        df.loc[active, "allocation"] = (
            df.loc[active, "deficit"] / total_deficit
        ) * available_cash
    else:
        # Edge case: all active stocks are at or above target weight.
        # Split by target weights as a fallback.
        active_weight_sum = df.loc[active, "weight"].sum()
        if active_weight_sum > 0:
            df.loc[active, "allocation"] = (
                available_cash * (df.loc[active, "weight"] / active_weight_sum)
            )

    # ── Step 4: Convert allocations to whole-share buy orders ──
    df["buy_qty"] = 0
    valid_buy = active & (df["price"] > 0)
    df.loc[valid_buy, "buy_qty"] = (
        df.loc[valid_buy, "allocation"] / df.loc[valid_buy, "price"]
    ).apply(math.floor).astype(int)
    df["buy_cost"] = df["buy_qty"] * df["price"]

    # ── Step 5: Post-rebalance state ──
    df["new_held"] = df["held"] + df["buy_qty"] - df["sell_qty"]
    df["new_value"] = df["new_held"] * df["price"]

    # ── Step 6: Weight analysis ──
    total_current = df["current_value"].sum()
    new_total = df["new_value"].sum()

    df["current_weight"] = 0.0
    if total_current > 0:
        df["current_weight"] = (df["current_value"] /
                                total_current * 100).round(1)

    df["new_weight"] = 0.0
    if new_total > 0:
        df["new_weight"] = (df["new_value"] / new_total * 100).round(1)

    df["drift"] = (df["current_weight"] - df["weight"]).round(1)

    return df

# =============================================================================
# 4. STATE PERSISTENCE
#
#    Design:
#    - Each rebalance run is saved as a dated JSON file in ./state/
#    - A separate "applied" flag is written ONLY when the user clicks Apply
#    - On startup, we load the latest file that was actually applied
#    - Ticker names are persisted in state to avoid API calls on startup
#    - Old state files are retained for audit/history (configurable cleanup)
#
#    File structure:
#    {
#      "timestamp": "...",
#      "applied": false,          <-- flipped to true on Apply
#      "amount": 10000,
#      "portfolio_before": [...],
#      "rebalance_result": [...],
#      "summary": {...},
#      "ticker_names": {"RELIANCE.NS": "Reliance Industries", ...}
#    }
# =============================================================================


STATE_DIR = Path(__file__).resolve().parent / "state"
MAX_STATE_FILES = 50  # Keep the last N state files, delete older ones


def _to_jsonable(v):
    """Convert pandas/numpy types to JSON-safe Python types."""
    if hasattr(v, "item"):  # numpy scalar (int64, float64, etc.)
        v = v.item()
    if isinstance(v, float):
        # NaN check: NaN != NaN in IEEE 754. Convert to None for JSON.
        return None if v != v else round(v, 4)
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v)
    return v


def save_rebalance_state(
    amount: float,
    portfolio: list[dict],
    result: pd.DataFrame,
    applied: bool = False,
) -> Path:
    """
    Persist rebalance data to a dated JSON file. The 'applied' flag indicates
    whether the user has confirmed execution of the trades. On startup, we
    only restore state from files where applied=True.
    """
    STATE_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = STATE_DIR / f"rebalance_{ts}.json"

    result_records = [
        {k: _to_jsonable(v) for k, v in row.to_dict().items()}
        for _, row in result.iterrows()
    ]

    liquidation_total = float(result["sell_value"].sum())

    # Persist ticker names so startup doesn't need API calls
    ticker_names = {d["ticker"]: _name_cache.get(
        d["ticker"], d.get("name", "")) for d in portfolio}

    data = {
        "timestamp": datetime.now().isoformat(),
        "applied": applied,
        "amount": amount,
        "portfolio_before": portfolio,
        "rebalance_result": result_records,
        "summary": {
            "current_value": float(result["current_value"].sum()),
            "total_invested": float(result["buy_cost"].sum()),
            "liquidation_proceeds": liquidation_total,
            "available_cash": amount + liquidation_total,
            "leftover": float(
                (amount + liquidation_total) - result["buy_cost"].sum()
            ),
        },
        "ticker_names": ticker_names,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    # Housekeeping: keep only the most recent N state files
    _cleanup_old_state_files()

    return filepath


def mark_state_applied(filepath: Path):
    """
    Flip the 'applied' flag to True in an existing state file. Called when
    the user clicks Apply, so we know this state represents real trades.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        data["applied"] = True
        data["applied_at"] = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not mark state as applied: {e}")


def load_latest_state() -> tuple[list[dict], float] | None:
    """
    Load the most recent APPLIED rebalance state. We deliberately skip files
    where applied=False, because those represent calculations the user never
    confirmed — loading them would show phantom trades that never happened.

    Returns (portfolio, amount) or None if no applied state exists.
    """
    if not STATE_DIR.exists():
        return None

    # Sort by filename (which embeds the timestamp) in reverse chronological order
    files = sorted(STATE_DIR.glob("rebalance_*.json"), reverse=True)

    for filepath in files:
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Skip un-applied states — they represent "what-if" calculations
        if not data.get("applied", False):
            continue

        result = data.get("rebalance_result", [])
        if not result:
            continue

        # Restore ticker name cache from state file to avoid API calls
        saved_names = data.get("ticker_names", {})
        for ticker, name in saved_names.items():
            if name and ticker not in _name_cache:
                _name_cache[ticker] = name

        # Reconstruct portfolio from the post-rebalance state
        portfolio = []
        for r in result:
            ticker = r["ticker"]
            portfolio.append({
                "ticker": ticker,
                "name": _name_cache.get(ticker, ticker.replace(".NS", "")),
                "weight": r["weight"],
                "held": int(r.get("new_held", r.get("held", 0))),
                "price": float(r.get("price", 0)),
            })

        amount = float(data.get("amount", 10_000))
        return (portfolio, amount)

    return None


def _cleanup_old_state_files():
    """Remove state files beyond the MAX_STATE_FILES limit (oldest first)."""
    if not STATE_DIR.exists():
        return
    files = sorted(STATE_DIR.glob("rebalance_*.json"))
    if len(files) > MAX_STATE_FILES:
        for old_file in files[: len(files) - MAX_STATE_FILES]:
            try:
                old_file.unlink()
            except OSError:
                pass

# =============================================================================
# 5. PANEL UI — widgets, callbacks, and layout
# =============================================================================


DEFAULT_PORTFOLIO = [
    {"ticker": "IDFCFIRSTB.NS", "weight": 20.0, "held": 0, "price": 0.0},
    {"ticker": "FIVESTAR.NS", "weight": 15.0, "held": 0, "price": 0.0},
    {"ticker": "HDFCBANK.NS", "weight": 15.0, "held": 0, "price": 0.0},
    {"ticker": "INFY.NS", "weight": 12.0, "held": 0, "price": 0.0},
    {"ticker": "TCS.NS", "weight": 10.0, "held": 0, "price": 0.0},
    {"ticker": "BHARTIARTL.NS", "weight": 10.0, "held": 0, "price": 0.0},
    {"ticker": "SBIN.NS", "weight": 8.0, "held": 0, "price": 0.0},
    {"ticker": "HINDUNILVR.NS", "weight": 10.0, "held": 0, "price": 0.0},
]


def _portfolio_with_names(rows: list[dict]) -> list[dict]:
    """Add 'name' derived from ticker to each portfolio row."""
    return [{**r, "name": name_for_ticker(r["ticker"])} for r in rows]


# ---- Widgets ----
monthly_input = pn.widgets.FloatInput(
    name="Monthly Investment (₹)", value=10_000, step=1000, start=0, width=220,
)
ticker_input = pn.widgets.TextInput(
    name="Add Ticker", placeholder="e.g. TATAMOTORS.NS", width=250,
)
weight_input = pn.widgets.FloatInput(
    name="Weight %", value=10.0, step=1.0, start=0, end=100, width=100,
)
held_input = pn.widgets.IntInput(
    name="Shares Held", value=0, step=1, start=0, width=100,
)
add_btn = pn.widgets.Button(
    name="➕ Add to Portfolio", button_type="primary", width=180,
)

# Initialize portfolio: try loading persisted state, else use defaults.
# Because load_latest_state now restores _name_cache from the state file,
# this does NOT make any yFinance API calls if a state file exists.
portfolio_data: list[dict] = []
_loaded = load_latest_state()
if _loaded:
    portfolio_data = _loaded[0]
    monthly_input.value = _loaded[1]
else:
    portfolio_data = _portfolio_with_names(DEFAULT_PORTFOLIO)

remove_input = pn.widgets.TextInput(
    name="Remove Ticker", placeholder="e.g. TATAMOTORS.NS", width=250,
)
remove_btn = pn.widgets.Button(
    name="🗑️ Remove", button_type="danger", width=120,
)
fetch_btn = pn.widgets.Button(
    name="📡 Fetch Live Prices", button_type="success", width=200,
)
calc_btn = pn.widgets.Button(
    name="🎯 Calculate Rebalance", button_type="warning", width=220,
)
apply_btn = pn.widgets.Button(
    name="✅ Apply Orders to Holdings", button_type="primary", width=250,
)

status_pane = pn.pane.Alert(
    "Ready. Add stocks and fetch prices to begin.", alert_type="info",
)
portfolio_table = pn.widgets.Tabulator(
    pd.DataFrame(portfolio_data),
    layout="fit_data_stretch",
    theme="midnight",
    height=320,
    selectable=False,
    show_index=False,
    titles={
        "ticker": "Ticker", "name": "Name", "weight": "Target Wt %",
        "held": "Shares Held", "price": "Price ₹",
    },
    editors={
        "weight": {"type": "number", "step": 0.5, "min": 0, "max": 100},
        "held":   {"type": "number", "step": 1, "min": 0},
        "price":  {"type": "number", "step": 0.05, "min": 0},
    },
    frozen_columns=["ticker"],
)

result_pane = pn.pane.HTML(
    "<i>No rebalance computed yet.</i>", sizing_mode="stretch_width",
)

# Global state for tracking the last calculation (so Apply can reference it)
last_result_df: pd.DataFrame | None = None
last_saved_path: Path | None = None     # track which file to mark as applied


def refresh_portfolio_table():
    """Push current portfolio_data into the Tabulator widget."""
    portfolio_table.value = pd.DataFrame(portfolio_data)


def sync_edits_from_table():
    """
    Read back any manual edits the user made directly in the Tabulator cells.
    This is called before every operation that reads portfolio_data to ensure
    inline table edits are never silently lost.
    """
    edited = portfolio_table.value
    if edited is None or edited.empty:
        return
    for i, row in edited.iterrows():
        if i < len(portfolio_data):
            portfolio_data[i]["weight"] = float(
                row.get("weight", portfolio_data[i]["weight"]))
            portfolio_data[i]["held"] = int(
                row.get("held", portfolio_data[i]["held"]))
            portfolio_data[i]["price"] = float(
                row.get("price", portfolio_data[i]["price"]))


# ---- Callbacks ----

def on_add_stock(event):
    sync_edits_from_table()
    ticker = ticker_input.value.strip().upper()
    if not ticker:
        status_pane.object = "⚠️ Please enter a ticker symbol."
        status_pane.alert_type = "warning"
        return
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker += ".NS"
    if any(d["ticker"] == ticker for d in portfolio_data):
        status_pane.object = f"⚠️ {ticker} is already in your portfolio."
        status_pane.alert_type = "warning"
        return
    name = name_for_ticker(ticker)
    portfolio_data.append({
        "ticker": ticker, "name": name,
        "weight": weight_input.value, "held": held_input.value, "price": 0.0,
    })
    refresh_portfolio_table()
    ticker_input.value = ""
    total_w = sum(d["weight"] for d in portfolio_data)
    status_pane.object = f"✅ Added {ticker}. Total weight: {total_w:.1f}%"
    status_pane.alert_type = "success"


def on_remove_stock(event):
    sync_edits_from_table()
    ticker = remove_input.value.strip().upper()
    before = len(portfolio_data)
    portfolio_data[:] = [d for d in portfolio_data if d["ticker"] != ticker]
    if len(portfolio_data) < before:
        refresh_portfolio_table()
        remove_input.value = ""
        status_pane.object = f"🗑️ Removed {ticker}."
        status_pane.alert_type = "info"
    else:
        status_pane.object = f"⚠️ {ticker} not found in portfolio."
        status_pane.alert_type = "warning"


def on_fetch_prices(event):
    sync_edits_from_table()
    status_pane.object = "📡 Fetching prices... this may take a few seconds."
    status_pane.alert_type = "info"

    tickers = [d["ticker"] for d in portfolio_data]
    prices = fetch_all_prices(tickers)

    failed = []
    for d in portfolio_data:
        p = prices.get(d["ticker"])
        if p is not None:
            d["price"] = p
        else:
            failed.append(d["ticker"])

    refresh_portfolio_table()

    if failed:
        status_pane.object = (
            f"⚠️ Prices updated, but failed for: {', '.join(failed)}. "
            "You can edit prices manually in the table."
        )
        status_pane.alert_type = "warning"
    else:
        status_pane.object = (
            f"✅ All {len(tickers)} prices fetched successfully "
            f"at {datetime.now():%H:%M:%S}."
        )
        status_pane.alert_type = "success"


def on_calculate(event):
    global last_result_df, last_saved_path

    sync_edits_from_table()

    # ── Validation ──
    # Only active stocks (weight > 0) must sum to 100%.
    # Stocks at 0% are either liquidation candidates or inert.
    active_weights = [d["weight"] for d in portfolio_data if d["weight"] > 0]
    total_active_w = sum(active_weights)
    if abs(total_active_w - 100) > 0.1:
        status_pane.object = (
            f"⚠️ Active weights (excluding 0% stocks) sum to "
            f"{total_active_w:.1f}%, must be 100%."
        )
        status_pane.alert_type = "danger"
        return

    # Price is required for any stock that is either active (weight>0)
    # or a liquidation candidate (weight=0 but held>0, because we need
    # the price to compute sale proceeds). Stocks with weight=0 and
    # held=0 are inert and don't need a price.
    needs_price = [
        d["ticker"] for d in portfolio_data
        if d["price"] <= 0 and (d["weight"] > 0 or d["held"] > 0)
    ]
    if needs_price:
        status_pane.object = (
            f"⚠️ Missing prices for: {', '.join(needs_price)}. "
            "Fetch or enter manually."
        )
        status_pane.alert_type = "danger"
        return

    # ── Run rebalance ──
    df = pd.DataFrame(portfolio_data)
    result = calculate_rebalance(df, monthly_input.value)
    last_result_df = result

    # ── Save state (applied=False — not yet confirmed by user) ──
    saved_path = save_rebalance_state(
        amount=monthly_input.value,
        portfolio=[{**d} for d in portfolio_data],
        result=result,
        applied=False,
    )
    last_saved_path = saved_path

    # ── Build HTML results ──
    total_invested = result["buy_cost"].sum()
    current_val = result["current_value"].sum()
    liquidation_val = result["sell_value"].sum()
    available_cash = monthly_input.value + liquidation_val
    leftover = available_cash - total_invested

    n_sells = int((result["sell_qty"] > 0).sum())
    n_buys = int((result["buy_qty"] > 0).sum())

    # ── Summary cards ──
    liquidation_card = ""
    if liquidation_val > 0:
        liquidation_card = f"""
        <div style="background:#2a1a1a; padding:12px 18px; border-radius:8px;
                    min-width:150px; border:1px solid #7f1d1d;">
          <div style="font-size:11px; color:#f87171; text-transform:uppercase;">
            Liquidation Proceeds ({n_sells} stock{'s' if n_sells != 1 else ''})
          </div>
          <div style="font-size:20px; font-weight:700; color:#f87171;">
            ₹{liquidation_val:,.0f}
          </div>
        </div>
        """

    summary_html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px;">
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:150px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Current Portfolio Value</div>
        <div style="font-size:20px; font-weight:700; color:#60a5fa;">₹{current_val:,.0f}</div>
      </div>
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:150px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Fresh SIP Money</div>
        <div style="font-size:20px; font-weight:700; color:#34d399;">₹{monthly_input.value:,.0f}</div>
      </div>
      {liquidation_card}
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:150px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Total Available Cash</div>
        <div style="font-size:20px; font-weight:700; color:#a78bfa;">₹{available_cash:,.0f}</div>
      </div>
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:150px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Actually Deployed</div>
        <div style="font-size:20px; font-weight:700; color:#fbbf24;">₹{total_invested:,.0f}</div>
      </div>
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:150px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Uninvested Cash</div>
        <div style="font-size:20px; font-weight:700; color:#f87171;">₹{leftover:,.0f}</div>
      </div>
    </div>
    """

    # ── Orders table ──
    rows_html = ""
    for _, r in result.iterrows():
        is_sell = int(r["sell_qty"]) > 0
        is_buy = int(r["buy_qty"]) > 0

        row_bg = ""
        if is_sell:
            row_bg = "background:#2a1015;"
        elif is_buy:
            row_bg = "background:#0d1a0d;"

        if is_sell:
            action = (
                f'<span style="color:#f87171; font-weight:700;">'
                f'🔴 SELL {int(r["sell_qty"])}</span>'
            )
            cashflow = f'<span style="color:#f87171;">+₹{r["sell_value"]:,.0f}</span>'
        elif is_buy:
            action = (
                f'<span style="color:#34d399; font-weight:700;">'
                f'🟢 BUY {int(r["buy_qty"])}</span>'
            )
            cashflow = f'<span style="color:#60a5fa;">−₹{r["buy_cost"]:,.0f}</span>'
        else:
            action = '<span style="color:#666;">—</span>'
            cashflow = '<span style="color:#666;">—</span>'

        drift_val = r["drift"]
        drift_color = "#f87171" if drift_val < - \
            1 else "#34d399" if drift_val > 1 else "#888"
        nw_color = "#34d399" if abs(
            r["new_weight"] - r["weight"]) < 2 else "#fbbf24"

        rows_html += f"""
        <tr style="{row_bg}">
          <td style="padding:8px 10px; font-weight:600;">{r['ticker']}</td>
          <td style="padding:8px 10px;">{r['name']}</td>
          <td style="padding:8px 10px;">₹{r['price']:,.2f}</td>
          <td style="padding:8px 10px; color:#fbbf24;">{r['weight']:.1f}%</td>
          <td style="padding:8px 10px;">{r['current_weight']:.1f}%</td>
          <td style="padding:8px 10px; color:{drift_color};">
            {'+' if drift_val > 0 else ''}{drift_val:.1f}%</td>
          <td style="padding:8px 10px; font-size:14px;">{action}</td>
          <td style="padding:8px 10px;">{cashflow}</td>
          <td style="padding:8px 10px;">{int(r['new_held'])}</td>
          <td style="padding:8px 10px; color:{nw_color};">{r['new_weight']:.1f}%</td>
        </tr>
        """

    headers = [
        "Ticker", "Name", "Price", "Target Wt", "Current Wt",
        "Drift", "Action", "Cash Flow", "New Held", "New Wt",
    ]
    table_html = f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px;
                  background:#111827; border-radius:8px; overflow:hidden;">
      <thead>
        <tr style="background:#0d1525;">
          {''.join(
        f'<th style="padding:10px; text-align:left; color:#888; '
        f'font-size:11px; text-transform:uppercase;">{h}</th>'
        for h in headers
    )}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """

    result_pane.object = summary_html + table_html

    parts = []
    if n_sells > 0:
        parts.append(f"{n_sells} sell{'s' if n_sells != 1 else ''}")
    if n_buys > 0:
        parts.append(f"{n_buys} buy{'s' if n_buys != 1 else ''}")
    order_summary = " and ".join(parts) if parts else "no orders"

    status_pane.object = (
        f"🎯 Rebalance calculated: {order_summary}. "
        f"Saved to {saved_path.name}. Review below and click Apply when ready."
    )
    status_pane.alert_type = "success"


def on_apply(event):
    """
    Apply the calculated rebalance orders to the portfolio. This:
      1. Updates held quantities to post-rebalance values
      2. Removes liquidated stocks (weight=0) from the portfolio entirely
      3. Marks the state file as applied (so it loads correctly next startup)
    """
    global last_result_df, last_saved_path

    if last_result_df is None:
        status_pane.object = "⚠️ No rebalance to apply. Calculate first."
        status_pane.alert_type = "warning"
        return

    # Step 1: Update holdings from rebalance result
    for _, r in last_result_df.iterrows():
        for d in portfolio_data:
            if d["ticker"] == r["ticker"]:
                d["held"] = int(r["new_held"])

    # Step 2: Remove liquidated stocks (weight=0, now held=0 after selling)
    # These are fully exited positions — keeping them would just clutter the table.
    liquidated = [d["ticker"]
                  for d in portfolio_data if d["weight"] == 0 and d["held"] == 0]
    portfolio_data[:] = [d for d in portfolio_data if not (
        d["weight"] == 0 and d["held"] == 0)]

    # Step 3: Mark the state file as applied so it loads correctly on restart
    if last_saved_path and last_saved_path.exists():
        mark_state_applied(last_saved_path)

    refresh_portfolio_table()
    last_result_df = None
    last_saved_path = None

    liquidated_msg = ""
    if liquidated:
        liquidated_msg = f" Removed liquidated: {', '.join(liquidated)}."

    result_pane.object = (
        "<i>Orders applied. Holdings updated. Ready for next month!</i>"
    )
    status_pane.object = f"✅ Holdings updated successfully.{liquidated_msg}"
    status_pane.alert_type = "success"


# Wire up callbacks
add_btn.on_click(on_add_stock)
remove_btn.on_click(on_remove_stock)
fetch_btn.on_click(on_fetch_prices)
calc_btn.on_click(on_calculate)
apply_btn.on_click(on_apply)

# =============================================================================
# 6. LAYOUT
# =============================================================================
header = pn.pane.Markdown(
    "# 📊 Portfolio Rebalancer\n"
    "*SIP-style monthly rebalancing with live NSE prices via yFinance*",
    styles={"border-bottom": "1px solid #333", "padding-bottom": "8px"},
)

add_row = pn.Row(ticker_input, weight_input, held_input, add_btn, align="end")
remove_row = pn.Row(remove_input, remove_btn, align="end")
controls = pn.Row(monthly_input, fetch_btn, calc_btn, apply_btn, align="end")

explanation = pn.pane.Markdown("""
---
### How it works
1. **Define** your target portfolio: stocks + weights (active stocks must sum to 100%).
2. **Enter** shares you already hold (0 for a fresh start).
3. **Fetch** live prices or type them manually in the table.
4. **Calculate** — the algorithm allocates your monthly SIP toward the most underweight stocks.
5. **Liquidate** — set any stock's weight to **0%** to mark it for a full sell. The proceeds
   get redeployed into your remaining active positions automatically.
6. **Apply** — after executing trades in your broker, click Apply to update holdings.
   Liquidated positions are automatically removed from the portfolio.
7. **Repeat** next month!

*Prices are fetched from Yahoo Finance (NSE). The `.NS` suffix is auto-added. You can
also type `.BO` for BSE. Edit any cell in the table directly. Portfolio state is saved
to disk automatically and restored on restart.*
""")

layout = pn.Column(
    header,
    status_pane,
    pn.pane.Markdown("### ➕ Add / Remove Stocks"),
    add_row,
    remove_row,
    pn.pane.Markdown("### 📋 Your Portfolio"),
    controls,
    portfolio_table,
    pn.pane.Markdown("### 🎯 Rebalance Orders"),
    result_pane,
    explanation,
    max_width=1000,
    styles={"margin": "0 auto", "padding": "20px"},
)

layout.servable(title="Portfolio Rebalancer")

if __name__ == "__main__":
    pn.serve(layout, port=5006, show=True, title="Portfolio Rebalancer")
