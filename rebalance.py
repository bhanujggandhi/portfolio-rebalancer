"""
Portfolio Rebalancer — Panel + yFinance
========================================
Multi-portfolio SIP-style rebalancer with live NSE prices, liquidation
support, state history browsing, and portfolio management.

Setup:
    pip install panel yfinance pandas

Run:
    panel serve rebalancer.py --show
    # or just:  python rebalancer.py
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path

import panel as pn
import yfinance as yf
import pandas as pd

pn.extension("tabulator", sizing_mode="stretch_width")

# =============================================================================
# 1. TICKER DISPLAY NAMES
#
#    Names are fetched from yFinance once per ticker, then cached in-memory
#    and persisted in state files so restarts don't need API calls.
# =============================================================================

_name_cache: dict[str, str] = {}


def name_for_ticker(ticker: str) -> str:
    """Get display name for a ticker. Cached to avoid repeated API calls."""
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
    fallback = ticker.replace(".NS", "").replace(".BO", "")
    _name_cache[ticker] = fallback
    return fallback

# =============================================================================
# 2. PRICE FETCHING
#
#    Uses yf.download() for batch fetching (single HTTP call for all tickers),
#    which is 5-10x faster than sequential per-ticker calls.
# =============================================================================


def fetch_all_prices(tickers: list[str]) -> dict[str, float | None]:
    """Fetch prices for multiple tickers in one batch via yf.download()."""
    prices: dict[str, float | None] = {t: None for t in tickers}
    if not tickers:
        return prices
    try:
        df = yf.download(tickers, period="5d", progress=False, threads=True)
        if df.empty:
            return prices
        for t in tickers:
            try:
                close = df["Close"] if len(tickers) == 1 else df["Close"][t]
                valid = close.dropna()
                if not valid.empty:
                    prices[t] = round(float(valid.iloc[-1]), 2)
            except (KeyError, IndexError):
                continue
    except Exception as e:
        print(
            f"[WARN] Batch download failed: {e}. Falling back to per-ticker.")
        for t in tickers:
            prices[t] = _fetch_price_single(t)
    return prices


def _fetch_price_single(ticker_symbol: str) -> float | None:
    """Fallback: fetch price for one ticker via Ticker.info then history."""
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
#    Active stocks (weight > 0): buy-only, deficit-proportional allocation.
#    Liquidation stocks (weight == 0, held > 0): full sell, proceeds recycled.
#    Cash pool = new_monthly_sip + sum(liquidation_proceeds)
# =============================================================================


def calculate_rebalance(
    portfolio: pd.DataFrame,
    new_cash: float,
) -> pd.DataFrame:
    """Compute buy/sell orders to push portfolio toward target weights."""
    df = portfolio.copy()
    df["current_value"] = df["held"] * df["price"]

    # ── Liquidations: weight=0 with shares held → sell everything ──
    is_liq = (df["weight"] == 0) & (df["held"] > 0)
    df["sell_qty"] = 0
    df.loc[is_liq, "sell_qty"] = df.loc[is_liq, "held"]
    df["sell_value"] = df["sell_qty"] * df["price"]
    liq_proceeds = df["sell_value"].sum()

    # ── Available cash = fresh SIP + liquidation proceeds ──
    available_cash = new_cash + liq_proceeds

    # ── Rebalance active stocks only ──
    active = df["weight"] > 0
    active_val = df.loc[active, "current_value"].sum()
    total_target = active_val + available_cash

    df["target_value"] = 0.0
    df.loc[active, "target_value"] = total_target * \
        (df.loc[active, "weight"] / 100.0)

    df["deficit"] = 0.0
    df.loc[active, "deficit"] = (
        df.loc[active, "target_value"] - df.loc[active, "current_value"]
    ).clip(lower=0)
    total_deficit = df.loc[active, "deficit"].sum()

    df["allocation"] = 0.0
    if total_deficit > 0:
        df.loc[active, "allocation"] = (
            df.loc[active, "deficit"] / total_deficit
        ) * available_cash
    else:
        ws = df.loc[active, "weight"].sum()
        if ws > 0:
            df.loc[active, "allocation"] = available_cash * \
                (df.loc[active, "weight"] / ws)

    df["buy_qty"] = 0
    ok = active & (df["price"] > 0)
    df.loc[ok, "buy_qty"] = (
        df.loc[ok, "allocation"] / df.loc[ok, "price"]
    ).apply(math.floor).astype(int)
    df["buy_cost"] = df["buy_qty"] * df["price"]

    df["new_held"] = df["held"] + df["buy_qty"] - df["sell_qty"]
    df["new_value"] = df["new_held"] * df["price"]

    tc = df["current_value"].sum()
    nt = df["new_value"].sum()
    df["current_weight"] = (df["current_value"] / tc *
                            100).round(1) if tc > 0 else 0.0
    df["new_weight"] = (df["new_value"] / nt * 100).round(1) if nt > 0 else 0.0
    df["drift"] = (df["current_weight"] - df["weight"]).round(1)

    return df

# =============================================================================
# 4. STATE & PORTFOLIO PERSISTENCE
#
#    Directory structure:
#      state/
#        portfolios.json              ← registry of all portfolios
#        {portfolio_slug}/
#          rebalance_2025-01-15_...json
#          rebalance_2025-02-15_...json
#
#    portfolios.json format:
#      {
#        "portfolios": [
#          {"slug": "growth", "name": "Growth Portfolio", "created": "..."},
#          {"slug": "conservative", "name": "Conservative", "created": "..."}
#        ]
#      }
#
#    Each rebalance JSON has an "applied" flag. On startup we only restore
#    from applied states. The state dropdown lets you load any state (applied
#    or not) for review or rollback.
# =============================================================================


STATE_DIR = Path(__file__).resolve().parent / "state"
MAX_STATE_FILES = 50
DEFAULT_PORTFOLIO_NAME = "Default Portfolio"

# The default set of stocks used when creating a brand-new portfolio.
DEFAULT_STOCKS = [
    {"ticker": "IDFCFIRSTB.NS", "weight": 20.0, "held": 0, "price": 0.0},
    {"ticker": "FIVESTAR.NS",   "weight": 15.0, "held": 0, "price": 0.0},
    {"ticker": "HDFCBANK.NS",   "weight": 15.0, "held": 0, "price": 0.0},
    {"ticker": "INFY.NS",       "weight": 12.0, "held": 0, "price": 0.0},
    {"ticker": "TCS.NS",        "weight": 10.0, "held": 0, "price": 0.0},
    {"ticker": "BHARTIARTL.NS", "weight": 10.0, "held": 0, "price": 0.0},
    {"ticker": "SBIN.NS",       "weight":  8.0, "held": 0, "price": 0.0},
    {"ticker": "HINDUNILVR.NS", "weight": 10.0, "held": 0, "price": 0.0},
]


def _slugify(name: str) -> str:
    """
    Convert a human-readable portfolio name into a filesystem-safe slug.
    'My Growth Portfolio' → 'my-growth-portfolio'
    """
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)      # remove special chars
    s = re.sub(r"[\s_]+", "-", s)        # spaces/underscores → hyphens
    s = re.sub(r"-+", "-", s).strip("-")  # collapse multiple hyphens
    return s or "unnamed"


def _to_jsonable(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return None if v != v else round(v, 4)
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v)
    return v


def _portfolio_with_names(rows: list[dict]) -> list[dict]:
    """Ensure each row has a 'name' field derived from its ticker."""
    return [{**r, "name": _name_cache.get(r["ticker"], r.get("name", ""))} for r in rows]


# ── Portfolio registry ──

def _registry_path() -> Path:
    return STATE_DIR / "portfolios.json"


def load_portfolio_registry() -> list[dict]:
    """
    Load the list of all portfolios from portfolios.json. Each entry has
    'slug', 'name', and 'created' keys. If no registry exists, we create
    one with a single default portfolio.
    """
    path = _registry_path()
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            portfolios = data.get("portfolios", [])
            if portfolios:
                return portfolios
        except (json.JSONDecodeError, OSError):
            pass
    # No registry yet — create one with the default portfolio
    default = [{
        "slug": _slugify(DEFAULT_PORTFOLIO_NAME),
        "name": DEFAULT_PORTFOLIO_NAME,
        "created": datetime.now().isoformat(),
    }]
    _save_portfolio_registry(default)
    return default


def _save_portfolio_registry(portfolios: list[dict]):
    """Write the portfolio registry to disk."""
    STATE_DIR.mkdir(exist_ok=True)
    with open(_registry_path(), "w", encoding="utf-8") as f:
        json.dump({"portfolios": portfolios}, f, indent=2)


def add_portfolio_to_registry(name: str) -> dict | None:
    """
    Add a new portfolio to the registry. Returns the new entry dict,
    or None if a portfolio with the same slug already exists.
    """
    slug = _slugify(name)
    registry = load_portfolio_registry()
    if any(p["slug"] == slug for p in registry):
        return None
    entry = {"slug": slug, "name": name.strip(
    ), "created": datetime.now().isoformat()}
    registry.append(entry)
    _save_portfolio_registry(registry)
    return entry


def delete_portfolio_from_registry(slug: str) -> bool:
    """Remove a portfolio from the registry. Returns True if found and removed."""
    registry = load_portfolio_registry()
    before = len(registry)
    registry = [p for p in registry if p["slug"] != slug]
    if len(registry) < before:
        _save_portfolio_registry(registry)
        return True
    return False


# ── Per-portfolio state directory ──

def _portfolio_state_dir(slug: str) -> Path:
    """Return the state directory for a specific portfolio."""
    return STATE_DIR / slug


def list_state_files(slug: str) -> list[dict]:
    """
    List all state files for a portfolio, most recent first. Each entry has:
      - 'path': full Path object
      - 'filename': just the filename
      - 'timestamp': parsed datetime from filename
      - 'applied': whether the state was confirmed
      - 'summary': brief text for dropdown display
      - 'label': formatted string for the dropdown widget
    """
    d = _portfolio_state_dir(slug)
    if not d.exists():
        return []

    entries = []
    for fp in sorted(d.glob("rebalance_*.json"), reverse=True):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        applied = data.get("applied", False)
        ts_str = data.get("timestamp", "")
        summary = data.get("summary", {})
        cur_val = summary.get("current_value", 0)
        amount = data.get("amount", 0)

        # Parse timestamp for display
        try:
            ts = datetime.fromisoformat(ts_str)
            ts_display = ts.strftime("%d %b %Y, %H:%M")
        except (ValueError, TypeError):
            ts_display = fp.stem

        status = "✅ Applied" if applied else "⏳ Calculated"
        label = f"{ts_display} | {status} | Val: ₹{cur_val:,.0f} | SIP: ₹{amount:,.0f}"

        entries.append({
            "path": fp,
            "filename": fp.name,
            "timestamp": ts_str,
            "applied": applied,
            "label": label,
        })

    return entries


def save_rebalance_state(
    slug: str,
    amount: float,
    portfolio: list[dict],
    result: pd.DataFrame,
    applied: bool = False,
) -> Path:
    """Save rebalance data to a dated JSON file under the portfolio's directory."""
    d = _portfolio_state_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = d / f"rebalance_{ts}.json"

    result_records = [
        {k: _to_jsonable(v) for k, v in row.to_dict().items()}
        for _, row in result.iterrows()
    ]

    liq_total = float(result["sell_value"].sum())
    ticker_names = {
        r["ticker"]: _name_cache.get(r["ticker"], r.get("name", ""))
        for r in portfolio
    }

    data = {
        "timestamp": datetime.now().isoformat(),
        "applied": applied,
        "amount": amount,
        "portfolio_before": portfolio,
        "rebalance_result": result_records,
        "summary": {
            "current_value": float(result["current_value"].sum()),
            "total_invested": float(result["buy_cost"].sum()),
            "liquidation_proceeds": liq_total,
            "available_cash": amount + liq_total,
            "leftover": float((amount + liq_total) - result["buy_cost"].sum()),
        },
        "ticker_names": ticker_names,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    _cleanup_old_state_files(slug)
    return filepath


def mark_state_applied(filepath: Path):
    """Flip the 'applied' flag to True in an existing state file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        data["applied"] = True
        data["applied_at"] = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not mark state as applied: {e}")


def load_state_from_file(filepath: Path) -> tuple[list[dict], float] | None:
    """
    Load portfolio state from a specific state file. Works for both applied
    and unapplied states. For applied states, 'new_held' is the post-trade
    holding. For unapplied states, we load 'portfolio_before' (pre-trade)
    so we don't show phantom trades the user never executed.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    applied = data.get("applied", False)

    # Restore ticker name cache
    for ticker, name in data.get("ticker_names", {}).items():
        if name and ticker not in _name_cache:
            _name_cache[ticker] = name

    amount = float(data.get("amount", 10_000))

    if applied:
        # Applied state: use rebalance_result with new_held as current holdings.
        # This represents the actual state of the portfolio after executing trades.
        result = data.get("rebalance_result", [])
        if not result:
            return None
        portfolio = [
            {
                "ticker": r["ticker"],
                "name": _name_cache.get(r["ticker"], r["ticker"].replace(".NS", "")),
                "weight": r["weight"],
                "held": int(r.get("new_held", r.get("held", 0))),
                "price": float(r.get("price", 0)),
            }
            for r in result
        ]
    else:
        # Unapplied state: use portfolio_before (the state BEFORE the hypothetical
        # rebalance). This avoids loading phantom trades the user never executed.
        before = data.get("portfolio_before", [])
        if not before:
            return None
        portfolio = [
            {
                "ticker": r["ticker"],
                "name": _name_cache.get(r["ticker"], r["ticker"].replace(".NS", "")),
                "weight": r["weight"],
                "held": int(r.get("held", 0)),
                "price": float(r.get("price", 0)),
            }
            for r in before
        ]

    return (portfolio, amount)


def load_latest_applied_state(slug: str) -> tuple[list[dict], float] | None:
    """
    Load the most recent APPLIED state for a portfolio. This is used on
    startup and when switching portfolios — we only auto-load confirmed states.
    """
    for entry in list_state_files(slug):
        if entry["applied"]:
            return load_state_from_file(entry["path"])
    return None


def _cleanup_old_state_files(slug: str):
    """Keep only the most recent MAX_STATE_FILES for a portfolio."""
    d = _portfolio_state_dir(slug)
    if not d.exists():
        return
    files = sorted(d.glob("rebalance_*.json"))
    if len(files) > MAX_STATE_FILES:
        for old in files[: len(files) - MAX_STATE_FILES]:
            try:
                old.unlink()
            except OSError:
                pass

# =============================================================================
# 5. PANEL UI
#
#    Layout (top to bottom):
#      Header
#      Portfolio selector row  (dropdown + new portfolio input + create/delete)
#      Status bar
#      State history row       (dropdown + load button)
#      Add / Remove stock row
#      Portfolio table + action buttons
#      Rebalance results
#      Explanation
# =============================================================================


# ── Initialize: discover portfolios, pick the first one, load its state ──
_registry = load_portfolio_registry()
_active_slug = _registry[0]["slug"]

# Try loading the latest applied state for the active portfolio
_init_state = load_latest_applied_state(_active_slug)
portfolio_data: list[dict] = []
_init_amount = 10_000.0

if _init_state:
    portfolio_data = _init_state[0]
    _init_amount = _init_state[1]
else:
    portfolio_data = _portfolio_with_names(DEFAULT_STOCKS)


# ── Widgets: Portfolio management ──

portfolio_selector = pn.widgets.Select(
    name="Active Portfolio",
    options={p["name"]: p["slug"] for p in _registry},
    value=_active_slug,
    width=280,
)
new_portfolio_input = pn.widgets.TextInput(
    name="New Portfolio Name",
    placeholder="e.g. Growth Portfolio",
    width=250,
)
create_portfolio_btn = pn.widgets.Button(
    name="➕ Create Portfolio", button_type="success", width=180,
)
delete_portfolio_btn = pn.widgets.Button(
    name="🗑️ Delete Portfolio", button_type="danger", width=180,
)


# ── Widgets: State history ──

def _build_state_options(slug: str) -> dict[str, str]:
    """
    Build the options dict for the state history dropdown. Keys are
    human-readable labels, values are filenames. The first entry is always
    a placeholder prompting the user to select a state.
    """
    entries = list_state_files(slug)
    if not entries:
        return {"(No saved states)": ""}
    opts = {"— Select a state to load —": ""}
    for e in entries:
        opts[e["label"]] = e["filename"]
    return opts


state_selector = pn.widgets.Select(
    name="State History",
    options=_build_state_options(_active_slug),
    width=550,
)
load_state_btn = pn.widgets.Button(
    name="📂 Load Selected State", button_type="primary", width=200,
)


# ── Widgets: Stock management ──

monthly_input = pn.widgets.FloatInput(
    name="Monthly Investment (₹)", value=_init_amount, step=1000, start=0, width=220,
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
    name="➕ Add Stock", button_type="primary", width=140)
remove_input = pn.widgets.TextInput(
    name="Remove Ticker", placeholder="e.g. TATAMOTORS.NS", width=250,
)
remove_btn = pn.widgets.Button(
    name="🗑️ Remove", button_type="danger", width=120)
fetch_btn = pn.widgets.Button(
    name="📡 Fetch Prices", button_type="success", width=160)
calc_btn = pn.widgets.Button(
    name="🎯 Calculate Rebalance", button_type="warning", width=200)
apply_btn = pn.widgets.Button(
    name="✅ Apply Orders", button_type="primary", width=160,
)


# ── Display panes ──

status_pane = pn.pane.Alert(
    f"Loaded portfolio: {_registry[0]['name']}. Add stocks and fetch prices to begin.",
    alert_type="info",
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


# ── Global tracking for Apply ──
last_result_df: pd.DataFrame | None = None
last_saved_path: Path | None = None


# ── Helper functions ──

def refresh_portfolio_table():
    portfolio_table.value = pd.DataFrame(portfolio_data)


def refresh_state_dropdown():
    """Refresh the state history dropdown for the currently active portfolio."""
    state_selector.options = _build_state_options(portfolio_selector.value)


def sync_edits_from_table():
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


def _load_portfolio_into_ui(slug: str, state: tuple[list[dict], float] | None):
    """
    Replace all in-memory portfolio data and refresh all UI elements to
    reflect a portfolio switch or state load. This is the single point of
    truth for "what does the UI show right now."
    """
    global portfolio_data, last_result_df, last_saved_path

    if state:
        portfolio_data = state[0]
        monthly_input.value = state[1]
    else:
        # No state found — start fresh with default stocks
        portfolio_data = _portfolio_with_names(DEFAULT_STOCKS)
        monthly_input.value = 10_000

    # Clear any pending rebalance (it belonged to the old portfolio/state)
    last_result_df = None
    last_saved_path = None
    result_pane.object = "<i>No rebalance computed yet.</i>"

    refresh_portfolio_table()
    refresh_state_dropdown()


# ── Callbacks: Portfolio management ──

def on_portfolio_switch(event):
    """Called when the user selects a different portfolio from the dropdown."""
    slug = event.new
    state = load_latest_applied_state(slug)
    _load_portfolio_into_ui(slug, state)

    # Find the display name for the status message
    registry = load_portfolio_registry()
    name = next((p["name"] for p in registry if p["slug"] == slug), slug)
    status_pane.object = f"📂 Switched to portfolio: {name}."
    status_pane.alert_type = "info"


def on_create_portfolio(event):
    """Create a new portfolio and switch to it."""
    name = new_portfolio_input.value.strip()
    if not name:
        status_pane.object = "⚠️ Please enter a name for the new portfolio."
        status_pane.alert_type = "warning"
        return

    entry = add_portfolio_to_registry(name)
    if entry is None:
        status_pane.object = f"⚠️ A portfolio named '{name}' (or similar) already exists."
        status_pane.alert_type = "warning"
        return

    # Update the dropdown options and switch to the new portfolio
    registry = load_portfolio_registry()
    portfolio_selector.options = {p["name"]: p["slug"] for p in registry}
    portfolio_selector.value = entry["slug"]

    # Start with default stocks for the new portfolio
    _load_portfolio_into_ui(entry["slug"], None)

    new_portfolio_input.value = ""
    status_pane.object = f"✅ Created new portfolio: {name}. Add your stocks and set weights."
    status_pane.alert_type = "success"


def on_delete_portfolio(event):
    """
    Delete the currently active portfolio. We prevent deleting the last
    remaining portfolio to avoid a broken state.
    """
    slug = portfolio_selector.value
    registry = load_portfolio_registry()
    if len(registry) <= 1:
        status_pane.object = "⚠️ Cannot delete the last remaining portfolio."
        status_pane.alert_type = "danger"
        return

    name = next((p["name"] for p in registry if p["slug"] == slug), slug)

    # Remove from registry
    delete_portfolio_from_registry(slug)

    # Optionally clean up state files on disk
    d = _portfolio_state_dir(slug)
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    # Switch to the first remaining portfolio
    registry = load_portfolio_registry()
    portfolio_selector.options = {p["name"]: p["slug"] for p in registry}
    portfolio_selector.value = registry[0]["slug"]

    state = load_latest_applied_state(registry[0]["slug"])
    _load_portfolio_into_ui(registry[0]["slug"], state)

    status_pane.object = f"🗑️ Deleted portfolio: {name}. Switched to {registry[0]['name']}."
    status_pane.alert_type = "info"


# ── Callbacks: State history ──

def on_load_state(event):
    """
    Load a specific historical state into the UI. This works for both applied
    and unapplied states — the key difference is which holdings get loaded
    (see load_state_from_file docstring for details).
    """
    filename = state_selector.value
    if not filename:
        status_pane.object = "⚠️ Please select a state from the dropdown first."
        status_pane.alert_type = "warning"
        return

    slug = portfolio_selector.value
    filepath = _portfolio_state_dir(slug) / filename

    if not filepath.exists():
        status_pane.object = f"⚠️ State file not found: {filename}"
        status_pane.alert_type = "danger"
        return

    state = load_state_from_file(filepath)
    if state is None:
        status_pane.object = f"⚠️ Could not parse state file: {filename}"
        status_pane.alert_type = "danger"
        return

    _load_portfolio_into_ui(slug, state)

    # Check if this was an applied or unapplied state for the message
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        applied = data.get("applied", False)
    except Exception:
        applied = False

    if applied:
        status_pane.object = (
            f"📂 Loaded applied state from {filename}. "
            "This reflects your portfolio after those trades were executed."
        )
    else:
        status_pane.object = (
            f"📂 Loaded unapplied state from {filename}. "
            "This shows the portfolio BEFORE that rebalance (trades were never confirmed)."
        )
    status_pane.alert_type = "info"


# ── Callbacks: Stock management ──

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
    status_pane.object = f"✅ Added {ticker}. Total active weight: {total_w:.1f}%"
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
            "Edit manually in the table."
        )
        status_pane.alert_type = "warning"
    else:
        status_pane.object = (
            f"✅ All {len(tickers)} prices fetched at {datetime.now():%H:%M:%S}."
        )
        status_pane.alert_type = "success"


def on_calculate(event):
    global last_result_df, last_saved_path
    sync_edits_from_table()

    active_w = sum(d["weight"] for d in portfolio_data if d["weight"] > 0)
    if abs(active_w - 100) > 0.1:
        status_pane.object = (
            f"⚠️ Active weights sum to {active_w:.1f}%, must be 100%."
        )
        status_pane.alert_type = "danger"
        return

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

    slug = portfolio_selector.value
    df = pd.DataFrame(portfolio_data)
    result = calculate_rebalance(df, monthly_input.value)
    last_result_df = result

    saved = save_rebalance_state(
        slug=slug,
        amount=monthly_input.value,
        portfolio=[{**d} for d in portfolio_data],
        result=result,
        applied=False,
    )
    last_saved_path = saved

    # Refresh the state dropdown so the new entry appears immediately
    refresh_state_dropdown()

    # ── Build HTML results ──
    total_invested = result["buy_cost"].sum()
    current_val = result["current_value"].sum()
    liq_val = result["sell_value"].sum()
    available = monthly_input.value + liq_val
    leftover = available - total_invested
    n_sells = int((result["sell_qty"] > 0).sum())
    n_buys = int((result["buy_qty"] > 0).sum())

    liq_card = ""
    if liq_val > 0:
        liq_card = f"""
        <div style="background:#2a1a1a; padding:12px 18px; border-radius:8px;
                    min-width:150px; border:1px solid #7f1d1d;">
          <div style="font-size:11px; color:#f87171; text-transform:uppercase;">
            Liquidation ({n_sells} stock{'s' if n_sells != 1 else ''})</div>
          <div style="font-size:20px; font-weight:700; color:#f87171;">₹{liq_val:,.0f}</div>
        </div>"""

    summary_html = f"""
    <div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:16px;">
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:140px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Portfolio Value</div>
        <div style="font-size:20px; font-weight:700; color:#60a5fa;">₹{current_val:,.0f}</div>
      </div>
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:140px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Fresh SIP</div>
        <div style="font-size:20px; font-weight:700; color:#34d399;">₹{monthly_input.value:,.0f}</div>
      </div>
      {liq_card}
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:140px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Total Cash</div>
        <div style="font-size:20px; font-weight:700; color:#a78bfa;">₹{available:,.0f}</div>
      </div>
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:140px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Deployed</div>
        <div style="font-size:20px; font-weight:700; color:#fbbf24;">₹{total_invested:,.0f}</div>
      </div>
      <div style="background:#1a2332; padding:12px 18px; border-radius:8px; min-width:140px;">
        <div style="font-size:11px; color:#888; text-transform:uppercase;">Uninvested</div>
        <div style="font-size:20px; font-weight:700; color:#f87171;">₹{leftover:,.0f}</div>
      </div>
    </div>"""

    rows_html = ""
    for _, r in result.iterrows():
        is_sell = int(r["sell_qty"]) > 0
        is_buy = int(r["buy_qty"]) > 0
        bg = "background:#2a1015;" if is_sell else (
            "background:#0d1a0d;" if is_buy else "")

        if is_sell:
            act = f'<span style="color:#f87171;font-weight:700;">🔴 SELL {int(r["sell_qty"])}</span>'
            cf = f'<span style="color:#f87171;">+₹{r["sell_value"]:,.0f}</span>'
        elif is_buy:
            act = f'<span style="color:#34d399;font-weight:700;">🟢 BUY {int(r["buy_qty"])}</span>'
            cf = f'<span style="color:#60a5fa;">−₹{r["buy_cost"]:,.0f}</span>'
        else:
            act = '<span style="color:#666;">—</span>'
            cf = '<span style="color:#666;">—</span>'

        dv = r["drift"]
        dc = "#f87171" if dv < -1 else "#34d399" if dv > 1 else "#888"
        nc = "#34d399" if abs(r["new_weight"] - r["weight"]) < 2 else "#fbbf24"

        rows_html += f"""
        <tr style="{bg}">
          <td style="padding:7px 10px;font-weight:600;">{r['ticker']}</td>
          <td style="padding:7px 10px;">{r['name']}</td>
          <td style="padding:7px 10px;">₹{r['price']:,.2f}</td>
          <td style="padding:7px 10px;color:#fbbf24;">{r['weight']:.1f}%</td>
          <td style="padding:7px 10px;">{r['current_weight']:.1f}%</td>
          <td style="padding:7px 10px;color:{dc};">{'+' if dv > 0 else ''}{dv:.1f}%</td>
          <td style="padding:7px 10px;font-size:14px;">{act}</td>
          <td style="padding:7px 10px;">{cf}</td>
          <td style="padding:7px 10px;">{int(r['new_held'])}</td>
          <td style="padding:7px 10px;color:{nc};">{r['new_weight']:.1f}%</td>
        </tr>"""

    hdrs = ["Ticker", "Name", "Price", "Target", "Current",
            "Drift", "Action", "Cash Flow", "New Held", "New Wt"]
    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;background:#111827;border-radius:8px;overflow:hidden;">
      <thead><tr style="background:#0d1525;">
        {''.join(f'<th style="padding:9px;text-align:left;color:#888;font-size:11px;text-transform:uppercase;">{h}</th>' for h in hdrs)}
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""

    result_pane.object = summary_html + table_html

    parts = []
    if n_sells:
        parts.append(f"{n_sells} sell{'s' if n_sells != 1 else ''}")
    if n_buys:
        parts.append(f"{n_buys} buy{'s' if n_buys != 1 else ''}")
    status_pane.object = (
        f"🎯 Rebalance: {' + '.join(parts) or 'no orders'}. "
        f"Saved to {saved.name}. Click Apply after executing trades."
    )
    status_pane.alert_type = "success"


def on_apply(event):
    global last_result_df, last_saved_path
    if last_result_df is None:
        status_pane.object = "⚠️ No rebalance to apply. Calculate first."
        status_pane.alert_type = "warning"
        return

    for _, r in last_result_df.iterrows():
        for d in portfolio_data:
            if d["ticker"] == r["ticker"]:
                d["held"] = int(r["new_held"])

    liquidated = [d["ticker"]
                  for d in portfolio_data if d["weight"] == 0 and d["held"] == 0]
    portfolio_data[:] = [d for d in portfolio_data if not (
        d["weight"] == 0 and d["held"] == 0)]

    if last_saved_path and last_saved_path.exists():
        mark_state_applied(last_saved_path)

    refresh_portfolio_table()
    refresh_state_dropdown()
    last_result_df = None
    last_saved_path = None

    liq_msg = f" Removed: {', '.join(liquidated)}." if liquidated else ""
    result_pane.object = "<i>Orders applied. Holdings updated. Ready for next month!</i>"
    status_pane.object = f"✅ Holdings updated.{liq_msg}"
    status_pane.alert_type = "success"


# ── Wire up all callbacks ──
portfolio_selector.param.watch(on_portfolio_switch, "value")
create_portfolio_btn.on_click(on_create_portfolio)
delete_portfolio_btn.on_click(on_delete_portfolio)
load_state_btn.on_click(on_load_state)
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
    "*Multi-portfolio SIP rebalancing with live NSE prices via yFinance*",
    styles={"border-bottom": "1px solid #333", "padding-bottom": "8px"},
)

portfolio_row = pn.Row(
    portfolio_selector, new_portfolio_input, create_portfolio_btn,
    delete_portfolio_btn, align="end",
)
state_row = pn.Row(state_selector, load_state_btn, align="end")
add_row = pn.Row(ticker_input, weight_input, held_input, add_btn, align="end")
remove_row = pn.Row(remove_input, remove_btn, align="end")
controls = pn.Row(monthly_input, fetch_btn, calc_btn, apply_btn, align="end")

explanation = pn.pane.Markdown("""
---
### How it works
1. **Create a portfolio** or pick an existing one from the dropdown.
2. **Define** stocks + weights (active weights must sum to 100%).
3. **Fetch** live prices (or type them manually).
4. **Calculate** — your monthly SIP is directed toward the most underweight stocks.
5. **Liquidate** — set weight to **0%** to sell everything. Proceeds are redeployed.
6. **Apply** — after executing trades in your broker, click Apply to confirm.
7. **Load history** — use the state dropdown to view or roll back to any past state.
8. **Repeat** next month!

*Ticker format: `.NS` for NSE, `.BO` for BSE (auto-appended if missing).
Each portfolio's state is saved independently under `state/{portfolio}/`.*
""")

layout = pn.Column(
    header,
    pn.pane.Markdown("### 🗂️ Portfolio"),
    portfolio_row,
    status_pane,
    pn.pane.Markdown("### 📜 State History"),
    state_row,
    pn.pane.Markdown("### ➕ Add / Remove Stocks"),
    add_row,
    remove_row,
    pn.pane.Markdown("### 📋 Holdings & Actions"),
    controls,
    portfolio_table,
    pn.pane.Markdown("### 🎯 Rebalance Orders"),
    result_pane,
    explanation,
    max_width=1050,
    styles={"margin": "0 auto", "padding": "20px"},
)

layout.servable(title="Portfolio Rebalancer")

if __name__ == "__main__":
    pn.serve(layout, port=5006, show=True, title="Portfolio Rebalancer")
