"""
Portfolio Rebalancer — Panel + yFinance
========================================
Multi-portfolio SIP-style rebalancer with live NSE prices, liquidation
support, trade restrictions, state history, and portfolio management.

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
# 3. REBALANCING LOGIC
#
#    Every stock falls into exactly one of four categories:
#
#      RESTRICTED   — restricted=True (any weight) → completely frozen
#      LIQUIDATION  — restricted=False, weight=0, held>0 → sell all
#      ACTIVE       — restricted=False, weight>0 → buy to close deficit
#      INERT        — restricted=False, weight=0, held=0 → no action
#
#    Cash pool = new_monthly_sip + sum(unrestricted liquidation proceeds)
#
#    Restricted stocks are frozen but their current value IS included in
#    total portfolio valuation so that active stocks' targets remain accurate.
#    This way, when the restriction lifts next month, the deficit algorithm
#    naturally corrects any drift that accumulated while it was restricted.
# =============================================================================


def calculate_rebalance(
    portfolio: pd.DataFrame,   # columns: ticker, name, weight, held, price, restricted
    new_cash: float,
) -> pd.DataFrame:
    """
    Compute buy/sell orders to push portfolio toward target weights,
    respecting trade restrictions on individual tickers.
    """
    df = portfolio.copy()
    df["current_value"] = df["held"] * df["price"]

    # ── Ensure restricted column exists (backward compat with old data) ──
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

    # ── LIQUIDATION: sell everything for unrestricted zero-weight stocks ──
    is_liq = df["category"] == "LIQUIDATION"
    df.loc[is_liq, "sell_qty"] = df.loc[is_liq, "held"]
    df.loc[is_liq, "sell_value"] = df.loc[is_liq,
                                          "sell_qty"] * df.loc[is_liq, "price"]
    liq_proceeds = df["sell_value"].sum()

    # ── Available cash = fresh SIP + liquidation proceeds ──
    available_cash = new_cash + liq_proceeds

    # ── ACTIVE stocks: deficit-proportional allocation ──
    # Total target includes restricted stocks' frozen value so that active
    # stocks' targets are computed in the context of the full portfolio.
    is_active = df["category"] == "ACTIVE"
    is_restricted = df["category"] == "RESTRICTED"

    restricted_value = df.loc[is_restricted, "current_value"].sum()
    active_value = df.loc[is_active, "current_value"].sum()
    total_target = restricted_value + active_value + available_cash

    df["target_value"] = 0.0
    df.loc[is_active, "target_value"] = total_target * \
        (df.loc[is_active, "weight"] / 100.0)

    df["deficit"] = 0.0
    df.loc[is_active, "deficit"] = (
        df.loc[is_active, "target_value"] - df.loc[is_active, "current_value"]
    ).clip(lower=0)
    total_deficit = df.loc[is_active, "deficit"].sum()

    df["allocation"] = 0.0
    if total_deficit > 0:
        df.loc[is_active, "allocation"] = (
            df.loc[is_active, "deficit"] / total_deficit
        ) * available_cash
    else:
        ws = df.loc[is_active, "weight"].sum()
        if ws > 0:
            df.loc[is_active, "allocation"] = available_cash * \
                (df.loc[is_active, "weight"] / ws)

    # ── Convert allocations to whole-share buy orders ──
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
#        portfolios.json                ← registry of all portfolios
#        {portfolio_slug}/
#          rebalance_YYYY-MM-DD_HHMMSS.json
#
#    Backward compatibility:
#      - Old state files without "restricted_tickers" → treated as []
#      - Old portfolio rows without "restricted" → treated as False
#      - Old state files without "applied" → treated as False
#      - Missing "ticker_names" → falls back to _name_cache or raw symbol
#
#    Restrictions are saved for audit but NOT restored on load (they reset
#    each session because they're inherently a per-month decision).
# =============================================================================


STATE_DIR = Path(__file__).resolve().parent / "state"
MAX_STATE_FILES = 50
DEFAULT_PORTFOLIO_NAME = "Default Portfolio"

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
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unnamed"


def _to_jsonable(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return None if v != v else round(v, 4)
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, bool):
        return bool(v)
    return v


def _ensure_portfolio_fields(rows: list[dict]) -> list[dict]:
    """
    Ensure every portfolio row has all required fields, with safe defaults
    for anything missing. This is the backward-compatibility layer — old
    state files may not have 'restricted' or even 'name' on every row.
    """
    out = []
    for r in rows:
        row = {**r}
        row.setdefault("restricted", False)
        row.setdefault("price", 0.0)
        row.setdefault("held", 0)
        row.setdefault("weight", 0.0)
        # Resolve name: prefer cache, then existing value, then raw symbol
        row["name"] = (
            _name_cache.get(row["ticker"])
            or row.get("name")
            or row["ticker"].replace(".NS", "").replace(".BO", "")
        )
        out.append(row)
    return out


# ── Portfolio registry ──

def _registry_path() -> Path:
    return STATE_DIR / "portfolios.json"


def load_portfolio_registry() -> list[dict]:
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
    default = [{
        "slug": _slugify(DEFAULT_PORTFOLIO_NAME),
        "name": DEFAULT_PORTFOLIO_NAME,
        "created": datetime.now().isoformat(),
    }]
    _save_portfolio_registry(default)
    return default


def _save_portfolio_registry(portfolios: list[dict]):
    STATE_DIR.mkdir(exist_ok=True)
    with open(_registry_path(), "w", encoding="utf-8") as f:
        json.dump({"portfolios": portfolios}, f, indent=2)


def add_portfolio_to_registry(name: str) -> dict | None:
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
    registry = load_portfolio_registry()
    before = len(registry)
    registry = [p for p in registry if p["slug"] != slug]
    if len(registry) < before:
        _save_portfolio_registry(registry)
        return True
    return False


# ── Per-portfolio state ──

def _portfolio_state_dir(slug: str) -> Path:
    return STATE_DIR / slug


def list_state_files(slug: str) -> list[dict]:
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
        # .get() with defaults everywhere for backward compat with old schemas
        applied = data.get("applied", False)
        ts_str = data.get("timestamp", "")
        summary = data.get("summary", {})
        cur_val = summary.get("current_value", 0)
        amount = data.get("amount", 0)
        n_restricted = len(data.get("restricted_tickers", []))
        try:
            ts = datetime.fromisoformat(ts_str)
            ts_display = ts.strftime("%d %b %Y, %H:%M")
        except (ValueError, TypeError):
            ts_display = fp.stem
        status = "✅ Applied" if applied else "⏳ Calculated"
        restriction_note = f" | 🔒{n_restricted}" if n_restricted else ""
        label = (
            f"{ts_display} | {status} | Val: ₹{cur_val:,.0f}"
            f" | SIP: ₹{amount:,.0f}{restriction_note}"
        )
        entries.append({
            "path": fp, "filename": fp.name,
            "timestamp": ts_str, "applied": applied, "label": label,
        })
    return entries


def save_rebalance_state(
    slug: str, amount: float, portfolio: list[dict],
    result: pd.DataFrame, applied: bool = False,
) -> Path:
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
    restricted_tickers = [r["ticker"]
                          for r in portfolio if r.get("restricted", False)]

    data = {
        "timestamp": datetime.now().isoformat(),
        "applied": applied,
        "amount": amount,
        "restricted_tickers": restricted_tickers,
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
    Load portfolio state from a specific state file.
    Applied → use new_held (post-trade). Unapplied → use portfolio_before.
    Restrictions are always reset to False on load (per-session decision).
    All field reads use .get() for backward compat with old schemas.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    applied = data.get("applied", False)

    # Restore ticker name cache from state file (if present)
    for ticker, name in data.get("ticker_names", {}).items():
        if name and ticker not in _name_cache:
            _name_cache[ticker] = name

    amount = float(data.get("amount", 10_000))

    if applied:
        result = data.get("rebalance_result", [])
        if not result:
            return None
        portfolio = [
            {
                "ticker": r["ticker"],
                "weight": r.get("weight", 0),
                "held": int(r.get("new_held", r.get("held", 0))),
                "price": float(r.get("price", 0)),
                "restricted": False,  # always reset on load
            }
            for r in result
        ]
    else:
        before = data.get("portfolio_before", [])
        if not before:
            return None
        portfolio = [
            {
                "ticker": r["ticker"],
                "weight": r.get("weight", 0),
                "held": int(r.get("held", 0)),
                "price": float(r.get("price", 0)),
                "restricted": False,
            }
            for r in before
        ]

    # Run through the compat layer to ensure all fields exist and names resolve
    return (_ensure_portfolio_fields(portfolio), amount)


def load_latest_applied_state(slug: str) -> tuple[list[dict], float] | None:
    for entry in list_state_files(slug):
        if entry["applied"]:
            return load_state_from_file(entry["path"])
    return None


def _cleanup_old_state_files(slug: str):
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
# =============================================================================


# ── Initialize ──
_registry = load_portfolio_registry()
_active_slug = _registry[0]["slug"]
_init_state = load_latest_applied_state(_active_slug)
portfolio_data: list[dict] = []
_init_amount = 10_000.0

if _init_state:
    portfolio_data = _init_state[0]
    _init_amount = _init_state[1]
else:
    portfolio_data = _ensure_portfolio_fields(DEFAULT_STOCKS)


# ── Widgets: Portfolio management ──

portfolio_selector = pn.widgets.Select(
    name="Active Portfolio",
    options={p["name"]: p["slug"] for p in _registry},
    value=_active_slug, width=280,
)
new_portfolio_input = pn.widgets.TextInput(
    name="New Portfolio Name", placeholder="e.g. Growth Portfolio", width=250,
)
create_portfolio_btn = pn.widgets.Button(
    name="➕ Create Portfolio", button_type="success", width=180,
)
delete_portfolio_btn = pn.widgets.Button(
    name="🗑️ Delete Portfolio", button_type="danger", width=180,
)

# ── Widgets: State history ──


def _build_state_options(slug: str) -> dict[str, str]:
    entries = list_state_files(slug)
    if not entries:
        return {"(No saved states)": ""}
    opts = {"— Select a state to load —": ""}
    for e in entries:
        opts[e["label"]] = e["filename"]
    return opts


state_selector = pn.widgets.Select(
    name="State History",
    options=_build_state_options(_active_slug), width=600,
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

# ── Widgets: Trade restriction (separate from the table) ──
# The restriction input works as a toggle: type a ticker and click the button.
# If the ticker is currently unrestricted, it becomes restricted.
# If already restricted, the restriction is lifted.
restrict_input = pn.widgets.TextInput(
    name="Restrict / Unrestrict Ticker",
    placeholder="e.g. RELIANCE.NS (toggles restriction)",
    width=300,
)
restrict_btn = pn.widgets.Button(
    name="🔒 Toggle Restriction", button_type="warning", width=200,
)

fetch_btn = pn.widgets.Button(
    name="📡 Fetch Prices", button_type="success", width=160)
calc_btn = pn.widgets.Button(
    name="🎯 Calculate Rebalance", button_type="warning", width=200)
apply_btn = pn.widgets.Button(
    name="✅ Apply Orders", button_type="primary", width=160)


# ── Display panes ──

status_pane = pn.pane.Alert(
    f"Loaded portfolio: {_registry[0]['name']}.", alert_type="info",
)

# Persistent weight warning banner — visible only when weights are invalid.
# Unlike the status_pane (which changes on every action), this stays visible
# as a constant reminder until the user fixes their weights.
weight_warning_pane = pn.pane.Alert(
    "", alert_type="danger", visible=False,
)


def _make_table_df(data: list[dict]) -> pd.DataFrame:
    """
    Build the DataFrame for the Tabulator. The 'restricted' column is
    displayed as a read-only text indicator rather than an editable checkbox.
    The user toggles restrictions via the dedicated input field instead.
    """
    rows = []
    for d in data:
        rows.append({
            "ticker": d["ticker"],
            "name": d["name"],
            "weight": d["weight"],
            "held": d["held"],
            "price": d["price"],
            # Display as a clear visual indicator, not a raw True/False
            "restricted": "🔒 Yes" if d.get("restricted", False) else "",
        })
    return pd.DataFrame(rows)


portfolio_table = pn.widgets.Tabulator(
    _make_table_df(portfolio_data),
    layout="fit_data_stretch",
    theme="midnight",
    height=340,
    selectable=False,
    show_index=False,
    titles={
        "ticker": "Ticker", "name": "Name", "weight": "Target Wt %",
        "held": "Shares Held", "price": "Price ₹", "restricted": "🔒 Status",
    },
    editors={
        "weight": {"type": "number", "step": 0.5, "min": 0, "max": 100},
        "held":   {"type": "number", "step": 1, "min": 0},
        "price":  {"type": "number", "step": 0.05, "min": 0},
        # restricted column is NOT editable — it's controlled by the toggle button
        "restricted": None,
    },
    frozen_columns=["ticker"],
    widths={"restricted": 100, "weight": 110, "held": 110, "price": 100},
)

result_pane = pn.pane.HTML(
    "<i>No rebalance computed yet.</i>", sizing_mode="stretch_width",
)


# ── Global tracking ──
last_result_df: pd.DataFrame | None = None
last_saved_path: Path | None = None


# ── Helpers ──

def refresh_portfolio_table():
    """Rebuild the table from portfolio_data. The restricted column is derived
    from the in-memory data, so toggling a restriction just needs a refresh."""
    portfolio_table.value = _make_table_df(portfolio_data)


def refresh_state_dropdown():
    state_selector.options = _build_state_options(portfolio_selector.value)


def _update_weight_warning():
    """
    Check whether the total of all non-zero weights is valid and update
    the persistent warning banner accordingly. This runs after every action
    that could change weights (add/remove stock, table edit, state load).

    The warning banner stays visible as a constant reminder — unlike the
    status bar which gets overwritten by the next action. This ensures
    the user can't miss a weight problem even after doing other things.
    """
    total_w = sum(d["weight"] for d in portfolio_data if d["weight"] > 0)

    if total_w > 100.1:
        weight_warning_pane.object = (
            f"⚠️ <b>Weights exceed 100%!</b> Non-zero weights sum to "
            f"<b>{total_w:.1f}%</b>. This must be fixed before rebalancing — "
            f"the algorithm cannot compute valid targets when weights exceed 100%. "
            f"Please reduce some weights so they sum to exactly 100%."
        )
        weight_warning_pane.alert_type = "danger"
        weight_warning_pane.visible = True
    elif abs(total_w - 100) > 0.1 and len(portfolio_data) > 0:
        weight_warning_pane.object = (
            f"ℹ️ Non-zero weights sum to <b>{total_w:.1f}%</b> "
            f"(need exactly 100% to rebalance)."
        )
        weight_warning_pane.alert_type = "warning"
        weight_warning_pane.visible = True
    else:
        weight_warning_pane.visible = False


def sync_edits_from_table():
    """
    Read back editable fields from the Tabulator into portfolio_data.
    The restricted column is NOT synced here because it's read-only in the
    table — restrictions are managed exclusively via the toggle button.
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
    # After syncing edits, weights may have changed — update the warning
    _update_weight_warning()


def _load_portfolio_into_ui(slug: str, state: tuple[list[dict], float] | None):
    global portfolio_data, last_result_df, last_saved_path
    if state:
        portfolio_data = state[0]
        monthly_input.value = state[1]
    else:
        portfolio_data = _ensure_portfolio_fields(DEFAULT_STOCKS)
        monthly_input.value = 10_000
    last_result_df = None
    last_saved_path = None
    result_pane.object = "<i>No rebalance computed yet.</i>"
    refresh_portfolio_table()
    refresh_state_dropdown()
    _update_weight_warning()


# ── Callbacks: Portfolio management ──

def on_portfolio_switch(event):
    slug = event.new
    state = load_latest_applied_state(slug)
    _load_portfolio_into_ui(slug, state)
    registry = load_portfolio_registry()
    name = next((p["name"] for p in registry if p["slug"] == slug), slug)
    status_pane.object = f"📂 Switched to portfolio: {name}."
    status_pane.alert_type = "info"


def on_create_portfolio(event):
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
    registry = load_portfolio_registry()
    portfolio_selector.options = {p["name"]: p["slug"] for p in registry}
    portfolio_selector.value = entry["slug"]
    _load_portfolio_into_ui(entry["slug"], None)
    new_portfolio_input.value = ""
    status_pane.object = f"✅ Created: {name}. Add stocks and set weights."
    status_pane.alert_type = "success"


def on_delete_portfolio(event):
    slug = portfolio_selector.value
    registry = load_portfolio_registry()
    if len(registry) <= 1:
        status_pane.object = "⚠️ Cannot delete the last remaining portfolio."
        status_pane.alert_type = "danger"
        return
    name = next((p["name"] for p in registry if p["slug"] == slug), slug)
    delete_portfolio_from_registry(slug)
    d = _portfolio_state_dir(slug)
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    registry = load_portfolio_registry()
    portfolio_selector.options = {p["name"]: p["slug"] for p in registry}
    portfolio_selector.value = registry[0]["slug"]
    state = load_latest_applied_state(registry[0]["slug"])
    _load_portfolio_into_ui(registry[0]["slug"], state)
    status_pane.object = f"🗑️ Deleted: {name}. Switched to {registry[0]['name']}."
    status_pane.alert_type = "info"


# ── Callbacks: State history ──

def on_load_state(event):
    filename = state_selector.value
    if not filename:
        status_pane.object = "⚠️ Please select a state from the dropdown."
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
        status_pane.object = f"⚠️ Could not parse: {filename}"
        status_pane.alert_type = "danger"
        return
    _load_portfolio_into_ui(slug, state)

    # Show which tickers were restricted in the historical state (informational)
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        applied = data.get("applied", False)
        hist_restricted = data.get("restricted_tickers", [])
    except Exception:
        applied = False
        hist_restricted = []

    restriction_note = ""
    if hist_restricted:
        restriction_note = f" Restrictions that month: {', '.join(hist_restricted)}."

    if applied:
        status_pane.object = (
            f"📂 Loaded applied state from {filename}. "
            f"Shows post-trade holdings.{restriction_note}"
        )
    else:
        status_pane.object = (
            f"📂 Loaded unapplied state from {filename}. "
            f"Shows pre-trade holdings.{restriction_note}"
        )
    status_pane.alert_type = "info"


# ── Callbacks: Trade restrictions ──

def on_toggle_restriction(event):
    """
    Toggle the restriction status of a ticker. The same button both adds and
    removes restrictions, which is simpler than having separate actions. The
    user types a ticker and clicks — if it's unrestricted, it becomes
    restricted; if already restricted, the restriction is lifted.
    """
    sync_edits_from_table()
    ticker = restrict_input.value.strip().upper()
    if not ticker:
        status_pane.object = "⚠️ Please enter a ticker to restrict/unrestrict."
        status_pane.alert_type = "warning"
        return
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker += ".NS"

    # Find the ticker in portfolio_data
    found = False
    for d in portfolio_data:
        if d["ticker"] == ticker:
            found = True
            was_restricted = d.get("restricted", False)
            d["restricted"] = not was_restricted
            if d["restricted"]:
                status_pane.object = (
                    f"🔒 Restricted {ticker}. It will be frozen during this "
                    f"month's rebalance — no buys or sells."
                )
            else:
                status_pane.object = (
                    f"🔓 Unrestricted {ticker}. It will participate in "
                    f"rebalancing normally."
                )
            status_pane.alert_type = "info"
            break

    if not found:
        status_pane.object = f"⚠️ {ticker} is not in your portfolio."
        status_pane.alert_type = "warning"
        return

    restrict_input.value = ""
    refresh_portfolio_table()


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
        "ticker": ticker, "name": name, "weight": weight_input.value,
        "held": held_input.value, "price": 0.0, "restricted": False,
    })
    refresh_portfolio_table()
    _update_weight_warning()
    ticker_input.value = ""
    total_w = sum(d["weight"] for d in portfolio_data if d["weight"] > 0)
    status_pane.object = f"✅ Added {ticker}. Total weight: {total_w:.1f}%"
    status_pane.alert_type = "success"


def on_remove_stock(event):
    sync_edits_from_table()
    ticker = remove_input.value.strip().upper()
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker += ".NS"
    before = len(portfolio_data)
    portfolio_data[:] = [d for d in portfolio_data if d["ticker"] != ticker]
    if len(portfolio_data) < before:
        refresh_portfolio_table()
        _update_weight_warning()
        remove_input.value = ""
        status_pane.object = f"🗑️ Removed {ticker}."
        status_pane.alert_type = "info"
    else:
        status_pane.object = f"⚠️ {ticker} not found."
        status_pane.alert_type = "warning"


def on_fetch_prices(event):
    sync_edits_from_table()
    status_pane.object = "📡 Fetching prices..."
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
        status_pane.object = f"⚠️ Failed for: {', '.join(failed)}. Edit manually."
        status_pane.alert_type = "warning"
    else:
        status_pane.object = f"✅ {len(tickers)} prices fetched at {datetime.now():%H:%M:%S}."
        status_pane.alert_type = "success"


def on_calculate(event):
    global last_result_df, last_saved_path
    sync_edits_from_table()

    # ── Weight validation ──
    # This is the critical guard: if non-zero weights exceed 100%, the
    # algorithm would compute target values that exceed the total portfolio
    # value, generating impossible buy orders. We block this with a clear
    # message explaining WHY it doesn't make sense.
    total_nonzero_w = sum(d["weight"]
                          for d in portfolio_data if d["weight"] > 0)

    if total_nonzero_w > 100.1:
        status_pane.object = (
            f"⛔ Cannot rebalance: weights sum to {total_nonzero_w:.1f}% which "
            f"exceeds 100%. Each stock's target value would exceed the portfolio's "
            f"total value, making the math impossible. Please reduce weights first."
        )
        status_pane.alert_type = "danger"
        _update_weight_warning()
        return

    if abs(total_nonzero_w - 100) > 0.1:
        status_pane.object = (
            f"⚠️ Non-zero weights sum to {total_nonzero_w:.1f}%, need exactly 100%."
        )
        status_pane.alert_type = "danger"
        _update_weight_warning()
        return

    # Price needed for any stock that's active, a liquidation candidate,
    # or restricted with holdings (we need its value for target computation).
    needs_price = [
        d["ticker"] for d in portfolio_data
        if d["price"] <= 0 and (d["weight"] > 0 or d["held"] > 0)
    ]
    if needs_price:
        status_pane.object = f"⚠️ Missing prices for: {', '.join(needs_price)}."
        status_pane.alert_type = "danger"
        return

    slug = portfolio_selector.value
    df = pd.DataFrame(portfolio_data)
    result = calculate_rebalance(df, monthly_input.value)
    last_result_df = result

    saved = save_rebalance_state(
        slug=slug, amount=monthly_input.value,
        portfolio=[{**d} for d in portfolio_data],
        result=result, applied=False,
    )
    last_saved_path = saved
    refresh_state_dropdown()

    # ── Build HTML results ──
    total_invested = result["buy_cost"].sum()
    current_val = result["current_value"].sum()
    liq_val = result["sell_value"].sum()
    available = monthly_input.value + liq_val
    leftover = available - total_invested

    n_sells = int((result["sell_qty"] > 0).sum())
    n_buys = int((result["buy_qty"] > 0).sum())
    n_restricted = int((result["category"] == "RESTRICTED").sum())
    restricted_val = result.loc[result["category"]
                                == "RESTRICTED", "current_value"].sum()

    # ── Summary cards ──
    liq_card = ""
    if liq_val > 0:
        liq_card = f"""
        <div style="background:#2a1a1a; padding:12px 18px; border-radius:8px;
                    min-width:140px; border:1px solid #7f1d1d;">
          <div style="font-size:11px; color:#f87171; text-transform:uppercase;">
            Liquidation ({n_sells})</div>
          <div style="font-size:20px; font-weight:700; color:#f87171;">₹{liq_val:,.0f}</div>
        </div>"""

    restricted_card = ""
    if n_restricted > 0:
        restricted_card = f"""
        <div style="background:#2a2a1a; padding:12px 18px; border-radius:8px;
                    min-width:140px; border:1px solid #92400e;">
          <div style="font-size:11px; color:#fbbf24; text-transform:uppercase;">
            🔒 Restricted ({n_restricted})</div>
          <div style="font-size:20px; font-weight:700; color:#fbbf24;">₹{restricted_val:,.0f}</div>
          <div style="font-size:11px; color:#92400e; margin-top:2px;">frozen this month</div>
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
      {restricted_card}
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

    # ── Orders table ──
    rows_html = ""
    for _, r in result.iterrows():
        cat = r["category"]
        is_sell = int(r["sell_qty"]) > 0
        is_buy = int(r["buy_qty"]) > 0

        if cat == "RESTRICTED":
            bg = "background:#2a2a15;"
            act = '<span style="color:#fbbf24;font-weight:700;">🔒 RESTRICTED</span>'
            cf = '<span style="color:#92400e;">frozen</span>'
        elif is_sell:
            bg = "background:#2a1015;"
            act = f'<span style="color:#f87171;font-weight:700;">🔴 SELL {int(r["sell_qty"])}</span>'
            cf = f'<span style="color:#f87171;">+₹{r["sell_value"]:,.0f}</span>'
        elif is_buy:
            bg = "background:#0d1a0d;"
            act = f'<span style="color:#34d399;font-weight:700;">🟢 BUY {int(r["buy_qty"])}</span>'
            cf = f'<span style="color:#60a5fa;">−₹{r["buy_cost"]:,.0f}</span>'
        else:
            bg = ""
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
    <table style="width:100%;border-collapse:collapse;font-size:13px;
                  background:#111827;border-radius:8px;overflow:hidden;">
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
    if n_restricted:
        parts.append(f"{n_restricted} restricted")
    status_pane.object = (
        f"🎯 Rebalance: {' + '.join(parts) or 'no orders'}. "
        f"Saved to {saved.name}."
    )
    status_pane.alert_type = "success"


def on_apply(event):
    global last_result_df, last_saved_path
    if last_result_df is None:
        status_pane.object = "⚠️ No rebalance to apply. Calculate first."
        status_pane.alert_type = "warning"
        return

    # Update holdings. Restricted stocks have new_held == held (unchanged).
    for _, r in last_result_df.iterrows():
        for d in portfolio_data:
            if d["ticker"] == r["ticker"]:
                d["held"] = int(r["new_held"])

    # Remove fully liquidated stocks
    liquidated = [d["ticker"]
                  for d in portfolio_data if d["weight"] == 0 and d["held"] == 0]
    portfolio_data[:] = [d for d in portfolio_data if not (
        d["weight"] == 0 and d["held"] == 0)]

    # Clear all restrictions (they're a one-month decision)
    for d in portfolio_data:
        d["restricted"] = False

    if last_saved_path and last_saved_path.exists():
        mark_state_applied(last_saved_path)

    refresh_portfolio_table()
    refresh_state_dropdown()
    _update_weight_warning()
    last_result_df = None
    last_saved_path = None

    liq_msg = f" Removed: {', '.join(liquidated)}." if liquidated else ""
    result_pane.object = "<i>Orders applied. Holdings updated. Restrictions cleared.</i>"
    status_pane.object = f"✅ Holdings updated. Restrictions cleared.{liq_msg}"
    status_pane.alert_type = "success"


# ── Wire up callbacks ──
portfolio_selector.param.watch(on_portfolio_switch, "value")
create_portfolio_btn.on_click(on_create_portfolio)
delete_portfolio_btn.on_click(on_delete_portfolio)
load_state_btn.on_click(on_load_state)
restrict_btn.on_click(on_toggle_restriction)
add_btn.on_click(on_add_stock)
remove_btn.on_click(on_remove_stock)
fetch_btn.on_click(on_fetch_prices)
calc_btn.on_click(on_calculate)
apply_btn.on_click(on_apply)

# Run initial weight check so the banner shows on startup if needed
_update_weight_warning()

# =============================================================================
# 6. LAYOUT
# =============================================================================
header = pn.pane.Markdown(
    "# 📊 Portfolio Rebalancer\n"
    "*Multi-portfolio SIP rebalancing · live NSE prices · trade restrictions*",
    styles={"border-bottom": "1px solid #333", "padding-bottom": "8px"},
)

portfolio_row = pn.Row(
    portfolio_selector, new_portfolio_input, create_portfolio_btn,
    delete_portfolio_btn, align="end",
)
state_row = pn.Row(state_selector, load_state_btn, align="end")
add_row = pn.Row(ticker_input, weight_input, held_input, add_btn, align="end")
remove_row = pn.Row(remove_input, remove_btn, align="end")
restrict_row = pn.Row(restrict_input, restrict_btn, align="end")
controls = pn.Row(monthly_input, fetch_btn, calc_btn, apply_btn, align="end")

explanation = pn.pane.Markdown("""
---
### How it works

1. **Create a portfolio** or pick an existing one from the dropdown.
2. **Define** stocks + weights (non-zero weights must sum to exactly 100%).
3. **Fetch** live prices (or type them manually in the table).
4. **Restrict** — type a ticker and click 🔒 Toggle to freeze it for this month.
   Restricted stocks cannot be bought or sold. Click again to unrestrict.
5. **Calculate** — your monthly SIP goes toward the most underweight unrestricted stocks.
6. **Liquidate** — set weight to **0%** to sell everything (only if not restricted).
7. **Apply** — after executing trades in your broker, click Apply.
   Restrictions auto-clear (they're a one-month decision).
8. **Load history** — use the state dropdown to review or roll back to any past state.

*Ticker format: `.NS` for NSE, `.BO` for BSE (auto-appended if missing).
Restrictions are saved in state files for audit but reset each session.*
""")

layout = pn.Column(
    header,
    pn.pane.Markdown("### 🗂️ Portfolio"),
    portfolio_row,
    status_pane,
    weight_warning_pane,   # persistent warning banner — only visible when weights are invalid
    pn.pane.Markdown("### 📜 State History"),
    state_row,
    pn.pane.Markdown("### ➕ Add / Remove / Restrict Stocks"),
    add_row,
    remove_row,
    restrict_row,
    pn.pane.Markdown("### 📋 Holdings & Actions"),
    controls,
    portfolio_table,
    pn.pane.Markdown("### 🎯 Rebalance Orders"),
    result_pane,
    explanation,
    max_width=1100,
    styles={"margin": "0 auto", "padding": "20px"},
)

layout.servable(title="Portfolio Rebalancer")

if __name__ == "__main__":
    pn.serve(layout, port=5006, show=True, title="Portfolio Rebalancer")
