"""
Panel UI — Widgets, Callbacks & Layout
=======================================
This module is the "glue" that wires together the ticker, engine, and state
modules into a Panel dashboard. It owns all widget definitions, event
callbacks, and the final layout object.

Imported by app.py, which simply calls layout.servable().
"""

from   datetime                 import datetime

import pandas as pd
import panel as pn

from   .                        import state
from   .config                  import APP_TITLE, DEFAULT_STOCKS
from   .engine                  import calculate_rebalance
from   .ticker                  import (fetch_all_prices, get_nifty_tickers,
                                        name_for_ticker)


# =============================================================================
# Initialization — discover portfolios, load the active one
# =============================================================================

_registry = state.load_portfolio_registry()
_active_slug = _registry[0]["slug"]
_init_state = state.load_latest_applied(_active_slug)

portfolio_data: list[dict] = []
INIT_AMOUNT = 10_0000.0

if _init_state:
    portfolio_data = _init_state[0]
    INIT_AMOUNT = _init_state[1]
else:
    portfolio_data = state.ensure_portfolio_fields(DEFAULT_STOCKS)


# =============================================================================
# Widget Definitions
# =============================================================================

# ── Portfolio management ──
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

# ── State history ──


def _build_state_options(slug: str) -> dict[str, str]:
    entries = state.list_state_files(slug)
    if not entries:
        return {"(No saved states)": ""}
    opts = {}
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

# ── Stock management ──
monthly_input = pn.widgets.FloatInput(
    name="Monthly Investment (₹)", value=INIT_AMOUNT, step=1000, start=0, width=220,
)
ticker_input = pn.widgets.AutocompleteInput(
    name="Add Company", placeholder="e.g. Tata Motors",
    case_sensitive=False, search_strategy='includes',
    options=get_nifty_tickers(), restrict=False, width=250,
)

weight_input = pn.widgets.FloatInput(
    name="Weight %", value=10.0, step=1.0, start=0, end=100, width=100,
)
held_input = pn.widgets.IntInput(
    name="Shares Held", value=0, step=1, start=0, width=100,
)
add_btn = pn.widgets.Button(
    name="➕ Add Stock", button_type="primary", width=140)
remove_input = pn.widgets.AutocompleteInput(
    name="Remove Company", placeholder="e.g. Tata Motors",
    case_sensitive=False, search_strategy='includes',
    options=get_nifty_tickers(), restrict=False, width=250,
)
remove_btn = pn.widgets.Button(
    name="🗑️ Remove", button_type="danger", width=120)

# ── Trade restrictions ──
restrict_input = pn.widgets.AutocompleteInput(
    name="Restrict / Unrestrict Ticker",
    placeholder="e.g. Tata Motors (toggles restriction)",
    case_sensitive=False, search_strategy='includes',
    options=get_nifty_tickers(), restrict=False, width=300,
)
restrict_btn = pn.widgets.Button(
    name="🔒 Toggle Restriction", button_type="warning", width=200,
)

# ── Action buttons ──
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
weight_warning_pane = pn.pane.Alert("", alert_type="danger", visible=False)
result_pane = pn.pane.HTML(
    "<i>No rebalance computed yet.</i>", sizing_mode="stretch_width",
)


# =============================================================================
# Portfolio Table
# =============================================================================

def _make_table_df(data: list[dict]) -> pd.DataFrame:
    """Build the display DataFrame. Restricted column is read-only text."""
    return pd.DataFrame([
        {"ticker": d["ticker"], "name": d["name"], "weight": d["weight"],
         "held": d["held"], "price": d["price"],
         "restricted": "🔒 Yes" if d.get("restricted") else ""}
        for d in data
    ])


portfolio_table = pn.widgets.Tabulator(
    _make_table_df(portfolio_data),
    layout="fit_data_stretch", theme="midnight",
    selectable=False, show_index=False,
    titles={"ticker": "Ticker", "name": "Name", "weight": "Target Wt %",
            "held": "Shares Held", "price": "Price ₹", "restricted": "🔒 Status"},
    editors={"weight": {"type": "number", "step": 0.5, "min": 0, "max": 100},
             "held": {"type": "number", "step": 1, "min": 0},
             "price": {"type": "number", "step": 0.05, "min": 0},
             "restricted": None},
    frozen_columns=["ticker"],
    widths={"restricted": 200, "weight": 110,
            "held": 110, "price": 100, "name": 300},
)


# =============================================================================
# Global State for Apply
# =============================================================================

last_result_df: pd.DataFrame | None = None
last_saved_path = None  # Path | None


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_ticker(raw: str) -> str:
    """Normalize user input to a proper ticker symbol."""
    t = raw.strip().upper()
    if t and not t.endswith(".NS") and not t.endswith(".BO"):
        t += ".NS"
    return t


def refresh_portfolio_table():
    portfolio_table.value = _make_table_df(portfolio_data)


def refresh_state_dropdown():
    state_selector.options = _build_state_options(portfolio_selector.value)


def _update_weight_warning():
    total_w = sum(d["weight"] for d in portfolio_data if d["weight"] > 0)
    if total_w > 100.1:
        weight_warning_pane.object = (
            f"⚠️ <b>Weights exceed 100%!</b> Current total: <b>{total_w:.1f}%</b>. "
            f"Rebalancing is blocked until this is fixed."
        )
        weight_warning_pane.alert_type = "danger"
        weight_warning_pane.visible = True
    elif abs(total_w - 100) > 0.1 and portfolio_data:
        weight_warning_pane.object = (
            f"ℹ️ Non-zero weights sum to <b>{total_w:.1f}%</b> (need 100%)."
        )
        weight_warning_pane.alert_type = "warning"
        weight_warning_pane.visible = True
    else:
        weight_warning_pane.visible = False


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
    _update_weight_warning()


def _load_portfolio_into_ui(slug: str, s):
    global portfolio_data, last_result_df, last_saved_path
    if s:
        portfolio_data[:] = s[0]
        monthly_input.value = s[1]
    else:
        portfolio_data[:] = state.ensure_portfolio_fields(DEFAULT_STOCKS)
        monthly_input.value = 10_000
    last_result_df = None
    last_saved_path = None
    result_pane.object = "<i>No rebalance computed yet.</i>"
    refresh_portfolio_table()
    refresh_state_dropdown()
    _update_weight_warning()


# =============================================================================
# Callbacks — Portfolio Management
# =============================================================================

def on_portfolio_switch(event):
    slug = event.new
    s = state.load_latest_applied(slug)
    _load_portfolio_into_ui(slug, s)
    registry = state.load_portfolio_registry()
    name = next((p["name"] for p in registry if p["slug"] == slug), slug)
    status_pane.object = f"📂 Switched to: {name}."
    status_pane.alert_type = "info"


def on_create_portfolio(event):
    name = new_portfolio_input.value.strip()
    if not name:
        status_pane.object = "⚠️ Enter a portfolio name."
        status_pane.alert_type = "warning"
        return
    entry = state.add_portfolio(name)
    if not entry:
        status_pane.object = f"⚠️ \'{name}\' already exists."
        status_pane.alert_type = "warning"
        return
    registry = state.load_portfolio_registry()
    portfolio_selector.options = {p["name"]: p["slug"] for p in registry}
    portfolio_selector.value = entry["slug"]
    _load_portfolio_into_ui(entry["slug"], None)
    new_portfolio_input.value = ""
    status_pane.object = f"✅ Created: {name}."
    status_pane.alert_type = "success"


def on_delete_portfolio(event):
    slug = portfolio_selector.value
    registry = state.load_portfolio_registry()
    if len(registry) <= 1:
        status_pane.object = "⚠️ Cannot delete the last portfolio."
        status_pane.alert_type = "danger"
        return
    name = next((p["name"] for p in registry if p["slug"] == slug), slug)
    state.delete_portfolio(slug)
    registry = state.load_portfolio_registry()
    portfolio_selector.options = {p["name"]: p["slug"] for p in registry}
    portfolio_selector.value = registry[0]["slug"]
    _load_portfolio_into_ui(
        registry[0]["slug"], state.load_latest_applied(registry[0]["slug"]))
    status_pane.object = f"🗑️ Deleted: {name}."
    status_pane.alert_type = "info"


# =============================================================================
# Callbacks — State History
# =============================================================================

def on_load_state(event):
    filename = state_selector.value
    if not filename:
        status_pane.object = "⚠️ Select a state first."
        status_pane.alert_type = "warning"
        return
    slug = portfolio_selector.value
    from .config import STATE_DIR
    filepath = STATE_DIR / slug / filename
    if not filepath.exists():
        status_pane.object = f"⚠️ File not found: {filename}"
        status_pane.alert_type = "danger"
        return
    s = state.load_state_from_file(filepath)
    if not s:
        status_pane.object = f"⚠️ Could not parse: {filename}"
        status_pane.alert_type = "danger"
        return
    _load_portfolio_into_ui(slug, s)
    meta = state.get_state_metadata(filepath)
    rt = meta["restricted_tickers"]
    rn = f" Restrictions: {', '.join(rt)}." if rt else ""
    tag = "post-trade" if meta["applied"] else "pre-trade"
    status_pane.object = f"📂 Loaded {tag} state from {filename}.{rn}"
    status_pane.alert_type = "info"


# =============================================================================
# Callbacks — Trade Restrictions
# =============================================================================

def on_toggle_restriction(event):
    sync_edits_from_table()
    ticker = _normalize_ticker(restrict_input.value)
    if not ticker:
        status_pane.object = "⚠️ Enter a ticker to restrict/unrestrict."
        status_pane.alert_type = "warning"
        return
    for d in portfolio_data:
        if d["ticker"] == ticker:
            d["restricted"] = not d.get("restricted", False)
            icon = "🔒 Restricted" if d["restricted"] else "🔓 Unrestricted"
            status_pane.object = f"{icon} {ticker}."
            status_pane.alert_type = "info"
            restrict_input.value = ""
            refresh_portfolio_table()
            return
    status_pane.object = f"⚠️ {ticker} not in portfolio."
    status_pane.alert_type = "warning"


# =============================================================================
# Callbacks — Stock Management
# =============================================================================

def on_add_stock(event):
    sync_edits_from_table()
    ticker = _normalize_ticker(ticker_input.value)
    if not ticker:
        status_pane.object = "⚠️ Enter a ticker."
        status_pane.alert_type = "warning"
        return
    if any(d["ticker"] == ticker for d in portfolio_data):
        status_pane.object = f"⚠️ {ticker} already in portfolio."
        status_pane.alert_type = "warning"
        return
    portfolio_data.append({
        "ticker": ticker, "name": name_for_ticker(ticker),
        "weight": weight_input.value, "held": held_input.value,
        "price": 0.0, "restricted": False,
    })
    refresh_portfolio_table()
    _update_weight_warning()
    ticker_input.value = ""
    tw = sum(d["weight"] for d in portfolio_data if d["weight"] > 0)
    status_pane.object = f"✅ Added {ticker}. Weight total: {tw:.1f}%"
    status_pane.alert_type = "success"


def on_remove_stock(event):
    sync_edits_from_table()
    ticker = _normalize_ticker(remove_input.value)
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
    failed = [d["ticker"]
              for d in portfolio_data if prices.get(d["ticker"]) is None]
    for d in portfolio_data:
        p = prices.get(d["ticker"])
        if p is not None:
            d["price"] = p
    refresh_portfolio_table()
    if failed:
        status_pane.object = f"⚠️ Failed: {', '.join(failed)}. Edit manually."
        status_pane.alert_type = "warning"
    else:
        status_pane.object = f"✅ {len(tickers)} prices fetched at {datetime.now():%H:%M:%S}."
        status_pane.alert_type = "success"


# =============================================================================
# Callbacks — Rebalance & Apply
# =============================================================================

def on_calculate(event):
    global last_result_df, last_saved_path
    sync_edits_from_table()

    # ── Weight validation ──
    total_w = sum(d["weight"] for d in portfolio_data if d["weight"] > 0)
    if total_w > 100.1:
        status_pane.object = (
            f"⛔ Weights sum to {total_w:.1f}% (>100%). Fix before rebalancing."
        )
        status_pane.alert_type = "danger"
        _update_weight_warning()
        return
    if abs(total_w - 100) > 0.1:
        status_pane.object = f"⚠️ Weights sum to {total_w:.1f}%, need 100%."
        status_pane.alert_type = "danger"
        _update_weight_warning()
        return

    needs_price = [
        d["ticker"] for d in portfolio_data
        if d["price"] <= 0 and (d["weight"] > 0 or d["held"] > 0)
    ]
    if needs_price:
        status_pane.object = f"⚠️ Missing prices: {', '.join(needs_price)}."
        status_pane.alert_type = "danger"
        return

    slug = portfolio_selector.value
    result = calculate_rebalance(pd.DataFrame(
        portfolio_data), monthly_input.value)
    last_result_df = result

    saved = state.save_state(
        slug=slug, amount=monthly_input.value,
        portfolio=[{**d} for d in portfolio_data],
        result=result, applied=False,
    )
    last_saved_path = saved
    refresh_state_dropdown()

    # ── Render HTML results ──
    result_pane.object = _render_results(result)

    ns = int((result["sell_qty"] > 0).sum())
    nb = int((result["buy_qty"] > 0).sum())
    nr = int((result["category"] == "RESTRICTED").sum())
    parts = []
    if ns:
        parts.append(f"{ns} sell{'s' if ns != 1 else ''}")
    if nb:
        parts.append(f"{nb} buy{'s' if nb != 1 else ''}")
    if nr:
        parts.append(f"{nr} restricted")
    status_pane.object = f"🎯 {' + '.join(parts) or 'No orders'}. Saved to {saved.name}."
    status_pane.alert_type = "success"


def on_apply(event):
    global last_result_df, last_saved_path
    if last_result_df is None:
        status_pane.object = "⚠️ Calculate first."
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
    for d in portfolio_data:
        d["restricted"] = False
    if last_saved_path:
        from pathlib import Path
        p = Path(last_saved_path) if not isinstance(
            last_saved_path, Path) else last_saved_path
        if p.exists():
            state.mark_applied(p)
    refresh_portfolio_table()
    refresh_state_dropdown()
    _update_weight_warning()
    last_result_df = None
    last_saved_path = None
    lm = f" Removed: {', '.join(liquidated)}." if liquidated else ""
    result_pane.object = "<i>Applied. Holdings updated. Restrictions cleared.</i>"
    status_pane.object = f"✅ Done.{lm}"
    status_pane.alert_type = "success"


# =============================================================================
# HTML Result Renderer
# =============================================================================

def _render_results(result: pd.DataFrame) -> str:
    """Build the full HTML output for the rebalance results section."""
    total_invested = result["buy_cost"].sum()
    current_val = result["current_value"].sum()
    liq_val = result["sell_value"].sum()
    available = monthly_input.value + liq_val
    leftover = available - total_invested
    n_sells = int((result["sell_qty"] > 0).sum())
    n_buys = int((result["buy_qty"] > 0).sum())
    n_restricted = int((result["category"] == "RESTRICTED").sum())
    restricted_val = result.loc[result["category"] == "RESTRICTED", "current_value"].sum()

    # ── Summary cards ──
    def _card(label, val, color, border_color="#374151", bg="#111827", extra_content=""):
        return (
            f'<div style="background:{bg};padding:12px 16px;border-radius:8px;min-width:145px;'
            f'border:1px solid {border_color};flex:1;box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">'
            f'<div style="font-size:10px;color:#9ca3af;text-transform:uppercase;font-weight:600;">{label}</div>'
            f'<div style="font-size:18px;font-weight:700;color:{color};margin-top:4px;">{val}</div>'
            f'{extra_content}</div>'
        )

    order_card_breakup = (
        f'<div style="display:flex;gap:8px;margin-top:4px;font-size:10px;font-weight:600;">'
        f'<span style="color:#34d399;">● {n_buys} BUYS</span>'
        f'<span style="color:#f87171;">● {n_sells} SELLS</span></div>'
    )

    summary = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">'
        + _card("Portfolio Value", f"₹{current_val:,.0f}", "#60a5fa")
        + _card("Invested Amount", f"₹{monthly_input.value:,.0f}", "#34d399")
        + _card("Total Orders", f"{n_buys + n_sells} Orders", "#fff", "#4b5563", "#1f2937", order_card_breakup)
        + (_card("Restricted", f"₹{restricted_val:,.0f}", "#fbbf24", "#78350f", "#1a1610") if n_restricted > 0 else "")
        + _card("Total Cash", f"₹{available:,.0f}", "#a78bfa")
        + _card("Deployed", f"₹{total_invested:,.0f}", "#fcd34d")
        + _card("Uninvested", f"₹{leftover:,.0f}", "#f87171" if leftover < 0 else "#94a3b8")
        + "</div>"
    )

    # ── Table Headers ──
    hdrs = [("Ticker", 0), ("Name", 0), ("Price", 1), ("Target", 1), ("Current", 1), 
            ("Drift", 1), ("Action", 0), ("Cash Flow", 1), ("New Held", 1), ("New Wt", 1)]
    
    header_html = "".join([
        f'<th style="padding:12px 8px;text-align:{"right" if a else "left"};color:#6b7280;'
        f'font-size:11px;text-transform:uppercase;border-bottom:1px solid #374151;">{h}</th>' 
        for h, a in hdrs
    ])

    rows = ""
    for _, r in result.iterrows():
        is_sell = int(r["sell_qty"]) > 0
        is_buy = int(r["buy_qty"]) > 0
        cat = r["category"]
        
        row_style = ("transparent", "transparent")
        if cat == "RESTRICTED":
            row_style = ("rgba(251, 191, 36, 0.08)", "#f59e0b")
            act = '<span style="background:#451a03;color:#fbbf24;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;border:1px solid #78350f;display:inline-block;width:95px;text-align:center;">🔒 FROZEN</span>'
            cf = '<span style="color:#92400e;">—</span>'
        elif is_sell:
            row_style = ("rgba(248, 113, 113, 0.08)", "#ef4444")
            act = f'<span style="background:#450a0a;color:#f87171;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;border:1px solid #7f1d1d;display:inline-block;width:95px;text-align:center;">🔴 SELL {int(r["sell_qty"])}</span>'
            cf = f'<span style="color:#f87171;font-weight:600;">+₹{r["sell_value"]:,.0f}</span>'
        elif is_buy:
            row_style = ("rgba(16, 185, 129, 0.08)", "#10b981")
            act = f'<span style="background:#064e3b;color:#34d399;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;border:1px solid #065f46;display:inline-block;width:95px;text-align:center;">🟢 BUY {int(r["buy_qty"])}</span>'
            cf = f'<span style="color:#60a5fa;font-weight:600;">−₹{r["buy_cost"]:,.0f}</span>'
        else:
            act = '<span style="color:#4b5563;">—</span>'
            cf = '<span style="color:#4b5563;">—</span>'

        dv = r["drift"]
        dc = "#f87171" if dv < -1 else "#34d399" if dv > 1 else "#9ca3af"

        rows += (
            f'<tr style="background:{row_style[0]};border-bottom:1px solid #1f2937;">'
            f'<td style="padding:12px 8px;font-weight:700;color:#fff;border-left:4px solid {row_style[1]};">{r["ticker"]}</td>'
            f'<td style="padding:12px 8px;color:#e5e7eb;font-size:12px;">{r["name"]}</td>'
            f'<td style="padding:12px 8px;text-align:right;color:#d1d5db;">₹{r["price"]:,.2f}</td>'
            f'<td style="padding:12px 8px;text-align:right;color:#fbbf24;font-weight:600;">{r["weight"]:.1f}%</td>'
            f'<td style="padding:12px 8px;text-align:right;color:#9ca3af;">{r["current_weight"]:.1f}%</td>'
            f'<td style="padding:12px 8px;text-align:right;color:{dc};font-weight:600;">{"+" if dv > 0 else ""}{dv:.1f}%</td>'
            f'<td style="padding:12px 8px;">{act}</td>'
            f'<td style="padding:12px 8px;text-align:right;">{cf}</td>'
            f'<td style="padding:12px 8px;text-align:right;color:#fff;">{int(r["new_held"])}</td>'
            f'<td style="padding:12px 8px;text-align:right;color:#fff;font-weight:700;">{r["new_weight"]:.1f}%</td>'
            f"</tr>"
        )

    table = (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;background:#0b0f1a;font-family:ui-sans-serif,system-ui,sans-serif;">'
        f'<thead><tr style="background:#020617;">{header_html}</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )
    
    return f'<div style="background:#0b0f1a;padding:10px;border-radius:12px;">{summary}{table}</div>'

# =============================================================================
# Wire Up Callbacks
# =============================================================================

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

_update_weight_warning()


# =============================================================================
# Layout
# =============================================================================

header = pn.pane.Markdown(
    f"# 📊 {APP_TITLE}\n"
    "*Multi-portfolio SIP rebalancing · live NSE prices · trade restrictions*",
    styles={"border-bottom": "1px solid #333", "padding-bottom": "8px"},
)

explanation = pn.pane.Markdown("""
---
### How it works
1. **Create a portfolio** or pick one from the dropdown.
2. **Define** stocks + weights (non-zero weights must sum to 100%).
3. **Fetch** prices or type them manually.
4. **Restrict** tickers you cannot trade this month (toggle via 🔒 button).
5. **Calculate** — SIP goes to the most underweight unrestricted stocks.
6. **Liquidate** — set weight to 0% to sell (if not restricted).
7. **Apply** — after executing trades, click Apply. Restrictions auto-clear.
8. **History** — load any past state to review or roll back.
""")

layout = pn.Column(
    header,
    pn.pane.Markdown("### 🗂️ Portfolio"),
    pn.Row(portfolio_selector, new_portfolio_input, create_portfolio_btn,
           delete_portfolio_btn),
    status_pane,
    weight_warning_pane,
    pn.pane.Markdown("### 📜 State History"),
    pn.Row(state_selector, load_state_btn),
    pn.pane.Markdown("### ➕ Add / Remove / Restrict"),
    pn.Row(ticker_input, weight_input, held_input, add_btn),
    pn.Row(remove_input, remove_btn),
    pn.Row(restrict_input, restrict_btn),
    pn.pane.Markdown("### 📋 Holdings & Actions"),
    pn.Row(monthly_input, fetch_btn, calc_btn, apply_btn),
    portfolio_table,
    pn.pane.Markdown("### 🎯 Rebalance Orders"),
    result_pane,
    explanation,
    max_width=1300,
    styles={"margin": "0 auto", "padding": "20px"},
)
