"""
State Persistence Layer
========================
Handles all file I/O for portfolio state: the portfolio registry, per-
portfolio rebalance state files, saving, loading, and cleanup.

Key design decisions:

1. **Applied flag.** State files are saved with `applied=False` at
   calculate time. Only when the user clicks Apply does the file get
   updated to `applied=True`. On startup, we only auto-load applied
   states, preventing phantom trades from corrupting holdings.

2. **Backward compatibility.** Every .get() call uses safe defaults so
   that state files from older schema versions load without errors.

3. **Restrictions are ephemeral.** They're saved for audit but NOT
   restored on load — they reset each session because they're a
   per-month decision.

4. **Ticker names are persisted.** Saved in state files so that startup
   doesn't require yFinance API calls just to display company names.
"""

from   datetime                 import datetime
import json
from   pathlib                  import Path
import re
import shutil


from   .config                  import (DEFAULT_PORTFOLIO_NAME,
                                        MAX_STATE_FILES, STATE_DIR)
from   .ticker                  import (get_name_cache, name_for_ticker,
                                        populate_name_cache)


# =============================================================================
# Utility helpers
# =============================================================================

def _slugify(name: str) -> str:
    """Convert a human-readable name into a filesystem-safe slug."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unnamed"


def _to_jsonable(v):
    """Convert numpy/pandas scalar types to JSON-safe Python types."""
    if hasattr(v, "item"):       # numpy int64, float64, etc.
        v = v.item()
    if isinstance(v, float):
        return None if v != v else round(v, 4)   # NaN → None
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, bool):
        return bool(v)
    return v


def ensure_portfolio_fields(rows: list[dict]) -> list[dict]:
    """
    Ensure every portfolio row has all required fields with safe defaults.
    This is the backward-compatibility layer — old state files may lack
    'restricted', 'name', or other fields added in later versions.
    """
    cache = get_name_cache()
    out = []
    for r in rows:
        row = {**r}
        row.setdefault("restricted", False)
        row.setdefault("price", 0.0)
        row.setdefault("held", 0)
        row.setdefault("weight", 0.0)
        row["name"] = (
            row.get("name", cache.get(row['ticker'], 
            name_for_ticker(row['ticker'])))
        )
        out.append(row)
    return out


# =============================================================================
# Portfolio Registry (portfolios.json)
# =============================================================================

def _registry_path() -> Path:
    return STATE_DIR / "portfolios.json"


def load_portfolio_registry() -> list[dict]:
    """Load the list of all portfolios. Creates a default one if none exist."""
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
    _save_registry(default)
    return default


def _save_registry(portfolios: list[dict]):
    STATE_DIR.mkdir(exist_ok=True)
    with open(_registry_path(), "w", encoding="utf-8") as f:
        json.dump({"portfolios": portfolios}, f, indent=2)


def add_portfolio(name: str) -> dict | None:
    """Add a portfolio to the registry. Returns the entry or None if duplicate."""
    slug = _slugify(name)
    registry = load_portfolio_registry()
    if any(p["slug"] == slug for p in registry):
        return None
    entry = {"slug": slug, "name": name.strip(
    ), "created": datetime.now().isoformat()}
    registry.append(entry)
    _save_registry(registry)
    return entry


def delete_portfolio(slug: str) -> bool:
    """Remove a portfolio from the registry and delete its state directory."""
    registry = load_portfolio_registry()
    before = len(registry)
    registry = [p for p in registry if p["slug"] != slug]
    if len(registry) < before:
        _save_registry(registry)
        d = STATE_DIR / slug
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        return True
    return False


# =============================================================================
# Per-Portfolio State Files
# =============================================================================

def _state_dir(slug: str) -> Path:
    return STATE_DIR / slug


def list_state_files(slug: str) -> list[dict]:
    """
    List all state files for a portfolio, most recent first.
    Each entry contains: path, filename, timestamp, applied, label.
    """
    d = _state_dir(slug)
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
        n_restricted = len(data.get("restricted_tickers", []))
        try:
            ts = datetime.fromisoformat(ts_str)
            ts_display = ts.strftime("%d %b %Y, %H:%M")
        except (ValueError, TypeError):
            ts_display = fp.stem
        status = "✅ Applied" if applied else "⏳ Calculated"
        rn = f" | 🔒{n_restricted}" if n_restricted else ""
        label = f"{ts_display} | {status} | Val: ₹{cur_val:,.0f} | SIP: ₹{amount:,.0f}{rn}"
        entries.append({
            "path": fp, "filename": fp.name,
            "timestamp": ts_str, "applied": applied, "label": label,
        })
    return entries


def save_state(
    slug: str, amount: float, portfolio: list[dict],
    result, applied: bool = False,
) -> Path:
    """
    Save a rebalance snapshot to a timestamped JSON file.

    Parameters
    ----------
    slug : str          Portfolio slug (directory name).
    amount : float      Monthly SIP amount.
    portfolio : list    Portfolio data (pre-rebalance).
    result : DataFrame  Output of calculate_rebalance().
    applied : bool      Whether this state has been confirmed by the user.
    """
    d = _state_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = d / f"rebalance_{ts}.json"

    result_records = [
        {k: _to_jsonable(v) for k, v in row.to_dict().items()}
        for _, row in result.iterrows()
    ]
    liq_total = float(result["sell_value"].sum())
    cache = get_name_cache()
    ticker_names = {r["ticker"]: cache.get(
        r["ticker"], r.get("name", "")) for r in portfolio}
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
    _cleanup(slug)
    return filepath


def mark_applied(filepath: Path):
    """Flip a state file's applied flag to True and record the timestamp."""
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
    Load portfolio state from a specific file.

    Applied files → use new_held (post-trade holdings).
    Unapplied files → use portfolio_before (avoids loading phantom trades).
    Restrictions always reset to False (they're per-session).
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    applied = data.get("applied", False)
    populate_name_cache(data.get("ticker_names", {}))
    amount = float(data.get("amount", 10_000))

    if applied:
        result = data.get("rebalance_result", [])
        if not result:
            return None
        portfolio = [
            {"ticker": r["ticker"], "weight": r.get("weight", 0),
             "held": int(r.get("new_held", r.get("held", 0))),
             "price": float(r.get("price", 0)), "restricted": r.get("restricted", False)}
            for r in result
        ]
    else:
        before = data.get("portfolio_before", [])
        if not before:
            return None
        portfolio = [
            {"ticker": r["ticker"], "weight": r.get("weight", 0),
             "held": int(r.get("held", 0)),
             "price": float(r.get("price", 0)), "restricted": r.get("restricted", False)}
            for r in before
        ]
    return (ensure_portfolio_fields(portfolio), amount)


def load_latest_applied(slug: str, applied_only=False) -> tuple[list[dict], float] | None:
    """Load the most recent applied state for a portfolio."""
    if applied_only:
        for entry in list_state_files(slug):
            if entry["applied"]:
                return load_state_from_file(entry["path"])
        return None
    state_files = list_state_files(slug)
    if state_files:
        return load_state_from_file(state_files[0]['path'])
    return None


def get_state_metadata(filepath: Path) -> dict:
    """Read metadata from a state file (applied status, restrictions, etc.)."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "applied": data.get("applied", False),
            "restricted_tickers": data.get("restricted_tickers", []),
        }
    except Exception:
        return {"applied": False, "restricted_tickers": []}


def _cleanup(slug: str):
    """Remove old state files beyond the retention limit."""
    d = _state_dir(slug)
    if not d.exists():
        return
    files = sorted(d.glob("rebalance_*.json"))
    if len(files) > MAX_STATE_FILES:
        for old in files[: len(files) - MAX_STATE_FILES]:
            try:
                old.unlink()
            except OSError:
                pass
