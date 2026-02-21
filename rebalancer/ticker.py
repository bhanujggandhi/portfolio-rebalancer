"""
Ticker Name Resolution & Price Fetching
========================================
Two responsibilities:

1. Resolving ticker symbols to human-readable company names (e.g.,
   "RELIANCE.NS" → "Reliance Industries Ltd"). Results are cached
   in-memory and persisted in state files to avoid repeated API calls.

2. Fetching live prices via yFinance. Uses batch downloading
   (yf.download) for speed — one HTTP call for all tickers instead of
   sequential per-ticker requests.
"""

from   functools                import cache

import pandas as pd
import yfinance as yf

from   rebalancer.config        import DATA_DIR

# ── In-memory name cache ──
# This is a module-level dict that acts as the single source of truth for
# ticker → display name mappings during the process lifetime. It gets
# populated from three sources:
#   1. yFinance API (when a new ticker is first encountered)
#   2. State files on disk (restored at startup to avoid API calls)
#   3. Manual insertion via populate_name_cache()
_name_cache: dict[str, str] = {}

@cache
def get_nifty_tickers():
    """Get dataframe for nifty data"""
    file_path = DATA_DIR / "nifty_metadata.csv"
    df = pd.read_csv(file_path)
    return df.set_index("NAME OF COMPANY")["SYMBOL"].to_dict()

def name_for_ticker(ticker: str) -> str:
    """
    Get the human-readable display name for a ticker symbol.

    Checks the in-memory cache first. If the ticker hasn't been seen before,
    makes a single yFinance API call to fetch the name and caches it. This
    means each ticker only triggers one API call across the entire process
    lifetime.

    Falls back to the raw symbol (minus exchange suffix) if the API call
    fails or returns no name.
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
    fallback = ticker.replace(".NS", "").replace(".BO", "")
    _name_cache[ticker] = fallback
    return fallback


def get_name_cache() -> dict[str, str]:
    """Return a reference to the name cache (for state persistence)."""
    return _name_cache


def populate_name_cache(names: dict[str, str]):
    """
    Bulk-insert names into the cache. Used when loading state files that
    contain saved ticker names, so startup doesn't need any API calls.
    Only inserts names that aren't already cached (existing entries take
    priority since they may be more recent).
    """
    for ticker, name in names.items():
        if name and ticker not in _name_cache:
            _name_cache[ticker] = name


def fetch_all_prices(tickers: list[str]) -> dict[str, float | None]:
    """
    Fetch current prices for multiple tickers in a single batch call.

    Uses yf.download() which sends one HTTP request to Yahoo Finance for
    all symbols at once — dramatically faster than per-ticker Ticker.info
    calls (2-4 seconds vs 10-25 seconds for 8 stocks).

    Returns a dict mapping each ticker to its price (or None if unavailable).
    Falls back to sequential per-ticker fetching if the batch call fails.
    """
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
        print(f"[WARN] Batch download failed: {e}. Falling back to per-ticker.")
        for t in tickers:
            prices[t] = _fetch_price_single(t)
    return prices


def _fetch_price_single(ticker_symbol: str) -> float | None:
    """
    Fallback price fetcher for a single ticker. Only used when the batch
    yf.download() fails entirely. Tries Ticker.info first (most real-time),
    then falls back to the last close from 5-day history (works on weekends).
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
