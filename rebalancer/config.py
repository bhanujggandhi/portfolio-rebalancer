"""
Configuration & Constants
=========================
All tuneable parameters live here. No business logic, no imports from
sibling modules — this is the foundation everything else builds on.
"""

from   pathlib                  import Path

# ── State persistence ──
# STATE_DIR is relative to the project root (where app.py lives), not
# relative to this config file. We resolve it at import time so that
# `panel serve app.py` and `python app.py` both find the same directory.
STATE_DIR = Path(__file__).resolve().parent.parent / "state"

# Maximum number of state files to retain per portfolio. Older files
# are automatically deleted when this limit is exceeded.
MAX_STATE_FILES = 50

# ── Default portfolio ──
# Used when creating a brand-new portfolio or when no state files exist.
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

# ── Panel server ──
PANEL_PORT = 5006
APP_TITLE = "Portfolio Rebalancer"
