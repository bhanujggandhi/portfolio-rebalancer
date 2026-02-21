# 📊 Portfolio Rebalancer

A multi-portfolio SIP-style equity rebalancer with live NSE/BSE prices,
trade restriction support, liquidation logic, and persistent state history.

Built with [Panel](https://panel.holoviz.org/) + [yFinance](https://github.com/ranaroussi/yfinance).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
panel serve app.py --show

# Or simply:
python app.py
```

The app opens at [http://localhost:5006](http://localhost:5006).

---

## Features

### Core Rebalancing

The rebalancer uses a **deficit-proportional allocation** algorithm. Instead
of naively splitting your monthly SIP by target weights, it looks at how far
each stock has drifted from its target and directs more money toward the most
underweight positions. This naturally corrects drift over time without ever
needing to sell.

**Example:** You invest ₹10,000/month into an 8-stock portfolio. Reliance
has drifted to 25% (target: 20%) while ITC dropped to 6% (target: 9%). The
algorithm allocates ₹0 to Reliance (already overweight) and proportionally
more to ITC and other underweight stocks.

### Liquidation

Set any stock's target weight to **0%** to signal a full exit. The algorithm
generates a SELL order for all held shares and adds the proceeds to your
available cash pool, which then gets redistributed across the remaining
active stocks.

### Trade Restrictions

Some months you may be unable to trade certain tickers (insider blackout
windows, compliance restrictions, personal preference). Type the ticker
into the restriction field and click 🔒 Toggle. Restricted stocks are
completely frozen — no buys or sells. Their value is still counted in
portfolio totals so that target calculations for other stocks remain accurate.

Restrictions are **per-session** — they auto-clear when you Apply orders or
restart the app. They're saved in state files for audit purposes.

### Multiple Portfolios

Create and manage separate portfolios, each with its own stocks, weights,
holdings, and state history. Portfolios are stored in isolated subdirectories
under `state/`. Switching portfolios completely swaps the UI context.

### State History

Every rebalance calculation is saved as a timestamped JSON file. The state
history dropdown lets you:

- **Review** past calculations (see what orders were generated)
- **Roll back** to a previous state (undo a mistake)
- **Audit** which tickers were restricted in any given month

States are marked as "Applied" only after you explicitly confirm, preventing
phantom trades from polluting your history.

---

## Project Structure

```
portfolio_rebalancer/
├── README.md               ← this file
├── requirements.txt        ← Python dependencies
├── app.py                  ← entry point (run this)
└── rebalancer/
    ├── __init__.py          ← package init, version info
    ├── config.py            ← constants, default portfolio, state paths
    ├── ticker.py            ← yFinance name resolution & batch price fetching
    ├── engine.py            ← rebalancing algorithm (pure function, no I/O)
    ├── state.py             ← JSON-based persistence (registry, state files)
    └── ui.py                ← Panel UI (widgets, callbacks, layout)
```

### Module Responsibilities

| Module       | Depends On           | Responsibility |
|-------------|----------------------|----------------|
| `config.py` | nothing              | Constants, file paths, default stock list |
| `ticker.py` | `config`             | Ticker name cache, batch price fetching via yFinance |
| `engine.py` | nothing (pure logic) | The `calculate_rebalance()` function |
| `state.py`  | `config`, `ticker`   | Portfolio registry, state file CRUD, backward compat |
| `ui.py`     | all of the above     | Panel widgets, event callbacks, HTML rendering |
| `app.py`    | `ui`                 | Entry point — calls `layout.servable()` |

### Why This Separation?

The key benefit is **testability**. The `engine` module is a pure function —
give it a DataFrame and a cash amount, get back a DataFrame with orders. You
can unit-test it without any file I/O, network calls, or UI framework. The
`state` module can be tested with temporary directories. The `ticker` module
can be tested by mocking yFinance responses.

---

## Data Storage

All state is stored under `state/` (created automatically on first run):

```
state/
├── portfolios.json                     ← registry of all portfolios
├── default-portfolio/
│   ├── rebalance_2025-01-15_120000.json
│   └── rebalance_2025-02-15_093000.json
└── growth-picks/
    └── rebalance_2025-02-01_110000.json
```

Each state file is a self-contained JSON snapshot that includes the portfolio
before the rebalance, the computed result, a summary, the ticker name cache,
and which tickers were restricted that month.

Old state files are automatically pruned (default: keep last 50 per portfolio).

---

## Monthly Workflow

1. Open the app. Your last Applied state loads automatically.
2. **(Optional)** Adjust weights if your strategy has changed.
3. **(Optional)** Restrict any tickers you can't trade this month.
4. Click **📡 Fetch Prices** to pull live data from Yahoo Finance.
5. Click **🎯 Calculate Rebalance** to see buy/sell orders.
6. Execute the trades in your broker (Zerodha, Groww, etc.).
7. Click **✅ Apply Orders** to update holdings and save state.
8. Come back next month and repeat from step 1.

---

## Configuration

Edit `rebalancer/config.py` to change:

- `DEFAULT_PORTFOLIO_NAME` — name of the portfolio created on first run
- `DEFAULT_STOCKS` — the initial stock list for new portfolios
- `MAX_STATE_FILES` — how many history files to keep per portfolio (default: 50)
- `STATE_DIR` — where state files are stored (default: `./state/`)

---

## Ticker Format

- NSE tickers use the `.NS` suffix: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- BSE tickers use the `.BO` suffix: `RELIANCE.BO`
- If you type a bare symbol (e.g., `RELIANCE`), `.NS` is auto-appended.

---

## Limitations & Known Issues

- **No fractional shares.** NSE doesn't support fractional trading, so buy
  quantities are rounded down to whole shares. The remainder shows up as
  "Uninvested Cash."
- **Price accuracy.** yFinance pulls from Yahoo Finance, which may lag by
  15-20 minutes during market hours. For end-of-day rebalancing this is fine.
- **No broker integration.** You execute trades manually. This is intentional
  for a POC — adding broker APIs (Zerodha Kite, etc.) is a natural next step.
- **Single-user.** The state directory is a local folder. For multi-user
  access, you'd need a database backend.

---

## License

MIT — use freely for personal or commercial purposes.
