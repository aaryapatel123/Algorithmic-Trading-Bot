# Trading Bot

Python algorithmic trading bot implementing a combined three-strategy momentum system for US equities, using Alpaca paper trading.

## Strategy

The bot uses a **Combined Momentum System** that layers three signals to decide what to hold each month:

### 1. S&P 500 Stock Momentum
Every month on the first trading day, ranks all S&P 500 constituents by 12-month return and buys the top 3 stocks.

### 2. Dual Momentum (SPY vs AGG)
Compares the 12-month return of SPY (US stocks) against AGG (US bonds). If bonds are outperforming stocks, the portfolio rotates to AGG instead of equities.

### 3. 200-Day MA Crash Protection
If SPY closes below its 200-day moving average, the portfolio exits to cash or AGG to protect against sustained downtrends.

**Rebalancing schedule:** First trading day of each month.

## Backtest Results (2015–2026)

| Metric | Strategy | SPY Buy & Hold |
|--------|----------|----------------|
| Total Return | +772% | +288% |
| Sharpe Ratio | 0.59 | 0.55 |
| Max Drawdown | 42.4% | — |

Notable: the strategy avoided the COVID crash (2020), the 2018 Q4 selloff, and the 2022 bear market by rotating to bonds/cash via the crash protection filters.

## Prerequisites

- Python 3.9+
- [Alpaca paper trading account](https://app.alpaca.markets) (free)
- API key and secret from the Alpaca dashboard (Paper Trading section)

## Setup

```bash
# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env
# Edit .env with your ALPACA_API_KEY and ALPACA_SECRET_KEY
```

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | required | Alpaca API key |
| `ALPACA_SECRET_KEY` | required | Alpaca API secret |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Paper trading URL |
| `MAX_POSITION_PCT` | `0.10` | Max portfolio % per position |
| `MAX_TOTAL_EXPOSURE_PCT` | `0.50` | Max total equity exposure |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DB_PATH` | `trading_bot.db` | SQLite database path |

## Running the Bot

```bash
source venv/bin/activate
python -m src.main
```

Stop with `Ctrl+C` — the bot shuts down gracefully.

## Backtesting

```bash
source venv/bin/activate

# Run the combined momentum strategy backtest (2015–2026)
python scripts/run_combined.py

# Run the original MA crossover backtest
python scripts/run_backtest.py

# Custom parameters for MA crossover backtest
python scripts/run_backtest.py --symbol MSFT --start 2022-01-01 --end 2024-01-01 \
    --short-ma 10 --long-ma 30 --cash 50000
```

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

## Architecture

```
src/
├── main.py              # Orchestration loop — market hours → fetch → signal → order → persist
├── config.py            # Environment-based configuration with validation
├── strategy/
│   └── ma_crossover.py  # SMA crossover signal computation
├── broker/
│   └── alpaca_client.py # Alpaca REST API wrapper (orders, positions, clock)
├── data/
│   └── market_data.py   # Historical bars via Alpaca (yfinance fallback)
├── models/
│   └── trade.py         # SQLite persistence: signals, trades, bot state
├── risk/
│   └── position_sizer.py # Position sizing with max exposure limits
└── utils/
    └── logging_config.py # Rotating file + console JSON logging

backtest/
├── bt_ma_crossover.py   # Backtrader MA crossover strategy with Sharpe/drawdown analyzers
└── bt_combined.py       # Combined momentum strategy backtest (dual momentum + MA filter)

scripts/
├── run_backtest.py      # CLI runner for MA crossover backtest
└── run_combined.py      # CLI runner for combined momentum strategy backtest
```

## Risk Controls

- No pyramiding (won't add to an existing position)
- Per-symbol position limit (default 10% of equity)
- Total exposure cap (default 50% of equity)
- Market-hours enforcement (no orders when market is closed)
- Duplicate signal prevention (won't re-fire for the same bar after restart)
- Automatic rotation to bonds/cash during downtrends (200-day MA filter)
