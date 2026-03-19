# Trading Bot

Python algorithmic trading bot implementing a Simple Moving Average (SMA) crossover strategy for US equities, using Alpaca paper trading.

## Strategy

- **BUY** when the short-period SMA crosses above the long-period SMA (golden cross)
- **SELL** when the short-period SMA crosses below the long-period SMA (death cross)
- Default periods: 20-day short, 50-day long

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
# Edit .env with your APCA_API_KEY_ID and APCA_API_SECRET_KEY
```

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `APCA_API_KEY_ID` | required | Alpaca API key |
| `APCA_API_SECRET_KEY` | required | Alpaca API secret |
| `APCA_API_BASE_URL` | `https://paper-api.alpaca.markets` | Paper trading URL |
| `SYMBOLS` | required | Comma-separated symbols, e.g. `AAPL,MSFT` |
| `SHORT_MA_PERIOD` | `20` | Short SMA period |
| `LONG_MA_PERIOD` | `50` | Long SMA period |
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

# Default: AAPL, 2-year lookback, 20/50 SMA, $100k starting cash
python scripts/run_backtest.py

# Custom parameters
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
│   └── ma_crossover.py  # Pure SMA crossover signal computation
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
└── bt_ma_crossover.py   # Backtrader strategy with Sharpe/drawdown analyzers

scripts/
└── run_backtest.py      # CLI backtest runner
```

## Risk Controls

- No pyramiding (won't add to an existing position)
- Per-symbol position limit (default 10% of equity)
- Total exposure cap (default 50% of equity)
- Market-hours enforcement (no orders when market is closed)
- Duplicate signal prevention (won't re-fire for the same bar after restart)
