# Trading Bot

Python algorithmic trading system for US equities with two independent parts:

1. **Live Alpaca bot** (`src/`) — an automated paper-trading bot that rebalances monthly using a three-signal combined momentum strategy.
2. **Signal generator / backtest suite** (`scripts/`, `backtest/`) — the actively-maintained "definitive" strategy (keep8 + smoothing + position/sector caps + a catastrophic stop + 10% gold sleeve), used to produce a manual buy/sell list rather than to place trades automatically.

These two strategies are related but have diverged — see [Two strategies, two purposes](#two-strategies-two-purposes) below before assuming a change to one affects the other.

## Prerequisites

- Python 3.9+
- [Alpaca paper trading account](https://app.alpaca.markets) (free) — only needed for the live bot
- Gmail account + [app password](https://myaccount.google.com/apppasswords) — only needed for email alerts from the signal generator

## Setup

```bash
# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env
```

Edit `.env` and set at minimum `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (required by `src/config.py`) if you plan to run the live bot. Note the example file uses the `APCA_*` variable names Alpaca's own docs use — the code in this repo reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL`, so use those names in your `.env`.

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|--------------|
| `ALPACA_API_KEY` | required | Alpaca API key |
| `ALPACA_SECRET_KEY` | required | Alpaca API secret |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Paper trading endpoint |
| `SYMBOLS` | empty (fetches S&P 500) | Comma-separated universe override |
| `TOP_N` | `3` | Number of top-momentum stocks to hold |
| `MOMENTUM_DAYS` | `252` | Lookback window (trading days) for momentum ranking |
| `MA_PERIOD` | `200` | SPY moving-average period used as the crash filter |
| `MAX_POSITION_PCT` | `0.40` | Max % of equity in a single position |
| `MAX_TOTAL_EXPOSURE_PCT` | `0.95` | Max total equity exposure |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DB_PATH` | `trading_bot.db` | SQLite database path (trades, signals, bot state) |

For emailing buy lists from `signal_generator.py`, also set `GMAIL_USER` and `GMAIL_APP_PASSWORD`.

## Two strategies, two purposes

### 1. Live bot (`src/main.py`)

Runs continuously, checks once a minute during market hours, and rebalances on the first trading day of each new month via the Alpaca paper API.

- **Dual momentum** — SPY vs AGG 12-month return decides stocks vs. bonds
- **200-day MA filter** — SPY below its 200MA forces a defensive (AGG) regime
- **Top-N stock selection** — ranks the universe by 12-month momentum, equal-weights the top `TOP_N`

```bash
source venv/bin/activate
python -m src.main
```

Stop with `Ctrl+C` — the bot shuts down gracefully and logs final account state.

### 2. Signal generator (`scripts/signal_generator.py`)

Computes the definitive strategy — **keep8 hysteresis + EWMA smoothing (0.5) + 25% position cap + 80% sector cap + 25% catastrophic stop + 10% GLD sleeve**, allocated 85% equity / 10% gold / 5% cash — and prints (or emails) a human-readable buy/sell list. It does **not** place trades automatically; it's a manual decision aid for a brokerage account that isn't API-tradable. State carries forward between runs in `scripts/.signal_state.json` so the momentum smoothing is continuous month to month.

```bash
source venv/bin/activate
python scripts/signal_generator.py --portfolio-value 50000
python scripts/signal_generator.py --portfolio-value 50000 --email you@gmail.com
```

For a fresh deployment with no existing holdings (every line is a buy), use:

```bash
./buy-list 50000
# or directly:
python scripts/buy_list.py --portfolio-value 50000
```

## Backtesting

```bash
source venv/bin/activate

# Combined 3-strategy momentum system (src/main.py's strategy)
python scripts/run_combined.py

# Original single-symbol MA crossover / RSI backtest
python scripts/run_backtest.py --symbol AAPL
```

`scripts/` also contains a number of research/comparison scripts used to validate the definitive strategy's parameters (`compare_sector_cap.py`, `compare_hysteresis.py`, `compare_topn_cap.py`, `sleeve_eval.py`, `walk_forward_validation.py`, `stress_test_2008.py`, `tax_analysis.py`, etc.) — these are exploratory tools, not part of the running system.

### Backtest results

**Definitive strategy** (`top_n=5, keep_n=8, max_weight=0.25, weight_smoothing=0.5, sector_cap=0.80, cat_stop_pct=0.25, sleeve_pct=0.10` GLD), 136–158 stock large-cap universe, judged primarily on Calmar (CAGR ÷ MaxDD) and Sortino rather than Sharpe — this strategy runs concentrated momentum bets, and Sharpe penalizes the upside volatility that *is* the alpha here just as much as the downside:

| Window | CAGR | Sortino | Calmar | Max Drawdown | Stop triggers |
|--------|------|---------|--------|---------------|----------------|
| 2021–2026 (out-of-sample) | 37.2% | 3.216 | 3.130 | 11.9% | 2 |
| 2015–2026 (full) | 32.6% | 2.536 | 1.271 | 25.7% | 10 |
| 2007–2026 (incl. GFC) | ~17.4%¹ | — | — | 51.6% | 11 |

¹ The 2007–2026 CAGR comes from a separate daily-bar GFC stress test (`scripts/stress_test_2008.py`) rather than the monthly-return pipeline used for the other two rows, so it isn't directly comparable metric-for-metric — treat it as the worst-case drawdown reference, not an apples-to-apples CAGR.

For reference, SPY buy-and-hold over 2015–2026 scores Sortino ~1.01 / Calmar ~0.56 on the same metrics — the strategy's edge is concentrated in risk-adjusted return, not just raw return.

**On Sharpe specifically:** the fully-loaded definitive config above (hysteresis + smoothing + GLD sleeve + catastrophic stop) backtests to a **Sharpe of ~0.79** over 2015–2026, not the ~0.90 figure from earlier research. That 0.899 number belongs to a much simpler, earlier-stage version of the strategy (plain momentum + inverse-vol weighting + a 25% cap, with no hysteresis, smoothing, gold sleeve, or stop). Each layer added after that baseline was deliberately chosen because it improved Sortino/Calmar/MaxDD and/or absolute return, even where it cost Sharpe — hysteresis alone cut Sharpe from 0.90 to 0.77 while raising 2015–2026 total return from +1776% to +2763% at unchanged drawdown, which is the tradeoff this project optimizes for.

**GFC stress test** (2007–2010, survivorship-bias-free 136-stock universe): the fully-loaded definitive config returns **+13.87% CAGR** with a **51.6% max drawdown**, versus **+13.33% CAGR / 47.0% MaxDD** for the same config *without* the catastrophic stop. The stop was kept anyway — it's designed for a fast, sector-specific collapse (the live book is concentrated in semiconductor names), not a slow-motion systemic crash like the GFC, where it fires near the bottom and locks in losses before the recovery. Its out-of-sample record (2021–2026: 2 triggers, +0.10 Sortino improvement) is the evidence it was accepted on. The 44.8%/21.0%-CAGR figures from an earlier research pass (GLD sleeve only, no sector cap or stop) are superseded by the numbers above. Survivorship bias caveat still applies: the backtest universe excludes stocks that failed outright during the GFC (Lehman, Bear Stearns, Wachovia, WaMu), so the real 2008 drawdown would likely have been worse.

Key structural findings from parameter research (see `scripts/compare_*.py`, `scripts/sleeve_eval.py`, `scripts/walk_forward_validation.py`, `scripts/stress_test_2008.py`):
- A 25% single-position cap and hysteresis (`keep_n=8`, only replacing a held stock when it drops out of the top 8) both improved risk-adjusted return with zero or negative added turnover.
- A static 10% gold (GLD) sleeve improved Sortino/Calmar/MaxDD across every tested window versus no sleeve, and the improvement held out-of-sample — the single biggest structural drawdown lever found.
- A per-sector concentration cap was tested from 30–80% and rejected as a return/Sortino lever (it clips upside as much as it limits downside); an 80% cap was kept anyway as a near-free light-touch buffer.
- A 25% per-name catastrophic stop (intraday-style daily check against average cost) was accepted based on out-of-sample evidence, despite making the historical GFC drawdown worse — see above.
- Inverse-vol/ERC reweighting, GDP-based recession rotation, and dynamic (stress-triggered) sleeve sizing were all tested and rejected or not promoted — in this concentrated momentum book, broadening, reweighting, or timing the signal has consistently traded away more return than the drawdown protection it bought.

All figures above are backtested, not live-trading results, and are sensitive to universe composition, survivorship bias, and the specific historical window tested — see the disclaimer below.

## Tests

```bash
source venv/bin/activate
pytest
```

`pyproject.toml` runs coverage automatically (`--cov=src --cov-report=term-missing`).

## Project layout

```
src/            Live Alpaca bot: broker client, config, strategy, risk sizing, SQLite models
backtest/       Backtrader strategies + helpers (price cache, sector map, hysteresis, weighting)
scripts/        CLI entry points: signal generator, buy-list, backtest runners, research scripts
tests/          pytest suite (unit + integration)
```

## Automation

Both entry points are designed to run unattended via a scheduler (e.g. `launchd` on macOS, `cron`/systemd timers elsewhere):

- `python -m src.main` is a long-running process — start it under a process supervisor that restarts it on crash/reboot.
- `python scripts/signal_generator.py --portfolio-value <amount> --email <you>` is a one-shot script — schedule it monthly (e.g. the 1st of each month) to email a fresh buy/sell list.

## Disclaimer

This is a personal project, not financial advice. Backtested results do not guarantee future performance. The live bot trades against Alpaca's **paper** endpoint by default — verify `ALPACA_BASE_URL` before pointing it at a live account.
