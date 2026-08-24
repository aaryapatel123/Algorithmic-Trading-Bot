"""Recession signal from US real GDP quarterly growth (BEA advance estimates).

Entry rule : two consecutive quarters of negative QoQ real GDP growth.
Exit rule  : one quarter of positive QoQ real GDP growth.
Lag        : each quarter's print becomes "known" the month *after* the quarter
             ends (≈ BEA advance estimate release, ~30 days post quarter-end).
             The monthly signal for month M reflects all prints published
             strictly before the 1st of M, so the rebalance on M-1 is the
             earliest that can act on a Q-end print.

Data: non-annualized QoQ % change approximated from BEA advance estimates.
      Values through 2024 are from published advance releases.
      2025+ are placeholders; update when official data is available.
"""
from __future__ import annotations

import datetime

# (year, quarter 1-4) → non-annualized QoQ real GDP growth.
_GDP_QOQ: dict[tuple[int, int], float] = {
    # Pre-GFC
    (2005, 1): 0.0082, (2005, 2): 0.0084, (2005, 3): 0.0090, (2005, 4): 0.0035,
    (2006, 1): 0.0064, (2006, 2): 0.0056, (2006, 3): 0.0048, (2006, 4): 0.0050,
    (2007, 1): 0.0001, (2007, 2): 0.0084, (2007, 3): 0.0074, (2007, 4): 0.0003,
    # GFC recession (NBER: Dec 2007 – Jun 2009)
    (2008, 1): -0.0005, (2008, 2):  0.0047, (2008, 3): -0.0008, (2008, 4): -0.0097,
    (2009, 1): -0.0156, (2009, 2): -0.0017, (2009, 3):  0.0087, (2009, 4):  0.0119,
    # Recovery
    (2010, 1):  0.0078, (2010, 2):  0.0089, (2010, 3):  0.0065, (2010, 4):  0.0035,
    (2011, 1): -0.0001, (2011, 2):  0.0025, (2011, 3):  0.0042, (2011, 4):  0.0072,
    (2012, 1):  0.0052, (2012, 2):  0.0041, (2012, 3):  0.0073, (2012, 4):  0.0001,
    (2013, 1):  0.0028, (2013, 2):  0.0062, (2013, 3):  0.0094, (2013, 4):  0.0086,
    (2014, 1): -0.0049, (2014, 2):  0.0105, (2014, 3):  0.0122, (2014, 4):  0.0052,
    (2015, 1):  0.0006, (2015, 2):  0.0097, (2015, 3):  0.0054, (2015, 4):  0.0035,
    (2016, 1): -0.0010, (2016, 2):  0.0063, (2016, 3):  0.0079, (2016, 4):  0.0049,
    (2017, 1):  0.0032, (2017, 2):  0.0076, (2017, 3):  0.0074, (2017, 4):  0.0065,
    (2018, 1):  0.0050, (2018, 2):  0.0101, (2018, 3):  0.0085, (2018, 4):  0.0029,
    (2019, 1):  0.0077, (2019, 2):  0.0053, (2019, 3):  0.0053, (2019, 4):  0.0056,
    # COVID recession (NBER: Feb 2020 – Apr 2020)
    (2020, 1): -0.0127, (2020, 2): -0.0910, (2020, 3):  0.0746, (2020, 4):  0.0103,
    (2021, 1):  0.0155, (2021, 2):  0.0165, (2021, 3):  0.0082, (2021, 4):  0.0172,
    # Technical 2022 recession (two negative quarters, not NBER-declared)
    (2022, 1): -0.0040, (2022, 2): -0.0022, (2022, 3):  0.0065, (2022, 4):  0.0072,
    (2023, 1):  0.0054, (2023, 2):  0.0065, (2023, 3):  0.0121, (2023, 4):  0.0083,
    (2024, 1):  0.0067, (2024, 2):  0.0072, (2024, 3):  0.0073, (2024, 4):  0.0057,
    # 2025+ placeholders — update as BEA publishes advance estimates
    (2025, 1): -0.0008, (2025, 2):  0.0030, (2025, 3):  0.0040, (2025, 4):  0.0040,
    (2026, 1):  0.0040, (2026, 2):  0.0040,
}


def _pub_month(year: int, quarter: int) -> datetime.date:
    """Advance estimate release: ~1st of the month after quarter-end."""
    end_month = quarter * 3
    if end_month == 12:
        return datetime.date(year + 1, 1, 1)
    return datetime.date(year, end_month + 1, 1)


def _next_month(d: datetime.date) -> datetime.date:
    if d.month == 12:
        return datetime.date(d.year + 1, 1, 1)
    return datetime.date(d.year, d.month + 1, 1)


def build_recession_months(
    start: datetime.date,
    end: datetime.date,
) -> frozenset[tuple[int, int]]:
    """Return a frozenset of (year, month) pairs where recession mode is active.

    The signal for month M uses only GDP prints published strictly before
    the 1st of M — no same-month look-ahead.
    """
    # Build sorted list of (pub_date, growth)
    prints = sorted(
        [(_pub_month(yr, q), g) for (yr, q), g in _GDP_QOQ.items()],
        key=lambda x: x[0],
    )

    recession_months: set[tuple[int, int]] = set()
    in_recession = False
    consecutive_neg = 0
    print_idx = 0

    # Warmup from 1 year before start so initial state is correct
    warmup = datetime.date(start.year - 1, start.month, 1)
    month = warmup
    end_month = datetime.date(end.year, end.month, 1)

    while month <= end_month:
        # Absorb all prints published strictly before this month
        while print_idx < len(prints) and prints[print_idx][0] < month:
            _, growth = prints[print_idx]
            if growth < 0:
                consecutive_neg += 1
                if consecutive_neg >= 2:
                    in_recession = True
            else:
                consecutive_neg = 0
                if in_recession:
                    in_recession = False
            print_idx += 1

        if month >= datetime.date(start.year, start.month, 1) and in_recession:
            recession_months.add((month.year, month.month))

        month = _next_month(month)

    return frozenset(recession_months)


# Consumer Staples, defensive Healthcare, and Utilities from UNIVERSE_2007.
# All 24 tickers are present in both UNIVERSE_2007 (136-stock) and _FALLBACK_UNIVERSE (158-stock).
DEFENSIVE_UNIVERSE: tuple[str, ...] = (
    # Consumer Staples
    "KO", "PEP", "PG", "CL", "MO", "MDLZ", "COST", "WMT",
    # Defensive Healthcare
    "JNJ", "MRK", "PFE", "ABT", "BMY", "AMGN", "GILD", "CVS", "MDT",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL",
)
