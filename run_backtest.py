“””
MQFS - Mediterranean Quantitative Finance Society
Real-Data Backtest: Two-Pair Comparison

Pairs analysed:
Pair 1: Coca-Cola (KO) vs PepsiCo (PEP)
Consumer Staples / Beverages — textbook cointegrated pair

```
Pair 2: ExxonMobil (XOM) vs Chevron (CVX)
        Energy sector — share identical macro drivers (oil price, refinery margins)
```

Why these two pairs?
- Both are among the most studied cointegrated pairs in academic literature
- Different sectors: defensive (KO/PEP) vs cyclical (XOM/CVX)
- Comparing them demonstrates the engine works generically, not just for one case
- Prices freely available via yfinance, adjusted for splits and dividends

Script flow:
1. Download adjusted close prices for all 4 tickers (yfinance, 2 years)
2. Fetch RSS news for each ticker, score with FinBERT (or VADER fallback)
3. Run cointegration + signal generation + performance metrics for each pair
4. Print side-by-side comparison table
5. Save a 6-panel chart (2 pairs x 3 panels each): backtest_results.png

Usage:
python run_backtest.py                 # FinBERT (default)
python run_backtest.py –no-finbert    # VADER fallback (faster, no GPU needed)
python run_backtest.py –period 1y     # Shorter backtest window

Author:   Nicolo Angileri (MQFS)
Version:  3.1
“””

import argparse
import logging
import sys
from typing import Optional, Tuple

import matplotlib
matplotlib.use(‘Agg’)  # Non-interactive backend: works on any environment, no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from mqfs_engine import SemanticStatArb, PerformanceMetrics, SignalResult, CointegrationResult
from sentiment_provider import SentimentProvider

# ==================== LOGGING ====================

logging.basicConfig(
level=logging.INFO,
format=’[%(asctime)s] [%(levelname)s] %(message)s’,
datefmt=’%Y-%m-%d %H:%M:%S’,
)
logger = logging.getLogger(**name**)

# ==================== CONFIG ====================

PAIRS = [
{“x”: “KO”,  “y”: “PEP”, “label”: “KO / PEP (Beverages)”},
{“x”: “XOM”, “y”: “CVX”, “label”: “XOM / CVX (Energy)”},
]

PERIOD      = “2y”
LOOKBACK    = 20
Z_THRESHOLD = 2.0
EXIT_Z      = 0.5
OUTPUT_PNG  = “backtest_results.png”

# ==================== DATA DOWNLOAD ====================

def download_aligned_prices(
ticker_x: str,
ticker_y: str,
period: str,
) -> Tuple[pd.Series, pd.Series]:
“””
Downloads and aligns adjusted close price series via yfinance.

```
Handles both old yfinance column format (flat) and new format
(MultiIndex introduced in yfinance >= 0.2.38).

Args:
    ticker_x: First ticker symbol.
    ticker_y: Second ticker symbol.
    period:   yfinance period string ('1y', '2y', '5y', etc.).

Returns:
    Tuple (asset_x, asset_y): aligned pd.Series, tz-naive, NaN-free.

Raises:
    ImportError: If yfinance is not installed.
    ValueError:  If fewer than 50 common trading days are found.
"""
try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance not installed. Run: pip install yfinance")

def _get_close(ticker: str) -> pd.Series:
    """Downloads Close series; handles MultiIndex and tz-aware index."""
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'][ticker]
    else:
        close = data['Close']

    # Remove timezone for alignment with news dates (tz-naive throughout)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    return close.dropna()

logger.info(f"Downloading prices: {ticker_x}, {ticker_y} | period={period}...")

series_x = _get_close(ticker_x)
series_y = _get_close(ticker_y)

# Align on common trading days (handles different holiday calendars)
common = series_x.index.intersection(series_y.index)

if len(common) < 50:
    raise ValueError(
        f"Only {len(common)} common trading days found for "
        f"{ticker_x}/{ticker_y}. Check tickers and period."
    )

asset_x = series_x.loc[common].copy()
asset_y = series_y.loc[common].copy()

logger.info(
    f"Prices aligned: N={len(common)} trading days | "
    f"[{common[0].date()} -> {common[-1].date()}]"
)
logger.info(f"  {ticker_x}: mean={asset_x.mean():.2f}, std={asset_x.std():.2f}")
logger.info(f"  {ticker_y}: mean={asset_y.mean():.2f}, std={asset_y.std():.2f}")

return asset_x, asset_y
```

# ==================== SINGLE PAIR RUNNER ====================

def run_single_pair(
ticker_x: str,
ticker_y: str,
period: str,
provider: SentimentProvider,
) -> Optional[Tuple[SemanticStatArb, CointegrationResult, SignalResult, PerformanceMetrics]]:
“””
Runs the full pipeline for one pair.

```
Returns:
    Tuple (engine, coint, signals, metrics) on success.
    None if the pair is not cointegrated or data is insufficient.
"""
try:
    asset_x, asset_y = download_aligned_prices(ticker_x, ticker_y, period)
except ValueError as exc:
    logger.error(f"Data error for {ticker_x}/{ticker_y}: {exc}")
    return None

sentiment_vec = provider.build_daily_sentiment(
    ticker_x=ticker_x,
    ticker_y=ticker_y,
    date_index=asset_x.index,
    fill_neutral=0.0,
    forward_fill_days=3,
)

engine = SemanticStatArb(
    asset_x=asset_x,
    asset_y=asset_y,
    lookback_window=LOOKBACK,
    name_x=ticker_x,
    name_y=ticker_y,
)
coint = engine.calculate_cointegration()

feasibility = engine.validate_strategy_feasibility()
for issue in feasibility['issues']:
    logger.warning(f"  [{ticker_x}/{ticker_y}] ISSUE: {issue}")
for warn in feasibility['warnings']:
    logger.warning(f"  [{ticker_x}/{ticker_y}] WARN:  {warn}")

if not coint.is_cointegrated:
    logger.warning(
        f"{ticker_x}/{ticker_y} not cointegrated "
        f"(ADF p={coint.p_value:.4f}). Skipping signals."
    )
    return None

signals = engine.generate_sentiment_adjusted_signals(
    sentiment_vector=sentiment_vec,
    base_z_threshold=Z_THRESHOLD,
    sentiment_scaler=0.3,
    exit_z_threshold=EXIT_Z,
)

metrics = engine.compute_performance_metrics(
    signal_result=signals,
    annual_factor=252.0,
    transaction_cost_per_trade=0.0,
)

return engine, coint, signals, metrics
```

# ==================== VISUALISATION ====================

def plot_two_pairs(
results: list,
pairs: list,
output_path: str,
) -> None:
“””
Saves a 6-panel chart (2 pairs x 3 panels) to disk.

```
Panels per pair:
    Row 1: Spread + Rolling Mean + 1-sigma band
    Row 2: Z-Score with dynamic thresholds and entry/exit markers
    Row 3: FinBERT Sentiment + Cumulative PnL (twin axes)

Args:
    results: List of (engine, coint, signals, metrics) tuples, one per pair.
             Entries may be None if the pair failed cointegration.
    pairs:   List of pair config dicts with 'x', 'y', 'label' keys.
    output_path: Output PNG file path.
"""
n_pairs = len(pairs)
fig, axes = plt.subplots(
    3, n_pairs, figsize=(14 * n_pairs // 2 + 2, 16),
    sharex='col',
)
# axes shape: (3 rows, n_pairs cols)
# If n_pairs == 1, axes would be 1D -> normalise to 2D
if n_pairs == 1:
    axes = np.array(axes).reshape(3, 1)

date_fmt = mdates.DateFormatter('%b %Y')

fig.suptitle(
    'MQFS — Sentiment-Adjusted Statistical Arbitrage\n'
    'Two-Pair Comparison: Beverages (KO/PEP) vs Energy (XOM/CVX)',
    fontsize=13, fontweight='bold', y=0.99,
)

for col, (pair, result) in enumerate(zip(pairs, results)):
    label = pair['label']

    if result is None:
        for row in range(3):
            axes[row][col].text(
                0.5, 0.5, f'{label}\nNot cointegrated',
                ha='center', va='center', transform=axes[row][col].transAxes,
                fontsize=11, color='red',
            )
        continue

    engine, coint, signals, metrics = result
    df = signals.signals_df

    # ---- Row 0: Spread ----
    ax0 = axes[0][col]
    ax0.plot(df.index, df['Spread'],       color='#2196F3', lw=1.0, label='Spread')
    ax0.plot(df.index, df['Rolling_Mean'], color='#FF9800', lw=1.5,
             linestyle='--', label=f'Rolling Mean ({LOOKBACK}d)')
    ax0.fill_between(
        df.index,
        df['Rolling_Mean'] - df['Rolling_Std'],
        df['Rolling_Mean'] + df['Rolling_Std'],
        alpha=0.10, color='#2196F3', label='±1σ',
    )
    ax0.set_title(f'{label}\nSpread & Rolling Statistics', fontsize=9)
    ax0.set_ylabel('Spread (USD)', fontsize=8)
    ax0.legend(fontsize=7, loc='upper left')
    ax0.grid(alpha=0.25)

    # ---- Row 1: Z-Score + signals ----
    ax1 = axes[1][col]
    ax1.plot(df.index, df['Z_Score'], color='#9C27B0', lw=0.8,
             label='Z-Score', zorder=2)
    ax1.fill_between(
        df.index,
        -df['Adjusted_Threshold'], df['Adjusted_Threshold'],
        alpha=0.07, color='grey', label='Neutral band',
    )
    ax1.plot(df.index,  df['Adjusted_Threshold'], color='#F44336',
             lw=0.8, linestyle=':', alpha=0.9, label='Dyn. Threshold')
    ax1.plot(df.index, -df['Adjusted_Threshold'], color='#4CAF50',
             lw=0.8, linestyle=':', alpha=0.9)
    ax1.axhline(0, color='black', lw=0.5, alpha=0.4)

    long_idx  = df[df['Entry_Signal'] ==  1].index
    short_idx = df[df['Entry_Signal'] == -1].index
    exit_idx  = df[df['Exit_Signal']  ==  2].index

    if len(long_idx) > 0:
        ax1.scatter(long_idx,  df.loc[long_idx,  'Z_Score'],
                    marker='^', color='#4CAF50', s=45, zorder=5,
                    label=f'Long ({len(long_idx)})')
    if len(short_idx) > 0:
        ax1.scatter(short_idx, df.loc[short_idx, 'Z_Score'],
                    marker='v', color='#F44336', s=45, zorder=5,
                    label=f'Short ({len(short_idx)})')
    if len(exit_idx) > 0:
        ax1.scatter(exit_idx,  df.loc[exit_idx,  'Z_Score'],
                    marker='x', color='#FF9800', s=35, zorder=5,
                    linewidths=1.2, label=f'Exit ({len(exit_idx)})')

    ax1.set_title('Z-Score & Dynamic Sentiment-Adjusted Thresholds', fontsize=9)
    ax1.set_ylabel('Z-Score', fontsize=8)
    ax1.legend(fontsize=7, ncol=2, loc='upper left')
    ax1.grid(alpha=0.25)

    # ---- Row 2: Sentiment + Cumulative PnL (twin axes) ----
    ax2 = axes[2][col]
    sent = df['Sentiment_Raw'].values
    ax2.fill_between(df.index, 0, sent,
                     where=sent >= 0, color='#4CAF50', alpha=0.40,
                     label='Positive sentiment')
    ax2.fill_between(df.index, 0, sent,
                     where=sent < 0,  color='#F44336', alpha=0.40,
                     label='Negative sentiment')
    ax2.plot(df.index, sent, color='#333333', lw=0.5, alpha=0.5)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_ylabel('Sentiment Score', fontsize=8)

    # Twin axis: Cumulative PnL
    ax2b = ax2.twinx()
    cum_pnl = np.cumsum(metrics.daily_returns)
    ax2b.plot(df.index, cum_pnl, color='#FF5722', lw=1.5,
              label='Cum. PnL', zorder=3)
    ax2b.axhline(0, color='#FF5722', lw=0.4, linestyle='--', alpha=0.5)
    ax2b.set_ylabel('Cum. PnL (spread units)', fontsize=8, color='#FF5722')
    ax2b.tick_params(axis='y', labelcolor='#FF5722', labelsize=7)

    # Metrics annotation
    perf_txt = (
        f"Sharpe={metrics.sharpe_ratio:.2f}  "
        f"MaxDD={metrics.max_drawdown:.3f}\n"
        f"WinRate={metrics.win_rate:.1%}  "
        f"PF={metrics.profit_factor:.2f}  "
        f"Trades={metrics.total_trades}"
    )
    ax2.set_title(f'FinBERT Sentiment + Cumulative PnL\n{perf_txt}', fontsize=8)

    # Combine legends from both axes
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')

    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(output_path, dpi=150, bbox_inches='tight')
logger.info(f"Chart saved: {output_path}")
plt.close(fig)  # Free memory (never use plt.show() with Agg backend)
```

# ==================== COMPARISON TABLE ====================

def print_comparison_table(pairs: list, results: list) -> None:
“”“Prints a formatted side-by-side comparison of both pairs.”””
print(”\n” + “=” * 72)
print(f”  {‘METRIC’:<28} {‘KO / PEP’:>18} {‘XOM / CVX’:>18}”)
print(”=” * 72)

```
metrics_keys = [
    ("Cointegrated",       lambda r: "YES" if r[1].is_cointegrated else "NO"),
    ("ADF p-value",        lambda r: f"{r[1].p_value:.4f}"),
    ("Hedge Ratio (beta)", lambda r: f"{r[1].hedge_ratio:.4f}"),
    ("R^2 OLS",            lambda r: f"{r[1].r_squared:.4f}"),
    ("Sharpe Ratio",       lambda r: f"{r[3].sharpe_ratio:.4f}"),
    ("Max Drawdown",       lambda r: f"{r[3].max_drawdown:.4f}"),
    ("Win Rate",           lambda r: f"{r[3].win_rate:.2%}"),
    ("Profit Factor",      lambda r: f"{r[3].profit_factor:.4f}"),
    ("Total Trades",       lambda r: f"{r[3].total_trades}"),
    ("Avg Trade PnL",      lambda r: f"{r[3].avg_trade_pnl:.6f}"),
    ("Long Entries",       lambda r: f"{r[2].long_signals}"),
    ("Short Entries",      lambda r: f"{r[2].short_signals}"),
    ("Exit Signals",       lambda r: f"{r[2].exit_signals}"),
]

vals = []
for result in results:
    if result is None:
        vals.append(["N/A"] * len(metrics_keys))
    else:
        vals.append([fn(result) for _, fn in metrics_keys])

for i, (name, _) in enumerate(metrics_keys):
    col0 = vals[0][i] if len(vals) > 0 else "N/A"
    col1 = vals[1][i] if len(vals) > 1 else "N/A"
    print(f"  {name:<28} {col0:>18} {col1:>18}")

print("=" * 72)
```

# ==================== MAIN ====================

def main(use_finbert: bool = True, period: str = PERIOD) -> None:
“””
Runs the full two-pair backtest pipeline.

```
Args:
    use_finbert: If True, use FinBERT. If False, use VADER fallback.
    period:      yfinance period string.
"""
logger.info("=" * 72)
logger.info("MQFS -- Sentiment-Adjusted Statistical Arbitrage Engine v3.1")
logger.info(f"Pairs: {[p['label'] for p in PAIRS]} | Period: {period}")
logger.info("=" * 72)

# Initialise sentiment provider once (model loaded once, reused for all pairs)
logger.info(
    f"\n[STEP 1] Loading sentiment model "
    f"({'FinBERT' if use_finbert else 'VADER'})..."
)
provider = SentimentProvider(use_finbert=use_finbert, device=-1)

# Run pipeline for each pair
results = []
for i, pair in enumerate(PAIRS):
    logger.info(
        f"\n[STEP {i + 2}] Running pair: {pair['label']} ..."
    )
    result = run_single_pair(
        ticker_x=pair['x'],
        ticker_y=pair['y'],
        period=period,
        provider=provider,
    )
    results.append(result)

# Comparison table
logger.info("\n[STEP 4] Results comparison:")
print_comparison_table(PAIRS, results)

# 6-panel chart
logger.info(f"\n[STEP 5] Generating chart -> {OUTPUT_PNG} ...")
plot_two_pairs(results, PAIRS, OUTPUT_PNG)

logger.info("\n" + "=" * 72)
logger.info("Backtest completed successfully. Chart saved to: " + OUTPUT_PNG)
logger.info("=" * 72)
```

if **name** == “**main**”:
parser = argparse.ArgumentParser(
description=“MQFS Sentiment-Adjusted Stat Arb – Two-Pair Backtest”
)
parser.add_argument(
‘–no-finbert’, action=‘store_true’,
help=“Use VADER instead of FinBERT (faster, no GPU/torch required)”
)
parser.add_argument(
‘–period’, type=str, default=PERIOD,
help=“yfinance period (‘1y’, ‘2y’, ‘5y’). Default: 2y”
)
args = parser.parse_args()

```
main(use_finbert=not args.no_finbert, period=args.period)
```
