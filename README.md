# MQFS — Sentiment-Statistical Arbitrage Engine

**Mediterranean Quantitative Finance Society**  
*Founder & Lead Researcher: Nicolò Angileri*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-3.1-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

A production-grade **pairs trading engine** that fuses classical statistical arbitrage (Engle-Granger cointegration + rolling z-score mean-reversion) with real-time **FinBERT NLP sentiment** scored on financial news headlines fetched from public RSS feeds — no API key required.

Trading thresholds are dynamically adjusted based on the directional *strength* of market sentiment: the engine becomes more aggressive when the market has a clear view and more conservative during noisy, neutral periods.

-----

## Backtest: KO/PEP vs XOM/CVX

The engine is benchmarked on two classic cointegrated pairs from different sectors:

|Pair         |Sector                      |Shared Drivers                                              |
|-------------|----------------------------|------------------------------------------------------------|
|**KO / PEP** |Consumer Staples / Beverages|Input costs, consumer spending, FX, advertising cycles      |
|**XOM / CVX**|Energy                      |Crude oil price, refinery margins, capex cycles, OPEC policy|

Comparing two pairs from different sectors demonstrates that the engine works generically and is not fitted to a single case.

> **To reproduce the backtest:**
> 
> ```bash
> python run_backtest.py
> ```
> 
> This downloads real prices via yfinance, fetches RSS headlines, runs FinBERT, and saves `backtest_results.png`.

-----

## Theory

### 1. Cointegration (Engle-Granger, 1987)

Two assets are cointegrated if a linear combination of their prices is stationary (I(0)). We estimate the hedge ratio β via OLS and verify stationarity with the Augmented Dickey-Fuller test:

```
Spread_t = Y_t − β · X_t

ADF test: H0 = unit root (non-stationary), reject if p-value < 0.05
```

### 2. Dynamic Z-Score

```
Z_t = (Spread_t − μ_rolling) / σ_rolling

Z >  +threshold  →  spread overvalued   →  SHORT spread (short Y, long X)
Z <  −threshold  →  spread undervalued  →  LONG  spread (long Y, short X)
|Z| < exit_z     →  mean-reversion done →  EXIT position
```

### 3. FinBERT Sentiment Adjustment

News headlines are fetched from **Yahoo Finance RSS** (free, no API key) and scored with **ProsusAI/finbert**, a BERT model fine-tuned on financial corpora:

```
article_score = P(positive) − P(negative)    ∈ [−1.0, +1.0]
daily_score   = mean(article_scores on date)
```

The absolute magnitude of the normalised daily sentiment lowers the z-score threshold:

```
adjusted_threshold = base_z − clip(|Z_sentiment| × scaler, 0, cap)
```

**Economic rationale**: strong directional sentiment (whether positive or negative) indicates the market has a clear view, making mean-reversion trades more likely to complete. Neutral sentiment leaves the threshold unchanged at `base_z`.

### 4. Exit Signal

The position is closed when `|Z| < exit_z_threshold` (mean-reversion completed). An opposing entry signal forces an immediate exit; the new position opens on the *next* bar (no same-bar flip — eliminates look-ahead bias).

### 5. Performance Metrics

The Sharpe ratio is computed on **daily mark-to-market returns** (not per-trade PnL), then annualised with √252. This is the only dimensionally correct method. Positions still open at end of series are closed at the last available spread price and included in metrics.

-----

## Architecture

```
MQFS-Semantic-Arbitrage/
├── mqfs_engine.py          Core: cointegration, signal generation, performance
├── sentiment_provider.py   FinBERT + Yahoo Finance RSS + yfinance fallback
├── run_backtest.py         End-to-end two-pair backtest + 6-panel chart
├── requirements.txt        Python dependencies
└── README.md
```

-----

## Installation

```bash
git clone https://github.com/nicoloangileri/MQFS-Semantic-Arbitrage.git
cd MQFS-Semantic-Arbitrage
pip install -r requirements.txt
```

**FinBERT requires PyTorch.** CPU-only install (recommended unless you have a GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Prefer the lightweight VADER fallback?** No GPU or torch required:

```bash
pip install vaderSentiment
python run_backtest.py --no-finbert
```

-----

## Usage

### Run the two-pair backtest (recommended)

```bash
python run_backtest.py              # FinBERT, 2-year window
python run_backtest.py --period 1y  # Shorter window
python run_backtest.py --no-finbert # VADER fallback
```

Produces:

- `backtest_results.png` — 6-panel chart (2 pairs × 3 panels)
- Full metrics printed in terminal

### Programmatic usage

```python
import pandas as pd
from mqfs_engine import SemanticStatArb
from sentiment_provider import SentimentProvider

# 1. Load price series (pd.Series with DatetimeIndex, adjusted close)
asset_x = ...  # e.g. KO
asset_y = ...  # e.g. PEP

# 2. Build sentiment vector from RSS + FinBERT
provider      = SentimentProvider(use_finbert=True)
sentiment_vec = provider.build_daily_sentiment("KO", "PEP", asset_x.index)

# 3. Run the engine
engine  = SemanticStatArb(asset_x, asset_y, lookback_window=20)
coint   = engine.calculate_cointegration()

if coint.is_cointegrated:
    signals = engine.generate_sentiment_adjusted_signals(sentiment_vec)
    metrics = engine.compute_performance_metrics(signals)
    print(f"Sharpe: {metrics.sharpe_ratio:.2f} | WinRate: {metrics.win_rate:.1%}")
```

-----

## Example Output

```
========================================================================
  METRIC                            KO / PEP          XOM / CVX
========================================================================
  Cointegrated                           YES                YES
  ADF p-value                         0.0041             0.0089
  Hedge Ratio (beta)                  1.2345             0.8761
  R^2 OLS                             0.9312             0.9087
  Sharpe Ratio                        1.4200             1.1830
  Max Drawdown                       -0.8134            -1.2045
  Win Rate                            62.50%             58.33%
  Profit Factor                        1.870              1.540
  Total Trades                            16                 18
========================================================================
```

*Note: actual results vary with market conditions and RSS news coverage.*

-----

## News Sources

|Source           |Type    |History                       |API Key      |
|-----------------|--------|------------------------------|-------------|
|Yahoo Finance RSS|Primary |Historical articles           |None required|
|yfinance .news   |Fallback|Recent only (~30-100 articles)|None required|

The fallback activates automatically when the RSS feed is unreachable. Coverage is always reported in logs (e.g. `"Sentiment coverage: 47/504 days with real news (9.3%)"`) so users are never misled about data quality.

-----

## Key Design Decisions

|Decision                                             |Rationale                                                                                             |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------|
|FinBERT over VADER                                   |Fine-tuned on financial text; measurably superior on domain-specific language (Araci, 2019)           |
|RSS over yfinance.news                               |Historical coverage; higher article count; no rate limiting                                           |
|Sentiment as threshold modifier, not signal generator|Avoids spurious signals; sentiment modulates risk appetite, not direction                             |
||Z_sentiment| (magnitude, not sign) as modifier      |Both strong positive and negative sentiment indicate directional certainty → justified lower threshold|
|Exit at |Z| < 0.5                                    |Closes when spread returns to near-equilibrium; avoids holding through full reversal                  |
|No same-bar flip                                     |Prevents look-ahead bias; position reversal requires one full bar                                     |
|Sharpe on daily MTM, not per-trade PnL               |Dimensionally correct annualisation with √252                                                         |
|Open position at end of series closed MTM            |Avoids silent exclusion of incomplete trades                                                          |

-----

## Limitations & Future Work

- **News coverage**: RSS feeds provide recent articles. For deep historical backtests, a paid news API (Bloomberg, Refinitiv, NewsAPI Pro) would give full coverage.
- **No transaction costs**: Set `transaction_cost_per_trade` in `compute_performance_metrics()` to model realistic friction.
- **Static hedge ratio**: OLS β is estimated once on the full sample. A rolling or Kalman-filter-based dynamic hedge ratio would better handle structural breaks.
- **Position sizing**: The engine generates binary signals. Kelly criterion or volatility-targeted sizing should be applied before live deployment.
- **Single-leg PnL**: PnL is in “spread units”. Real deployment requires mapping to dollar P&L via position size and notional value.

-----

## Academic References

1. **Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G.** (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*. Review of Financial Studies, 19(3), 797–827.
1. **Krauss, C.** (2017). *Statistical Arbitrage Pairs Trading Strategies: Review and Outlook*. Journal of Economic Surveys, 31(2), 513–545.
1. **Araci, D.** (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv:1908.10063.
1. **Engle, R. F., & Granger, C. W. J.** (1987). *Co-Integration and Error Correction: Representation, Estimation, and Testing*. Econometrica, 55(2), 251–276.
1. **Vidyamurthy, G.** (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley Finance.

-----


-----

*MQFS — Mediterranean Quantitative Finance Society*
