“””
MQFS - Mediterranean Quantitative Finance Society
Sentiment-Statistical Arbitrage Engine
Founder & Lead Researcher: Nicolò Angileri

Theory:
1. COINTEGRAZIONE (Engle-Granger):
Due asset sono cointegrati se esiste una combinazione lineare stazionaria
(spread = Y − β·X). Quando lo spread devia dall’equilibrio storico,
la mean-reversion genera un’opportunità di arbitraggio.

```
2. Z-SCORE DINAMICO ROLLING:
   Z_t = (Spread_t − μ_rolling) / σ_rolling
   Z >  +threshold → spread ipercomprato → SHORT spread (short Y, long X)
   Z <  −threshold → spread ipervenduto  → LONG  spread (long Y, short X)

3. SENTIMENT ADJUSTMENT:
   Il sentiment viene z-scored internamente. La sua magnitude assoluta
   (|Z_sentiment|) abbassa i threshold: un mercato con una view direzionale
   forte giustifica segnali più aggressivi. Sentiment neutro lascia invariato
   il threshold base.
   adjusted_threshold = base_z − clip(|Z_sentiment| × scaler, 0, cap)

4. EXIT SIGNAL:
   La posizione viene chiusa quando |Z| < exit_z_threshold (mean-reversion
   completata). Un segnale opposto alla posizione aperta forza un EXIT sul
   bar corrente; la re-entry avviene sul bar successivo (no same-bar flip).

5. PERFORMANCE METRICS:
   Sharpe ratio calcolato su rendimenti giornalieri mark-to-market (MTM),
   annualizzato con sqrt(252). Posizioni aperte a fine serie vengono chiuse
   MTM sull'ultimo prezzo disponibile e incluse nelle metriche.
```

Author:   Nicolò Angileri (MQFS)
Version:  3.0
Date:     2026-03-15

Changelog vs v2.2:
[UPDATE] Versione allineata all’integrazione con SentimentProvider e run_backtest.
[CLEAN]  Rimosso codice dead (prepare_lookback_series era utility pubblica non usata
internamente → ora è metodo privato _prepare_lookback_series effettivamente
chiamato in calculate_cointegration per la pulizia dello spread).
[CLEAN]  Costante ADF_SIGNIFICANCE_LEVEL esposta come proprietà pubblica di classe.
“””

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ==================== LOGGING ====================

logging.basicConfig(
level=logging.INFO,
format=’[%(asctime)s] [%(levelname)s] %(message)s’,
datefmt=’%Y-%m-%d %H:%M:%S’,
)
logger = logging.getLogger(**name**)
warnings.filterwarnings(‘ignore’, category=FutureWarning)

# ==================== DATACLASSES ====================

@dataclass
class CointegrationResult:
“””
Container per i risultati dell’analisi di cointegrazione.

```
Attributes:
    hedge_ratio        (float):     β OLS — unità di asset_x per neutralizzare 1 di asset_y.
    intercept          (float):     α OLS (costante della regressione).
    p_value            (float):     P-value ADF test sullo spread residuo.
    adf_statistic      (float):     Test statistic ADF.
    r_squared          (float):     R² della regressione OLS.
    spread             (pd.Series): Serie storica: Spread_t = Y_t − β·X_t.
    is_cointegrated    (bool):      True se p_value < 0.05.
    spread_normality_p (float):     P-value Shapiro-Wilk o Jarque-Bera sullo spread.
"""
hedge_ratio: float
intercept: float
p_value: float
adf_statistic: float
r_squared: float
spread: pd.Series
is_cointegrated: bool
spread_normality_p: float = 0.0
```

@dataclass
class SignalResult:
“””
Container per i segnali di trading generati.

```
Attributes:
    signals_df    (pd.DataFrame): DataFrame con Spread, Z-score, Sentiment,
                                  Threshold, Entry, Exit, Signal.
    long_signals  (int):          N. entry LONG  (spread sottovalutato).
    short_signals (int):          N. entry SHORT (spread sopravalutato).
    exit_signals  (int):          N. segnali EXIT (chiusura posizione).
    statistics    (Dict):         Statistiche di sintesi della sessione.
"""
signals_df: pd.DataFrame
long_signals: int
short_signals: int
exit_signals: int
statistics: Dict[str, Any]
```

@dataclass
class PerformanceMetrics:
“””
Metriche di performance del backtest.

```
Il Sharpe è annualizzato su rendimenti giornalieri MTM (rf = 0).
I PnL sono in "unità di spread" (non normalizzati monetariamente).
I costi di transazione non sono inclusi di default.

Attributes:
    sharpe_ratio   (float):       Sharpe ratio annualizzato.
    max_drawdown   (float):       Massimo drawdown in unità di spread.
    win_rate       (float):       Frazione di trade chiusi in profitto [0, 1].
    profit_factor  (float):       Gross profit / Gross loss.
    total_trades   (int):         Trade completati (include MTM finale se aperto).
    avg_trade_pnl  (float):       PnL medio per trade.
    daily_returns  (np.ndarray):  Serie giornaliera MTM (lunghezza = N barre).
    trade_pnl_list (List[float]): PnL per singolo trade.
"""
sharpe_ratio: float
max_drawdown: float
win_rate: float
profit_factor: float
total_trades: int
avg_trade_pnl: float
daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
trade_pnl_list: List[float] = field(default_factory=list)
```

# ==================== UTILITY FUNCTIONS ====================

def normalize_sentiment(sentiment: Optional[float]) -> float:
“””
Normalizza un valore di sentiment singolo, gestendo None e NaN.

```
Args:
    sentiment: Valore atteso in [-1.0, 1.0], oppure None.

Returns:
    float: Valore clippato in [-1.0, 1.0]; 0.0 se None, NaN o non numerico.
"""
if sentiment is None:
    logger.warning("Sentiment è None — defaulting a 0.0.")
    return 0.0
try:
    value = float(sentiment)
    if np.isnan(value):
        logger.warning("Sentiment è NaN — defaulting a 0.0.")
        return 0.0
    return float(np.clip(value, -1.0, 1.0))
except (TypeError, ValueError):
    logger.error("Sentiment non convertibile a float — defaulting a 0.0.")
    return 0.0
```

def validate_hedge_ratio(ratio: float, max_ratio: float = 10.0) -> None:
“””
Valida che l’hedge ratio sia economicamente ragionevole.

```
Args:
    ratio:     Valore dell'hedge ratio da validare.
    max_ratio: Soglia massima accettabile (default = 10.0).

Raises:
    ValueError: Se ratio <= 0 oppure ratio > max_ratio.
"""
if ratio <= 0:
    raise ValueError(
        f"Hedge ratio deve essere positivo, ricevuto {ratio:.6f}. "
        "Verificare l'ordine degli asset nella regressione OLS."
    )
if ratio > max_ratio:
    raise ValueError(
        f"Hedge ratio {ratio:.6f} supera il limite di {max_ratio}. "
        "Verificare la scala degli asset o la selezione della coppia."
    )
```

def adjust_sentiment(sentiment: float, adjustment: float) -> float:
“””
Applica un aggiustamento al sentiment, clippando il risultato a [0.0, 1.0].

```
Args:
    sentiment:   Valore corrente del sentiment.
    adjustment:  Delta da sommare (può essere positivo o negativo).

Returns:
    float: Sentiment aggiustato, garantito in [0.0, 1.0].
"""
try:
    return float(np.clip(sentiment + adjustment, 0.0, 1.0))
except TypeError:
    logger.error("sentiment e adjustment devono essere numerici.")
    return float(np.clip(sentiment, 0.0, 1.0))
```

# ==================== CLASSE PRINCIPALE ====================

class SemanticStatArb:
“””
Motore di Arbitraggio Statistico Sentiment-Adjusted (v3.0 — Production-Ready).

```
Pipeline:
    1. calculate_cointegration()              → CointegrationResult
    2. validate_strategy_feasibility()        → Dict (go/no-go check)
    3. generate_sentiment_adjusted_signals()  → SignalResult
    4. compute_performance_metrics()          → PerformanceMetrics

Esempio di utilizzo base:
```python
engine  = SemanticStatArb(asset_x, asset_y, lookback_window=20)
coint   = engine.calculate_cointegration()

if coint.is_cointegrated:
    signals = engine.generate_sentiment_adjusted_signals(sentiment_vec)
    metrics = engine.compute_performance_metrics(signals)
```
"""

# ---- Costanti di classe (fonte unica di verità) ----
DEFAULT_LOOKBACK: int               = 60
DEFAULT_Z_THRESHOLD: float          = 2.0
DEFAULT_EXIT_Z: float               = 0.5
MIN_OBSERVATION_SIZE: int           = 30
Z_SCORE_EPSILON: float              = 1e-10
ADF_SIGNIFICANCE_LEVEL: float       = 0.05
SENTIMENT_ADJUSTMENT_CAP: float     = 0.5
SENTIMENT_ADJUSTMENT_MULTIPLIER: float = 0.3
MAX_HEDGE_RATIO: float              = 10.0

def __init__(
    self,
    asset_x: pd.Series,
    asset_y: pd.Series,
    lookback_window: int = DEFAULT_LOOKBACK,
    name_x: str = "Asset_X",
    name_y: str = "Asset_Y",
) -> None:
    """
    Inizializza l'engine.

    Args:
        asset_x:         Serie prezzi del primo asset (deve essere strettamente > 0).
        asset_y:         Serie prezzi del secondo asset (stessa lunghezza di asset_x).
        lookback_window: Periodi per il rolling mean/std (default = 60).
        name_x:          Etichetta per i log di asset_x.
        name_y:          Etichetta per i log di asset_y.

    Raises:
        TypeError:  Se i prezzi non sono pd.Series.
        ValueError: Se lunghezze diverse, prezzi non positivi, o campione troppo piccolo.
    """
    if not isinstance(asset_x, pd.Series) or not isinstance(asset_y, pd.Series):
        raise TypeError("asset_x e asset_y devono essere pd.Series.")

    if len(asset_x) != len(asset_y):
        raise ValueError(
            f"Lunghezze non coerenti: asset_x={len(asset_x)}, asset_y={len(asset_y)}."
        )

    if len(asset_x) < self.MIN_OBSERVATION_SIZE:
        raise ValueError(
            f"Richieste almeno {self.MIN_OBSERVATION_SIZE} osservazioni, "
            f"ricevute {len(asset_x)}."
        )

    if (asset_x <= 0).any() or (asset_y <= 0).any():
        raise ValueError("I prezzi devono essere strettamente positivi (> 0).")

    self.asset_x  = asset_x.copy()
    self.asset_y  = asset_y.copy()
    self.lookback = lookback_window
    self.name_x   = name_x
    self.name_y   = name_y

    # Stato interno inizializzato a None / vuoto
    self.hedge_ratio: float                           = 0.0
    self.spread: pd.Series                            = pd.Series(dtype=float)
    self.coint_result: Optional[CointegrationResult] = None

    logger.info(
        f"Engine inizializzato: {name_x} vs {name_y} | "
        f"N={len(asset_x)} | Lookback={lookback_window}"
    )

# ------------------------------------------------------------------
# METODI PRIVATI
# ------------------------------------------------------------------

def _prepare_lookback_series(self, data: pd.Series, lookback: int) -> pd.Series:
    """
    Estrae gli ultimi `lookback` periodi e pulisce i NaN.

    Usa .ffill().bfill() compatibile con Pandas >= 2.0, in sostituzione
    del pattern deprecato fillna(method='ffill').

    Args:
        data:     Serie temporale completa.
        lookback: Numero di periodi da estrarre.

    Returns:
        pd.Series con NaN interpolati via forward/backward fill.
    """
    window = data.iloc[-lookback:].copy()
    if window.isnull().any():
        window = window.ffill().bfill()
        logger.debug("NaN nel lookback: applicato ffill/bfill.")
    return window

# ------------------------------------------------------------------
# STEP 1: COINTEGRAZIONE
# ------------------------------------------------------------------

def calculate_cointegration(self) -> CointegrationResult:
    """
    Calcola la cointegrazione con la procedura Engle-Granger (OLS + ADF).

    Metodologia:
        1. OLS:  Y = α + β·X  →  hedge ratio β, spread S = Y − β·X
        2. ADF:  Test di stazionarietà su S (H0: unit root esistente)
                 Rifiutiamo H0 se p < 0.05 → spread stazionario → cointegrati
        3. Test normalità spread (Shapiro-Wilk per n ≤ 5000, Jarque-Bera altrimenti)
           Serve a segnalare se lo z-score sottostima il rischio nelle code.

    Returns:
        CointegrationResult con tutti i parametri calcolati.

    Raises:
        ValueError: Se lo spread residuo contiene troppi NaN.
    """
    try:
        logger.info(f"Calcolo cointegrazione: {self.name_x} vs {self.name_y}...")

        # Step 1: OLS — Y = α + β·X
        x_const = sm.add_constant(self.asset_x)
        ols     = sm.OLS(self.asset_y, x_const).fit()

        self.hedge_ratio = float(ols.params.iloc[1])
        intercept        = float(ols.params.iloc[0])

        logger.info(
            f"OLS → α={intercept:.6f}, β={self.hedge_ratio:.6f}, R²={ols.rsquared:.4f}"
        )

        # Step 2: Calcolo spread e pulizia NaN
        self.spread  = self.asset_y - (self.hedge_ratio * self.asset_x)
        clean_spread = self._prepare_lookback_series(
            self.spread.dropna(), len(self.spread.dropna())
        )

        if len(clean_spread) < self.MIN_OBSERVATION_SIZE:
            raise ValueError(
                f"Spread con troppi NaN: solo {len(clean_spread)} valori validi "
                f"su {len(self.spread)} totali."
            )

        # Step 3: ADF Test
        adf_stat, p_val, *_ = sm.tsa.stattools.adfuller(clean_spread, autolag='AIC')
        is_coint = bool(p_val < self.ADF_SIGNIFICANCE_LEVEL)
        logger.info(
            f"ADF → stat={adf_stat:.4f}, p={p_val:.4f} → "
            f"{'STAZIONARIO ✓' if is_coint else 'NON STAZIONARIO ✗'}"
        )

        # Step 4: Test di normalità (scipy.stats)
        if len(clean_spread) <= 5000:
            _, sw_p = stats.shapiro(clean_spread)
            norm_label = "Shapiro-Wilk"
        else:
            _, sw_p = stats.jarque_bera(clean_spread)
            norm_label = "Jarque-Bera"
        logger.info(
            f"{norm_label} p={sw_p:.4f} → "
            f"{'NORMALE ✓' if sw_p > 0.05 else 'NON NORMALE ✗'}"
        )

        logger.info(
            f"Spread → μ={float(clean_spread.mean()):.4f}, "
            f"σ={float(clean_spread.std()):.4f}, "
            f"range=[{float(clean_spread.min()):.4f}, {float(clean_spread.max()):.4f}]"
        )

        self.coint_result = CointegrationResult(
            hedge_ratio=self.hedge_ratio,
            intercept=intercept,
            p_value=float(p_val),
            adf_statistic=float(adf_stat),
            r_squared=float(ols.rsquared),
            spread=self.spread.copy(),
            is_cointegrated=is_coint,
            spread_normality_p=float(sw_p),
        )
        return self.coint_result

    except Exception as exc:
        logger.error(f"Errore cointegrazione: {exc}")
        raise

# ------------------------------------------------------------------
# STEP 2: VALIDAZIONE STRATEGIA
# ------------------------------------------------------------------

def validate_strategy_feasibility(self) -> Dict[str, Any]:
    """
    Valida la coppia di asset per l'applicabilità della strategia.

    Controlli eseguiti:
        - Cointegrazione (ADF p < 0.05)
        - Hedge ratio positivo e ≤ MAX_HEDGE_RATIO (stessa soglia di validate_hedge_ratio)
        - Spread sufficientemente volatile (std > 0.01)
        - Distribuzione spread non estrema (avviso se fortemente non-normale)

    Returns:
        Dict con:
            'feasible'  (bool):      Strategia applicabile?
            'issues'    (List[str]): Problemi bloccanti.
            'warnings'  (List[str]): Avvertimenti non critici.
    """
    issues: List[str]        = []
    warnings_list: List[str] = []

    if self.coint_result is None:
        return {
            'feasible': False,
            'issues':   ['calculate_cointegration() non ancora eseguito.'],
            'warnings': [],
        }

    # Cointegrazione
    if not self.coint_result.is_cointegrated:
        issues.append(
            f"Spread NON stazionario "
            f"(ADF p={self.coint_result.p_value:.4f} > {self.ADF_SIGNIFICANCE_LEVEL})."
        )

    # Hedge ratio — usa MAX_HEDGE_RATIO, stessa costante di validate_hedge_ratio()
    if self.hedge_ratio <= 0:
        issues.append(
            f"Hedge ratio non positivo ({self.hedge_ratio:.4f}): "
            "relazione lineare inversa o assente."
        )
    elif self.hedge_ratio > self.MAX_HEDGE_RATIO:
        warnings_list.append(
            f"Hedge ratio elevato ({self.hedge_ratio:.4f} > {self.MAX_HEDGE_RATIO}): "
            "verificare la scala degli asset."
        )

    # Volatilità spread
    spread_std = float(self.spread.std())
    if spread_std < 0.01:
        warnings_list.append(
            f"Spread poco volatile (std={spread_std:.6f}): "
            "attesi pochi segnali di trading."
        )

    # Normalità (avviso informativo)
    if self.coint_result.spread_normality_p < 0.01:
        warnings_list.append(
            f"Spread fortemente non-normale "
            f"(p={self.coint_result.spread_normality_p:.4f}): "
            "lo z-score potrebbe sottostimare il rischio nelle code."
        )

    return {
        'feasible': len(issues) == 0,
        'issues':   issues,
        'warnings': warnings_list,
    }

# ------------------------------------------------------------------
# STEP 3: GENERAZIONE SEGNALI
# ------------------------------------------------------------------

def generate_sentiment_adjusted_signals(
    self,
    sentiment_vector: np.ndarray,
    base_z_threshold: float = DEFAULT_Z_THRESHOLD,
    sentiment_scaler: float = SENTIMENT_ADJUSTMENT_MULTIPLIER,
    exit_z_threshold: float = DEFAULT_EXIT_Z,
) -> SignalResult:
    """
    Genera segnali ENTRY e EXIT con sentiment adjustment dinamico.

    CODIFICA SEGNALI:
        Signal = +1 : ENTRY LONG  (Z < −adjusted_threshold)
        Signal = −1 : ENTRY SHORT (Z >  adjusted_threshold)
        Signal =  2 : EXIT        (|Z| < exit_z_threshold oppure segnale opposto)
        Signal =  0 : HOLD / FLAT

    REGOLE DI GESTIONE POSIZIONE (deterministiche):
        - Nessun re-entry mentre si è già in posizione nella stessa direzione.
        - Un segnale contrario forza EXIT nello stesso bar; la nuova posizione
          viene aperta solo nel bar successivo (no same-bar flip, evita look-ahead).
        - Una sola posizione aperta per volta.

    SENTIMENT ADJUSTMENT:
        Il sentiment_vector viene z-scored internamente. La magnitude assoluta
        |Z_sentiment| abbassa la soglia:
            adjusted_threshold = base_z − clip(|Z_sentiment| × scaler, 0, cap)
        Sentiment neutro → threshold invariato.
        Sentiment direzionale forte → threshold più basso → più segnali.

    Args:
        sentiment_vector:  Array NLP in [−1.0, +1.0], len deve uguagliare len(spread).
        base_z_threshold:  Soglia z-score base per entry (default = 2.0).
        sentiment_scaler:  Impatto del sentiment sul threshold (default = 0.3).
        exit_z_threshold:  Soglia per chiusura posizione (default = 0.5).

    Returns:
        SignalResult con DataFrame completo e statistiche di sessione.

    Raises:
        RuntimeError: Se calculate_cointegration() non è stato eseguito.
        ValueError:   Se sentiment_vector ha lunghezza incompatibile o contiene inf/NaN.
    """
    try:
        if self.coint_result is None:
            raise RuntimeError(
                "Eseguire calculate_cointegration() prima di generare segnali."
            )

        sentiment_vector = np.asarray(sentiment_vector, dtype=float)

        if len(sentiment_vector) != len(self.spread):
            raise ValueError(
                f"Lunghezza sentiment_vector ({len(sentiment_vector)}) != "
                f"lunghezza spread ({len(self.spread)})."
            )

        if not np.all(np.isfinite(sentiment_vector)):
            raise ValueError("sentiment_vector contiene NaN o valori infiniti.")

        if (sentiment_vector < -1.0).any() or (sentiment_vector > 1.0).any():
            logger.warning("sentiment_vector fuori [−1, 1]: applicato clip.")
            sentiment_vector = np.clip(sentiment_vector, -1.0, 1.0)

        logger.info(
            f"Generazione segnali: z_base={base_z_threshold}, "
            f"scaler={sentiment_scaler}, exit_z={exit_z_threshold}"
        )

        # ---- Z-score rolling ----
        rolling_mean = self.spread.rolling(window=self.lookback).mean()
        rolling_std  = self.spread.rolling(window=self.lookback).std()
        z_score      = (self.spread - rolling_mean) / (rolling_std + self.Z_SCORE_EPSILON)
        z_arr        = z_score.values

        # ---- Normalizzazione sentiment ----
        s_mean = float(np.nanmean(sentiment_vector))
        s_std  = float(np.nanstd(sentiment_vector))
        sentiment_norm = (
            (sentiment_vector - s_mean) / s_std
            if s_std > 0
            else np.zeros_like(sentiment_vector)
        )

        # ---- Threshold adjustment ----
        # |Z_sentiment| abbassa il threshold (vedi docstring)
        raw_adj       = np.abs(sentiment_norm) * sentiment_scaler
        adjustment    = np.clip(raw_adj, 0.0, self.SENTIMENT_ADJUSTMENT_CAP)
        adj_threshold = base_z_threshold - adjustment

        # ---- Segnali entry grezzi (senza gestione posizione) ----
        spread_vals = self.spread.values
        raw_long    = np.where(z_arr < -adj_threshold,  1, 0)
        raw_short   = np.where(z_arr >  adj_threshold, -1, 0)
        raw_entry   = np.where(raw_short != 0, raw_short, raw_long)

        # ---- Loop gestione posizione ----
        entry_signal  = np.zeros(len(z_arr), dtype=int)
        exit_signal   = np.zeros(len(z_arr), dtype=int)
        position      = 0   # 0=flat, +1=long spread, −1=short spread
        pending_entry = 0   # entry da eseguire al bar successivo post-flip

        for i in range(len(z_arr)):

            # Esegue pending entry dal bar precedente (post-flip)
            if pending_entry != 0 and position == 0:
                if not np.isnan(z_arr[i]):
                    entry_signal[i] = pending_entry
                    position        = pending_entry
                pending_entry = 0  # svuota sempre, anche su NaN
                continue           # non valutare exit nello stesso bar di entry

            if position == 0:
                # Flat: entra se c'è segnale valido
                sig = int(raw_entry[i])
                if sig != 0 and not np.isnan(z_arr[i]):
                    entry_signal[i] = sig
                    position        = sig

            else:
                # In posizione: controlla exit
                # NaN sul z-score → mantieni posizione, non fare nulla
                if np.isnan(z_arr[i]):
                    continue

                sig = int(raw_entry[i])

                # Condizione 1: mean-reversion completata
                if abs(z_arr[i]) < exit_z_threshold:
                    exit_signal[i] = 2
                    position       = 0

                # Condizione 2: segnale opposto → exit ora, entry al prossimo bar
                elif sig != 0 and sig != position:
                    exit_signal[i] = 2
                    position       = 0
                    pending_entry  = sig

                # Condizione 3: stesso segnale → ignora (no averaging/piramiding)

        # ---- DataFrame risultati ----
        combined = np.where(
            exit_signal == 2, 2,
            np.where(entry_signal != 0, entry_signal, 0)
        )

        df = pd.DataFrame(index=self.spread.index)
        df['Spread']               = spread_vals
        df['Rolling_Mean']         = rolling_mean.values
        df['Rolling_Std']          = rolling_std.values
        df['Z_Score']              = z_arr
        df['Sentiment_Raw']        = sentiment_vector
        df['Sentiment_Normalized'] = sentiment_norm
        df['Threshold_Adjustment'] = adjustment
        df['Adjusted_Threshold']   = adj_threshold
        df['Entry_Signal']         = entry_signal
        df['Exit_Signal']          = exit_signal
        df['Signal']               = combined

        n_long  = int((entry_signal ==  1).sum())
        n_short = int((entry_signal == -1).sum())
        n_exit  = int((exit_signal  ==  2).sum())
        n_hold  = int((combined     ==  0).sum())

        statistics: Dict[str, Any] = {
            'total_bars':         len(df),
            'long_entries':       n_long,
            'short_entries':      n_short,
            'exit_signals':       n_exit,
            'hold_bars':          n_hold,
            'signal_ratio':       round(n_long / n_short, 4) if n_short > 0 else None,
            'mean_z_score':       float(np.nanmean(z_arr)),
            'std_z_score':        float(np.nanstd(z_arr)),
            'mean_sentiment':     float(sentiment_vector.mean()),
            'std_sentiment':      float(sentiment_vector.std()),
            'mean_adj_threshold': float(adj_threshold.mean()),
        }

        logger.info(
            f"Segnali: LONG={n_long}, SHORT={n_short}, "
            f"EXIT={n_exit}, HOLD={n_hold}, Ratio={statistics['signal_ratio']}"
        )

        return SignalResult(
            signals_df=df,
            long_signals=n_long,
            short_signals=n_short,
            exit_signals=n_exit,
            statistics=statistics,
        )

    except Exception as exc:
        logger.error(f"Errore generazione segnali: {exc}")
        raise

# ------------------------------------------------------------------
# STEP 4: PERFORMANCE METRICS
# ------------------------------------------------------------------

def compute_performance_metrics(
    self,
    signal_result: SignalResult,
    annual_factor: float = 252.0,
    transaction_cost_per_trade: float = 0.0,
) -> PerformanceMetrics:
    """
    Calcola metriche di performance su rendimenti giornalieri MTM.

    SHARPE (dimensionalmente corretto):
        1. Calcola daily_pnl[i] = position × (spread[i] − spread[i−1]) per ogni bar
        2. Sharpe = (mean(daily_pnl) / std(daily_pnl)) × sqrt(annual_factor)
        Questo è l'unico modo dimensionalmente corretto con sqrt(252).
        Applicare sqrt(252) su PnL per-trade (come in v2.1) era un errore.

    GESTIONE POSIZIONE APERTA A FINE SERIE:
        Se una posizione è ancora aperta all'ultima barra, viene chiusa MTM
        (valued at last spread price) e inclusa in trade_pnl. In v2.1 veniva
        scartata silenziosamente, distorcendo tutte le metriche.

    Args:
        signal_result:              Output di generate_sentiment_adjusted_signals().
        annual_factor:              Fattore annualizzazione (252 per daily, 52 per weekly).
        transaction_cost_per_trade: Costo fisso per trade in unità di spread (default = 0.0).

    Returns:
        PerformanceMetrics con Sharpe, drawdown, win rate, profit factor.
    """
    df          = signal_result.signals_df.copy()
    spread_vals = df['Spread'].values
    signals     = df['Signal'].values
    n           = len(signals)

    daily_pnl:  np.ndarray    = np.zeros(n, dtype=float)
    trade_pnl:  List[float]   = []
    position:   int           = 0
    running_pnl: float        = 0.0

    for i in range(n):
        sig = int(signals[i])

        # Aggiorna MTM se in posizione (dal bar successivo all'entry)
        if position != 0 and i > 0:
            bar_pnl       = position * (spread_vals[i] - spread_vals[i - 1])
            daily_pnl[i]  = bar_pnl
            running_pnl  += bar_pnl

        # Transizioni di stato
        if sig in (1, -1) and position == 0:
            position    = sig
            running_pnl = 0.0

        elif sig == 2 and position != 0:
            net_pnl = running_pnl - transaction_cost_per_trade
            trade_pnl.append(net_pnl)
            position    = 0
            running_pnl = 0.0

    # Gestione posizione aperta a fine serie (FIX critico vs v2.1)
    if position != 0:
        logger.warning("Posizione aperta a fine serie: chiusa MTM sull'ultimo spread.")
        net_pnl = running_pnl - transaction_cost_per_trade
        trade_pnl.append(net_pnl)

    if not trade_pnl:
        logger.warning("Nessun trade completato: metriche non disponibili.")
        return PerformanceMetrics(
            sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
            profit_factor=0.0, total_trades=0, avg_trade_pnl=0.0,
            daily_returns=daily_pnl, trade_pnl_list=[],
        )

    pnl_arr = np.array(trade_pnl, dtype=float)

    # Sharpe su rendimenti giornalieri MTM (dimensionalmente corretto)
    mean_d = float(np.mean(daily_pnl))
    std_d  = float(np.std(daily_pnl))
    sharpe = (mean_d / (std_d + self.Z_SCORE_EPSILON)) * np.sqrt(annual_factor)

    # Max drawdown su PnL cumulativo giornaliero
    cumulative  = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cumulative)
    max_dd      = float(np.min(cumulative - running_max))

    # Win rate e profit factor
    winners       = pnl_arr[pnl_arr > 0]
    losers        = pnl_arr[pnl_arr < 0]
    win_rate      = float(len(winners) / len(pnl_arr))
    gross_win     = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss    = float(abs(np.sum(losers))) if len(losers) > 0 else self.Z_SCORE_EPSILON
    profit_factor = gross_win / gross_loss

    metrics = PerformanceMetrics(
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 6),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        total_trades=len(pnl_arr),
        avg_trade_pnl=round(float(np.mean(pnl_arr)), 6),
        daily_returns=daily_pnl,
        trade_pnl_list=list(pnl_arr),
    )

    logger.info(
        f"Performance → Sharpe={metrics.sharpe_ratio:.2f}, "
        f"MaxDD={metrics.max_drawdown:.4f}, WinRate={metrics.win_rate:.2%}, "
        f"PF={metrics.profit_factor:.2f}, Trades={metrics.total_trades}"
    )
    return metrics
```
