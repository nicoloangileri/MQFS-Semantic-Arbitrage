"""
MQFS - Mediterranean Quantitative Finance Society
Sentiment-Adjusted Statistical Arbitrage Engine (Production Version)
Founder & Lead Researcher: Nicolò Angileri

Description:
    Un motore di arbitraggio statistico che integra analisi di cointegrazione
    con sentiment NLP real-time. La strategia identifica coppie di asset mean-reverting
    e genera segnali di trading dinamici dove i threshold z-score vengono adattati
    sulla base del sentiment estratto da notizie finanziarie.

Theory:
    1. COINTEGRAZIONE: Due asset sono cointegrati se esiste una combinazione lineare
       stazionaria (spread). Quando lo spread devia dall'equilibrio, rappresenta
       un'opportunità di mean-reversion.
    
    2. Z-SCORE DINAMICO: Misura quante deviazioni standard lo spread è distante
       dalla media mobile. Z > 2 = ipercomprato (vendi), Z < -2 = ipervenduto (compra).
    
    3. SENTIMENT ADJUSTMENT: Il sentiment da NLP modula l'aggressività dei segnali.
       Sentiment positivo riduce i threshold (segnali più aggressivi),
       sentiment negativo li aumenta (segnali più conservatori).

Author: Nicolò Angileri (MQFS)
Date: 2026-03-10
Version: 2.0 (Production-Ready)
"""

import logging
import warnings
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ==================== CONFIGURAZIONE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Sopprimere warning di statsmodels
warnings.filterwarnings('ignore', category=FutureWarning)


# ==================== DATACLASS PER RISULTATI ====================
@dataclass
class CointegrationResult:
    """
    Container per i risultati dell'analisi di cointegrazione.
    
    Attributes:
        hedge_ratio (float): Coefficiente di regressione OLS (Y = hedge_ratio * X + c)
        p_value (float): P-value del test ADF sullo spread (< 0.05 = stazionario)
        spread (pd.Series): Serie storica dello spread calcolato
        is_cointegrated (bool): True se p_value < 0.05 (spread stazionario)
        adf_statistic (float): Test statistic del test ADF
    """
    hedge_ratio: float
    p_value: float
    spread: pd.Series
    is_cointegrated: bool
    adf_statistic: float


@dataclass
class SignalResult:
    """
    Container per i segnali di trading generati.
    
    Attributes:
        signals_df (pd.DataFrame): DataFrame con Z-score, Sentiment, e segnali
        long_signals (int): Numero di segnali long generati
        short_signals (int): Numero di segnali short generati
        statistics (Dict[str, float]): Statistiche di sintesi (mean, std, min, max)
    """
    signals_df: pd.DataFrame
    long_signals: int
    short_signals: int
    statistics: Dict[str, Any]


# ==================== CLASSE PRINCIPALE ====================
class SemanticStatArb:
    """
    Motore di Arbitraggio Statistico Sentiment-Adjusted (Production-Grade).
    
    Questo engine implementa una strategia pairs trading che:
    1. Identifica asset cointegrati (spread mean-reverting)
    2. Monitora deviazioni del spread via z-score rolling
    3. Adatta i threshold di trading in base al sentiment NLP
    4. Genera segnali long/short dinamici
    
    Esempio di utilizzo:
    ```
    engine = SemanticStatArb(asset_x, asset_y, lookback_window=20)
    coint_result = engine.calculate_cointegration()
    
    if coint_result.is_cointegrated:
        signals = engine.generate_sentiment_adjusted_signals(sentiment_vector)
        print(signals.signals_df)
    else:
        logger.warning("Asset non cointegrati - strategia non applicabile")
    ```
    """
    
    # Costanti di default
    DEFAULT_LOOKBACK = 60
    DEFAULT_Z_THRESHOLD = 2.0
    MIN_OBSERVATION_SIZE = 10
    Z_SCORE_EPSILON = 1e-10
    ADF_SIGNIFICANCE_LEVEL = 0.05
    SENTIMENT_ADJUSTMENT_CAP = 0.5  # Max |adjustment| al threshold
    SENTIMENT_ADJUSTMENT_MULTIPLIER = 0.3
    
    def __init__(
        self,
        asset_x: pd.Series,
        asset_y: pd.Series,
        lookback_window: int = DEFAULT_LOOKBACK,
        name_x: str = "Asset_X",
        name_y: str = "Asset_Y"
    ):
        """
        Inizializza l'engine di arbitraggio statistico.
        
        Args:
            asset_x (pd.Series): Prima serie temporale dei prezzi (asset long hedge).
                                 Deve avere index temporale e valori numerici.
            asset_y (pd.Series): Seconda serie temporale dei prezzi (asset short/long).
                                 Deve avere stessa lunghezza di asset_x.
            lookback_window (int, optional): Numero di periodi per il rolling window.
                                             Default = 60 (2-3 mesi di dati giornalieri).
            name_x (str, optional): Etichetta per asset_x (usato nei log). Default = "Asset_X".
            name_y (str, optional): Etichetta per asset_y (usato nei log). Default = "Asset_Y".
        
        Raises:
            ValueError: Se asset_x e asset_y hanno lunghezze diverse.
            TypeError: Se gli input non sono pd.Series.
        """
        # Validazione input
        if not isinstance(asset_x, pd.Series) or not isinstance(asset_y, pd.Series):
            raise TypeError("asset_x e asset_y devono essere pd.Series")
        
        if len(asset_x) != len(asset_y):
            raise ValueError(
                f"Lunghezze non coerenti: asset_x={len(asset_x)}, asset_y={len(asset_y)}"
            )
        
        if len(asset_x) < self.MIN_OBSERVATION_SIZE:
            raise ValueError(
                f"Almeno {self.MIN_OBSERVATION_SIZE} osservazioni richieste, "
                f"ricevute {len(asset_x)}"
            )
        
        if (asset_x <= 0).any() or (asset_y <= 0).any():
            raise ValueError("I prezzi devono essere strettamente positivi")
        
        # Memorizzazione serie
        self.asset_x = asset_x.copy()
        self.asset_y = asset_y.copy()
        self.lookback = lookback_window
        self.name_x = name_x
        self.name_y = name_y
        
        # Stato interno
        self.hedge_ratio: float = 0.0
        self.spread = pd.Series(dtype=float)
        self.coint_result: Optional[CointegrationResult] = None
        
        logger.info(
            f"Engine inizializzato: {name_x} vs {name_y} | "
            f"Osservazioni={len(asset_x)} | Lookback={lookback_window}"
        )
    
    def calculate_cointegration(self) -> CointegrationResult:
        """
        Calcola la cointegrazione tra asset_x e asset_y usando regressione OLS.
        
        METODOLOGIA:
        1. OLS Regression: Y = a + β*X, dove β è l'hedge ratio
        2. Calcolo spread: S_t = Y_t - β*X_t
        3. ADF Test: Verifica stazionarietà dello spread
        4. Interpretazione: p_value < 0.05 → spread stazionario → cointegrati
        
        Returns:
            CointegrationResult: Contenitore con hedge_ratio, p_value, spread, etc.
        
        Raises:
            ValueError: Se OLS non converge o spread contiene troppi NaN.
        
        Note:
            - L'hedging dinamico richiede che lo spread sia stazionario (I(0))
            - Se p_value > 0.05, la strategia NON è applicabile (asset random walk)
            - Il hedge_ratio rappresenta il numero di unità di asset_x per
              neutralizzare 1 unità di asset_y
        """
        try:
            logger.info(f"Inizio calcolo cointegrazione ({self.name_x} vs {self.name_y})...")
            
            # Step 1: Regressione OLS
            # Aggiungi costante per il modello: Y = a + β*X
            x_with_const = sm.add_constant(self.asset_x)
            model = sm.OLS(self.asset_y, x_with_const)
            results = model.fit()
            
            # Estrai hedge ratio (coefficiente di slope)
            self.hedge_ratio = results.params.iloc[1]
            intercept = results.params.iloc[0]
            
            # Step 2: Calcola spread
            # Spread = Y_t - hedge_ratio * X_t (zero-mean portfolio)
            self.spread = self.asset_y - (self.hedge_ratio * self.asset_x)
            
            # Log diagnostici OLS
            logger.info(
                f"OLS Results: α={intercept:.6f}, β={self.hedge_ratio:.6f}, "
                f"R²={results.rsquared:.4f}"
            )
            
            # Step 3: Rimuovi NaN dal spread prima dell'ADF test
            clean_spread = self.spread.dropna()
            
            if len(clean_spread) < self.MIN_OBSERVATION_SIZE:
                raise ValueError(
                    f"Spread contiene troppi NaN: {len(self.spread) - len(clean_spread)} su "
                    f"{len(self.spread)}"
                )
            
            # Step 4: ADF Test (Augmented Dickey-Fuller)
            # H0: Unit root (non stazionario) | H1: Stazionario
            # Rifiutiamo H0 se p_value < 0.05
            adf_result = sm.tsa.stattools.adfuller(clean_spread, autolag='AIC')
            adf_stat = adf_result[0]
            p_value = adf_result[1]
            is_cointegrated = p_value < self.ADF_SIGNIFICANCE_LEVEL
            
            # Log risultati ADF
            stationarity_status = "STAZIONARIO ✓" if is_cointegrated else "NON STAZIONARIO ✗"
            logger.info(
                f"ADF Test: Statistic={adf_stat:.4f}, p-value={p_value:.4f} → {stationarity_status}"
            )
            
            # Step 5: Statistiche spread
            spread_mean = clean_spread.mean()
            spread_std = clean_spread.std()
            logger.info(
                f"Spread Statistics: μ={spread_mean:.4f}, σ={spread_std:.4f}, "
                f"Range=[{clean_spread.min():.4f}, {clean_spread.max():.4f}]"
            )
            
            # Crea result container
            self.coint_result = CointegrationResult(
                hedge_ratio=self.hedge_ratio,
                p_value=p_value,
                spread=self.spread,
                is_cointegrated=is_cointegrated,
                adf_statistic=adf_stat
            )
            
            return self.coint_result
        
        except Exception as e:
            logger.error(f"Errore nel calcolo cointegrazione: {str(e)}")
            raise
    
    def generate_sentiment_adjusted_signals(
        self,
        sentiment_vector: np.ndarray,
        base_z_threshold: float = DEFAULT_Z_THRESHOLD,
        sentiment_scaler: float = SENTIMENT_ADJUSTMENT_MULTIPLIER
    ) -> SignalResult:
        """
        Genera segnali di trading dinamici con sentiment adjustment.
        
        LOGICA:
        1. Calcola Z-score rolling del spread (deviazione standard da media mobile)
        2. Normalizza sentiment vector (-1 a +1)
        3. Adatta threshold z-score in base al sentiment
           - Sentiment POSITIVO → threshold RIDOTTO → segnali più aggressivi LONG
           - Sentiment NEGATIVO → threshold AUMENTATO → segnali più conservatori
        4. Genera segnali binary (1=LONG, -1=SHORT, 0=NEUTRALE)
        
        INTERPRETAZIONE ECONOMICA:
        - Z-score > threshold → Spread ipercomprato → VENDI (short spread = short Y, long X)
        - Z-score < -threshold → Spread ipervenduto → COMPRA (long spread = long Y, short X)
        - Sentiment modula l'intensità del segnale (non il tipo)
        
        Args:
            sentiment_vector (np.ndarray): Array di sentiment NLP (-1.0 a +1.0)
                                          Lunghezza deve uguagliare len(self.spread)
            base_z_threshold (float, optional): Z-score threshold base. Default = 2.0
            sentiment_scaler (float, optional): Moltiplicatore sentiment. Default = 0.3
                                               Controlla l'impatto del sentiment sui threshold.
        
        Returns:
            SignalResult: Contenitore con DataFrame segnali + statistiche.
        
        Raises:
            ValueError: Se sentiment_vector ha lunghezza non coerente.
            RuntimeError: Se calculate_cointegration() non è stato eseguito.
        
        Note:
            - I NaN nei primi lookback periodi sono naturali (rolling window)
            - Il sentiment adjustment è capped a ±0.5 per evitare soglie inverosimili
            - I segnali sono solo raccomandazioni; il position sizing segue logica separata
        """
        try:
            # Validazione prerequisiti
            if self.coint_result is None:
                raise RuntimeError(
                    "Devi eseguire calculate_cointegration() prima di generare segnali"
                )
            
            # Validazione sentiment_vector
            sentiment_vector = np.asarray(sentiment_vector, dtype=float)
            
            if len(sentiment_vector) != len(self.spread):
                raise ValueError(
                    f"Lunghezza sentiment_vector ({len(sentiment_vector)}) != "
                    f"lunghezza spread ({len(self.spread)})"
                )
            
            if not np.all(np.isfinite(sentiment_vector)):
                raise ValueError("sentiment_vector contiene NaN o infiniti")
            
            if (sentiment_vector < -1.0).any() or (sentiment_vector > 1.0).any():
                logger.warning(
                    "sentiment_vector contiene valori fuori [-1, 1]; verrà clippato"
                )
                sentiment_vector = np.clip(sentiment_vector, -1.0, 1.0)
            
            logger.info(
                f"Generazione segnali: base_threshold={base_z_threshold}, "
                f"sentiment_scaler={sentiment_scaler}"
            )
            
            # Step 1: Calcola rolling mean e std dello spread
            rolling_mean = self.spread.rolling(window=self.lookback).mean()
            rolling_std = self.spread.rolling(window=self.lookback).std()
            
            # Step 2: Calcola Z-score con protezione da divisione per zero
            # Z = (X - μ) / σ → misura di deviazione standard da media
            z_score = (self.spread - rolling_mean) / (rolling_std + self.Z_SCORE_EPSILON)
            
            # Step 3: Normalizza sentiment vector
            # Centra e scala il sentiment per meglio distribuire l'impatto
            sentiment_clean = sentiment_vector.copy()
            sentiment_mean = np.nanmean(sentiment_clean)
            sentiment_std = np.nanstd(sentiment_clean)
            
            if sentiment_std > 0:
                sentiment_normalized = (sentiment_clean - sentiment_mean) / sentiment_std
            else:
                sentiment_normalized = np.zeros_like(sentiment_clean)
            
            # Step 4: Calcola adjustment ai threshold
            # Sentiment positivo → riduce threshold (più aggressivo per LONG)
            # Sentiment negativo → aumenta threshold (più conservatore)
            raw_adjustment = sentiment_normalized * sentiment_scaler
            adjustment = np.clip(raw_adjustment, -self.SENTIMENT_ADJUSTMENT_CAP,
                                self.SENTIMENT_ADJUSTMENT_CAP)
            
            # Step 5: Adatta threshold dinamicamente
            # Long threshold: più negativo = facile comprare. Sentiment+ lo rende meno negativo
            # Short threshold: più positivo = facile vendere. Sentiment+ lo rende più positivo
            adjusted_threshold_long = -base_z_threshold + adjustment
            adjusted_threshold_short = base_z_threshold + adjustment
            
            # Step 6: Genera segnali
            # Long: Z < -threshold (spread sottovalutato)
            # Short: Z > +threshold (spread sopravalutato)
            long_signals = np.where(z_score < adjusted_threshold_long, 1, 0)
            short_signals = np.where(z_score > adjusted_threshold_short, -1, 0)
            
            # Combined: Short ha priorità
            combined_signal = np.where(short_signals != 0, short_signals, long_signals)
            
            # Step 7: Crea DataFrame risultati
            signals_df = pd.DataFrame(index=self.spread.index)
            signals_df['Z_Score'] = z_score
            signals_df['Rolling_Mean'] = rolling_mean
            signals_df['Rolling_Std'] = rolling_std
            signals_df['Sentiment'] = sentiment_vector
            signals_df['Sentiment_Normalized'] = sentiment_normalized
            signals_df['Threshold_Adjustment'] = adjustment
            signals_df['Adjusted_Threshold_Long'] = adjusted_threshold_long
            signals_df['Adjusted_Threshold_Short'] = adjusted_threshold_short
            signals_df['Long_Signal'] = long_signals
            signals_df['Short_Signal'] = short_signals
            signals_df['Signal'] = combined_signal
            
            # Step 8: Calcola statistiche
            n_long = int((long_signals == 1).sum())
            n_short = int((short_signals == -1).sum())
            n_neutral = int((combined_signal == 0).sum())
            
            statistics = {
                'total_bars': len(signals_df),
                'long_signals': n_long,
                'short_signals': n_short,
                'neutral_bars': n_neutral,
                'signal_ratio': n_long / max(n_short, 1),  # long/short ratio
                'mean_z_score': z_score.mean(),
                'std_z_score': z_score.std(),
                'mean_sentiment': sentiment_vector.mean(),
                'std_sentiment': sentiment_vector.std(),
            }
            
            logger.info(
                f"Segnali generati: LONG={n_long}, SHORT={n_short}, "
                f"NEUTRAL={n_neutral}, Ratio={statistics['signal_ratio']:.2f}"
            )
            
            return SignalResult(
                signals_df=signals_df,
                long_signals=n_long,
                short_signals=n_short,
                statistics=statistics
            )
        
        except Exception as e:
            logger.error(f"Errore nella generazione segnali: {str(e)}")
            raise
    
    def validate_strategy_feasibility(self) -> Dict[str, Any]:
        """
        Valida se la coppia di asset è adatta per la strategia.
        
        Returns:
            Dict con chiavi:
                - 'feasible' (bool): Strategia applicabile?
                - 'issues' (List[str]): Problemi riscontrati
                - 'warnings' (List[str]): Avvertimenti non critici
        
        Note:
            Controlla:
            - Cointegrazione (p_value < 0.05)
            - Hedge ratio > 0 e < 10 (ragionevole)
            - Spread non costante (std > 0.01)
        """
        issues = []
        warnings_list = []
        
        if self.coint_result is None:
            return {
                'feasible': False,
                'issues': ['calculate_cointegration() non eseguito'],
                'warnings': []
            }
        
        # Check cointegrazione
        if not self.coint_result.is_cointegrated:
            issues.append(
                f"Spread NON stazionario (ADF p-value={self.coint_result.p_value:.4f} > 0.05)"
            )
        
        # Check hedge ratio ragionevole
        if self.hedge_ratio <= 0:
            issues.append(f"Hedge ratio negativo/zero ({self.hedge_ratio:.4f})")
        elif self.hedge_ratio > 10:
            warnings_list.append(f"Hedge ratio molto alto ({self.hedge_ratio:.4f}), possibile leva")
        
        # Check spread volatility
        spread_std = self.spread.std()
        if spread_std < 0.01:
            warnings_list.append(f"Spread poco volatile (std={spread_std:.6f}), pochi trading)
        
        return {
            'feasible': len(issues) == 0,
            'issues': issues,
            'warnings': warnings_list
        }


# ==================== ESECUZIONE EXAMPLE (BACKTEST) ====================
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("MQFS - Sentiment-Adjusted Statistical Arbitrage Engine (v2.0)")
    logger.info("Founder & Lead Researcher: Nicolò Angileri")
    logger.info("=" * 80)
    
    try:
        # ========== SIMULAZIONE DATI ==========
        logger.info("\n[STEP 1] Generazione dati simulati...")
        np.random.seed(42)
        
        # Crea 100 giorni di trading
        dates = pd.date_range(start="2026-01-01", periods=100, freq='D')
        
        # Asset A: Random walk
        asset_a = pd.Series(
            np.cumsum(np.random.normal(0, 1, 100)) + 100,
            index=dates,
            name="Asset_A"
        )
        
        # Asset B: Correlato ad Asset A con relazione lineare + rumore
        # B ≈ 1.5 * A + rumore piccolo → cointegrati
        asset_b = pd.Series(
            asset_a * 1.5 + np.random.normal(0, 0.5, 100),
            index=dates,
            name="Asset_B"
        )
        
        # Sentiment simulato: Oscillazione + trend (news cycle)
        time_factor = np.linspace(0, 4 * np.pi, 100)
        mock_sentiment = np.sin(time_factor) * 0.8 + np.random.normal(0, 0.2, 100)
        mock_sentiment = np.clip(mock_sentiment, -1, 1)
        
        logger.info(f"Asset A: μ={asset_a.mean():.2f}, σ={asset_a.std():.2f}")
        logger.info(f"Asset B: μ={asset_b.mean():.2f}, σ={asset_b.std():.2f}")
        logger.info(f"Sentiment: μ={mock_sentiment.mean():.2f}, σ={mock_sentiment.std():.2f}")
        
        # ========== CALCOLO COINTEGRAZIONE ==========
        logger.info("\n[STEP 2] Analisi cointegrazione...")
        engine = SemanticStatArb(
            asset_x=asset_a,
            asset_y=asset_b,
            lookback_window=20,
            name_x="Asset_A",
            name_y="Asset_B"
        )
        
        coint_result = engine.calculate_cointegration()
        
        # Validazione strategia
        feasibility = engine.validate_strategy_feasibility()
        logger.info(f"Strategia applicabile: {feasibility['feasible']}")
        if feasibility['issues']:
            logger.warning(f"Problemi: {feasibility['issues']}")
        if feasibility['warnings']:
            logger.warning(f"Avvertimenti: {feasibility['warnings']}")
        
        # ========== GENERAZIONE SEGNALI ==========
        if coint_result.is_cointegrated:
            logger.info("\n[STEP 3] Generazione segnali sentiment-adjusted...")
            signal_result = engine.generate_sentiment_adjusted_signals(
                sentiment_vector=mock_sentiment,
                base_z_threshold=2.0,
                sentiment_scaler=0.3
            )
            
            # Visualizza risultati
            logger.info("\n[STEP 4] Risultati finali:")
            logger.info(f"\nStatistiche segnali:\n{pd.Series(signal_result.statistics)}\n")
            
            # Mostra ultimi 10 segnali
            logger.info("Ultimi 10 segnali generati:")
            print(signal_result.signals_df[[
                'Z_Score', 'Sentiment', 'Adjusted_Threshold_Long',
                'Adjusted_Threshold_Short', 'Signal'
            ]].tail(10))
            
        else:
            logger.error("Asset NON cointegrati - strategia NON applicabile!")
            
        logger.info("\n" + "=" * 80)
        logger.info("Esecuzione completata con successo ✓")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Errore fatale: {str(e)}", exc_info=True)
        raise
