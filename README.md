# MQFS-Semantic-Arbitrage
<div align="center">
**A foundational Quantitative Research architecture by the Mediterranean Quantitative Finance Society.**

</div>

##  Executive Summary
In modern quantitative finance, alpha generation relies heavily on the delta between market latency and contextual comprehension. This repository contains a **Proof of Concept (PoC)** for a statistical arbitrage (pairs trading) engine that dynamically recalibrates its execution thresholds by integrating real-time Natural Language Processing (NLP) sentiment vectors.

Instead of relying on rigid, historical Z-score thresholds, this model fuses traditional econometric rigor with Agentic AI logic, demonstrating how semantic data can preemptively identify market inefficiencies.

##  Core Architecture
The `SemanticStatArb` class is built with institutional-grade logic, focusing on Object-Oriented Programming (OOP) and strict type hinting.

### Key Modules:
1. **Cointegration & Stationarity Engine:** - Utilizes Ordinary Least Squares (OLS) regression to calculate dynamic hedge ratios.
   - Implements Augmented Dickey-Fuller (ADF) testing to validate the stationarity of the spread.
2. **Semantic Signal Generator:**
   - Computes rolling means and standard deviations for Z-score normalization.
   - **The Edge:** Dynamically adjusts the Long/Short entry thresholds based on an injected NLP sentiment vector (ranging from -1.0 to 1.0). Highly positive news lowers the barrier for long entries, anticipating price action before full market digestion.

##  Tech Stack & Libraries
- **Language:** Python 3.9+
- **Data Manipulation:** `pandas`, `numpy`
- **Statistical Modeling:** `statsmodels` (Time Series Analysis)

## 🚀 Usage (Simulation Mode)
The script includes a built-in Monte Carlo simulation module to test the logic without requiring live market data feeds. 


# Navigate to the directory
cd MQFS-Semantic-Arbitrage

# Run the engine simulation
python Semantic_StatArb_Engine.py

The Mediterranean Quantitative Finance Society (MQFS) is a student-driven think-tank based in Southern Italy. Our mission is to bridge the gap between rigorous mathematical modeling, coding, and global financial markets. We focus on social entrepreneurship by unlocking the untapped analytical talent in the Mediterranean basin and building direct bridges to top-tier algorithmic trading hubs.

Local Impact. Global Alpha.
Disclaimer: This code is for academic and research purposes only. It is not financial advice, nor is it production-ready for live capital deployment without extensive API integration, slippage modeling, and risk management protocols.
