# Trading & Financial Calculators

A collection of Python-based tools for financial analysis, option pricing, and trading strategy evaluation, built with Streamlit.

## ⚠️ Project Status: Under Development

This project is currently a work in progress. The existing modules are standalone scripts.

### Future Plans

The primary goal is to merge all individual calculators into a single, unified Streamlit web application for a more integrated and seamless user experience.

## Included Modules

Currently, the repository contains the following calculators:

1.  **Black-Scholes Option Pricer (`BSmodel.py`)**
    *   Calculates the theoretical price and Greeks (Delta, Gamma, Vega, Theta, Rho) for European options.
    *   Can fetch live data from Deribit for specific instruments.

2.  **Quantitative Volatility Analysis (`Qnt_VolaAnalysis.py`)**
    *   Fits a GJR-GARCH model to historical price data to forecast volatility.
    *   Simulates future price paths and calculates realized volatility (RV).
    *   Fetches data from Yahoo Finance, Binance, or Interactive Brokers.

3.  **Kelly Criterion Calculator (`Trading_Calc.py`)**
    *   Calculates the optimal position size based on the Kelly Criterion formula to maximize long-term capital growth.

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  It's recommended to create a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Calculators

Use the provided batch script to easily launch any module:

```bash
run.bat
```

This will open a menu in your terminal allowing you to select which calculator to run.
arch
ib_insync
```
