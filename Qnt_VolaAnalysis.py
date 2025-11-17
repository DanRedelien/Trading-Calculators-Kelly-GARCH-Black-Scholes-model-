# Qnt_VolaAnalysis.py
# Updated: added correct calculation for realized volatility (RV_realized)
# Requirements:
# pip install streamlit yfinance requests plotly arch scipy ib_insync

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model
from scipy.stats import gaussian_kde

st.set_page_config(layout="wide", page_title="Volatility Range Forecast (GJR-GARCH) + Realized RV")

# ---------------- Helpers / Fetchers ----------------
@st.cache_data(show_spinner=False)
def fetch_yahoo(ticker, period_days=365):
    end_dt = datetime.now().date()
    start_dt = end_dt - timedelta(days=period_days)

    try:
        df = yf.download(
            ticker,
            start=start_dt,
            end=end_dt,
            progress=False,
            threads=False,
            auto_adjust=True,
            group_by="ticker"
        )
    except Exception as e:
        st.warning(f"Yahoo fetch error: {e}")
        return None

    if df is None or df.empty:
        st.warning("⚠️ Empty data from Yahoo Finance.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df[ticker]
        except Exception:
            first_level = df.columns.levels[0][0]
            df = df[first_level]

    df.columns = [str(c).strip().lower() for c in df.columns]

    price_col = None
    for candidate in ['adj close', 'close']:
        if candidate in df.columns:
            price_col = candidate
            break

    if price_col is None:
        st.warning(f"⚠️ Price column (Adj Close or Close) not found in Yahoo data for {ticker}.")
        st.write("Available columns:", list(df.columns))
        return None

    df = df[[price_col]].rename(columns={price_col: 'price'})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=['price'])
    return df

@st.cache_data(show_spinner=False)
def fetch_binance(symbol, period_days=365):
    symbol = symbol.upper().replace("С", "C").replace("А", "A")
    end_ms = int(time.time() * 1000)
    start_dt = datetime.now(timezone.utc).date() - timedelta(days=period_days + 5)
    start_ms = int(datetime.combine(start_dt, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        time.sleep(0.1)
    except Exception as e:
        st.warning(f"Binance fetch error: {e}")
        return None

    if not data or (isinstance(data, dict) and data.get("code")):
        return None

    df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time')[['close']].rename(columns={'close':'price'})
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(show_spinner=False)
def fetch_ib(ticker, period_days=365, host='127.0.0.1', port=7497, clientId=1):
    try:
        from ib_insync import IB, Stock, util
    except Exception as e:
        st.warning("ib_insync is not installed or not available in the environment.")
        return None

    ib = IB()
    try:
        ib.connect(host, port, clientId=clientId)
    except Exception as e:
        st.warning(f"Failed to connect to IB Gateway/TWS: {e}")
        return None

    contract = Stock(symbol=ticker, exchange='SMART', currency='USD')
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f'{period_days} D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
        )
        df = util.df(bars)
    except Exception as e:
        st.warning(f"IB historical fetch error: {e}")
        try:
            ib.disconnect()
        except Exception:
            pass
        return None

    try:
        if df is None or df.empty:
            return None
        df = df.set_index('date').rename(columns={'close': 'price'})[['price']]
        df.index = pd.to_datetime(df.index)
        return df
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

# ---------------- Calculations ----------------
def prepare_returns(df, max_obs=252):
    df = df.copy().sort_index()
    df['ret'] = np.log(df['price']).diff()
    df = df.dropna()
    if len(df) > max_obs:
        df = df.iloc[-max_obs:]
    return df

def fit_gjr_garch(returns):
    y = returns.dropna() * 100.0  # model in percent
    if len(y) < 30:
        raise ValueError("Not enough data for GJR-GARCH fitting.")
    am = arch_model(y, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t')
    res = am.fit(disp='off')
    return am, res

def simulate_paths(res, last_price, last_ret_pct, last_var_pct, last_date, steps=30, n_sims=500):
    params = res.params.to_dict()
    mu = params.get('mu', params.get('const', 0.0))
    omega = params.get('omega', 1e-6)
    alpha = params.get('alpha[1]', params.get('alpha1', 0.0))
    gamma = params.get('gamma[1]', params.get('gamma1', 0.0))
    beta = params.get('beta[1]', params.get('beta1', 0.0))
    nu = params.get('nu', params.get('shape', 8.0))

    last_price = float(np.squeeze(last_price)) if np.size(last_price) == 1 else float(np.asarray(last_price).ravel()[-1])
    h0 = float(last_var_pct)

    paths = np.zeros((steps, n_sims))
    np.random.seed(int(time.time()) % 2**32)

    for sim in range(n_sims):
        h_t = h0
        log_ret_path = []
        for _ in range(steps):
            if nu and nu > 2:
                z = np.random.standard_t(nu)
            else:
                z = np.random.normal()
            r_pct = mu + np.sqrt(max(h_t, 1e-12)) * z
            r_pct = np.tanh(r_pct / 50) * 50
            r = np.log1p(r_pct / 100)
            log_ret_path.append(r)
            indicator = 1 if r_pct < 0 else 0
            h_t = omega + alpha * (r_pct ** 2) + gamma * indicator * (r_pct ** 2) + beta * h_t

        cum_log = np.cumsum(log_ret_path)
        paths[:, sim] = last_price * np.exp(cum_log)

    dates = pd.bdate_range(start=pd.to_datetime(last_date).date() + timedelta(days=1), periods=steps)
    return pd.DataFrame(paths, index=dates)

def compute_bands(paths_df, lower_pct=0.075, upper_pct=0.925):
    median = paths_df.median(axis=1)
    lower = paths_df.quantile(lower_pct, axis=1)
    upper = paths_df.quantile(upper_pct, axis=1)
    return median, lower, upper

def forecast_expected_variance(res, last_var_pct, steps=30):
    params = res.params.to_dict()
    omega = params.get('omega', 1e-6)
    alpha = params.get('alpha[1]', params.get('alpha1', 0.0))
    gamma = params.get('gamma[1]', params.get('gamma1', 0.0))
    beta = params.get('beta[1]', params.get('beta1', 0.0))
    nu = params.get('nu', params.get('shape', 8.0))
    mu = params.get('mu', params.get('const', 0.0))

    Ez2 = nu / (nu - 2.0) if nu and nu > 2 else 1.0

    h = float(last_var_pct)
    expected_vars = []
    for _ in range(steps):
        Er2 = h * Ez2 + (mu ** 2)
        Ei_r2_neg = 0.5 * Er2
        h = omega + alpha * Er2 + gamma * Ei_r2_neg + beta * h
        expected_vars.append(Er2)
    return np.array(expected_vars)

# ---------------- Streamlit UI ----------------
st.title("Volatility analysis — GJR-GARCH model + Realized Volatility (RV)")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker (AAPL, O, BTCUSDT)", "AAPL").upper()
    market = st.selectbox("Data source", ["Auto", "Yahoo", "Binance", "IB"])
with col2:
    days_history = st.number_input("History days (for fitting & realized RV)", 60, 2000, 252)
    forecast_days = st.slider("Forecast days (simulations)", 7, 60, 14)
    n_sims = st.number_input("Simulations", 100, 5000, 800, 100)

with st.expander("IB connection settings (optional)"):
    ib_host = st.text_input("IB Host", "127.0.0.1")
    ib_port = st.number_input("IB Port", 4000, 10000, 7497)
    ib_clientid = st.number_input("IB clientId", 1, 1000, 1)

# New: window for realized vol
rv_window = st.number_input("RV (realized) window in days", 5, 252, 21, 1, help="Window length W for realized volatility calculation (days). RV = sqrt(sum(r^2) * 252/W)")

debug = st.checkbox("Show debug info", value=False)

if st.button("Build Forecast"):
    source = market
    if market == "Auto":
        source = "Binance" if ticker.endswith("USDT") else "Yahoo"

    if source == "IB":
        df = fetch_ib(ticker, days_history, host=ib_host, port=int(ib_port), clientId=int(ib_clientid))
    elif source == "Binance":
        df = fetch_binance(ticker, days_history)
    else:
        df = fetch_yahoo(ticker, days_history)

    if df is None or df.empty:
        st.error("Failed to fetch data from selected source. Check ticker/source/connection.")
        st.stop()

    if 'price' not in df.columns:
        st.error('Price column missing in fetched data')
        st.stop()

    df_ret = prepare_returns(df, max_obs=days_history)
    if df_ret is None or df_ret.empty or len(df_ret) < max(30, rv_window):
        st.error(f'Not enough returns data for analysis. Need at least {max(30, rv_window)} daily returns.')
        st.stop()

    # Fit model
    try:
        am, res = fit_gjr_garch(df_ret['ret'])
    except Exception as e:
        st.error(f"GARCH fitting failed: {e}")
        st.stop()

    # Last values
    last_price_raw = df['price'].iloc[-1]
    try:
        _val = np.squeeze(last_price_raw)
        last_price_val = float(_val)
    except Exception:
        try:
            last_price_val = float(last_price_raw.iloc[-1])
        except Exception:
            last_price_val = None

    last_ret = df_ret['ret'].iloc[-1]  # log return (not percent)
    last_ret_pct = last_ret * 100.0
    last_var_pct = float(res.conditional_volatility.iloc[-1]) ** 2  # model variance in (%^2) because we fitted on *100

    # Model simulations
    paths = simulate_paths(res, last_price_val, last_ret_pct, last_var_pct, df.index[-1], steps=forecast_days, n_sims=n_sims)
    median_85, lower_85, upper_85 = compute_bands(paths, 0.075, 0.925)
    median_95, lower_95, upper_95 = compute_bands(paths, 0.025, 0.975)

    # Model forecasted RV (annualized, percent) — same as before but labelled explicitly
    expected_vars = forecast_expected_variance(res, last_var_pct, forecast_days)  # in (%^2)
    rv_model_annual_pct = np.sqrt(np.mean(expected_vars)) * np.sqrt(252)  # percent
    rv_model_annual = rv_model_annual_pct / 100.0

    # === NEW: Realized Volatility (ex-post) calculation ===
    # We use log-returns (df_ret['ret']) and compute realized volatility on window rv_window:
    # RV_realized = sqrt( sum_{i=1..W} r_i^2 * (252 / W) )  -> expressed in percent
    recent_rets = df_ret['ret'].iloc[-rv_window:]
    sum_sq = np.sum(recent_rets ** 2)
    rv_realized_pct = np.sqrt(sum_sq * (252.0 / rv_window)) * 100.0
    rv_realized = rv_realized_pct / 100.0

    # Rolling realized volatility series (for plotting)
    if len(df_ret) >= rv_window:
        rolling_rv = df_ret['ret'].rolling(window=rv_window).apply(lambda x: np.sqrt(np.sum(x**2) * (252.0 / rv_window)), raw=True) * 100.0
        rolling_rv = rolling_rv.dropna()
    else:
        rolling_rv = pd.Series(dtype=float)

    # KDE for terminal prices
    terminal_prices = paths.iloc[-1, :].values
    terminal_prices = terminal_prices[np.isfinite(terminal_prices)]
    kde_y, kde_x = [], []
    if len(terminal_prices) > 10:
        kde = gaussian_kde(terminal_prices)
        kde_x = np.linspace(np.percentile(terminal_prices, 0.5), np.percentile(terminal_prices, 99.5), 200)
        kde_y = kde(kde_x)

    # Plotting
    base_theme = st.get_option("theme.base") if hasattr(st, "get_option") else 'light'
    is_dark = str(base_theme).lower() == 'dark'
    template = 'plotly_dark' if is_dark else 'plotly_white'
    paper_bg = '#111111' if is_dark else 'white'
    plot_bg = '#111111' if is_dark else 'white'
    axis_color = 'white' if is_dark else 'black'

    color_hist = 'royalblue'
    color_median = 'orange'
    fill_85 = 'rgba(144,238,144,0.25)'
    fill_95 = 'rgba(173,216,230,0.18)'

    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], row_heights=[0.65, 0.35],
                        specs=[[{"type":"xy"}, {"type":"xy"}],
                               [{"type":"xy", "colspan":2}, None]])

    # Price + forecast bands (top-left)
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Historical Price', line=dict(color=color_hist, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=median_85.index, y=median_85, mode='lines', name='Median Forecast', line=dict(dash='dash', color=color_median, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=np.concatenate([lower_85.index, upper_85.index[::-1]]), y=np.concatenate([lower_85, upper_85[::-1]]), fill='toself', fillcolor=fill_85, line=dict(width=0), name='85% band'), 1, 1)
    fig.add_trace(go.Scatter(x=np.concatenate([lower_95.index, upper_95.index[::-1]]), y=np.concatenate([lower_95, upper_95[::-1]]), fill='toself', fillcolor=fill_95, line=dict(width=0), name='95% band'), 1, 1)

    # KDE vertical (top-right)
    if len(kde_x) > 0:
        kde_scaled = kde_y / kde_y.max() * 0.8
        fig.add_trace(
            go.Scatter(
                x=kde_scaled,
                y=kde_x,
                fill='tozerox',
                fillcolor='rgba(255,165,0,0.3)',
                orientation='v',
                name='KDE',
                line=dict(color='orange'),
                showlegend=True
            ),
            1, 2
        )

    # Bottom: realized vol rolling and model implied (single plot)
    # Rolling realized RV
    if rolling_rv.size > 0:
        fig.add_trace(go.Scatter(x=rolling_rv.index, y=rolling_rv.values, mode='lines', name=f'Rolling RV ({rv_window}d)', line=dict(color='green', width=2)), 2, 1)
    # Model implied vol (conditional volatility from GARCH, annualized)
    cond_vol_pct = res.conditional_volatility  # percent (because fitted on *100)
    cond_vol_ann_pct = np.sqrt(np.mean(cond_vol_pct**2)) * np.sqrt(252)  # single scalar not series
    # But better plot recent conditional volatility series annualized (per day -> annualized)
    cond_vol_series_ann = res.conditional_volatility / 100.0 * np.sqrt(252) * 100.0  # in percent: (vol% -> decimal -> annualize -> %)
    cond_vol_series_ann = pd.Series(cond_vol_series_ann, index=df_ret.index[-len(cond_vol_series_ann):]) if len(cond_vol_series_ann) == len(df_ret) else pd.Series(cond_vol_series_ann, index=df_ret.index[-len(cond_vol_series_ann):])

    fig.add_trace(go.Scatter(x=cond_vol_series_ann.index, y=cond_vol_series_ann.values, mode='lines', name='GJR cond. vol (annualized, %)', line=dict(color='red', width=2, dash='dash')), 2, 1)

    # Plot annotations for current RVs
    ann_text = [
        f"Realized RV ({rv_window}d): {rv_realized_pct:.2f}%",
        f"Model forecast RV (annualized): {rv_model_annual_pct:.2f}%"
    ]

    fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg, font=dict(color=axis_color), width=1200, height=760)
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volatility (%)', row=2, col=1)

    st.markdown(f"### **{ticker}** on **{('Binance' if source=='Binance' else ('Interactive Brokers' if source=='IB' else 'Yahoo Finance'))}** — GJR-GARCH & Realized RV")
    st.plotly_chart(fig, use_container_width=True)

    # Output table
    st.write("### RV summary")
    df_summary = pd.DataFrame({
        'metric': ['Realized_RV_window_pct', 'Realized_RV_window_decimal', 'Model_RV_annual_pct', 'Model_RV_annual_decimal'],
        'value': [rv_realized_pct, rv_realized, rv_model_annual_pct, rv_model_annual]
    })
    df_summary['value'] = df_summary['value'].round(6)
    st.dataframe(df_summary.set_index('metric'))

    # Save session
    st.session_state['paths'] = paths
    st.session_state['median_85'] = median_85
    st.session_state['rolling_rv'] = rolling_rv
    st.session_state['rv_realized_pct'] = rv_realized_pct
    st.session_state['rv_model_annual_pct'] = rv_model_annual_pct
    if last_price_val is not None and np.isfinite(last_price_val):
        st.session_state['last_price'] = float(last_price_val)
    else:
        st.session_state['last_price'] = None

# === Interactive probability block ===
if 'paths' in st.session_state and st.session_state['paths'] is not None:
    paths = st.session_state['paths']
    median_85 = st.session_state.get('median_85', None)

    # --- Safely read UI theme from session_state with defaults ---
    template = st.session_state.get('template', 'plotly_white')
    paper_bg = st.session_state.get('paper_bg', 'white')
    plot_bg = st.session_state.get('plot_bg', 'white')
    axis_color = st.session_state.get('axis_color', 'black')

    st.subheader("Probability of Reaching Target Price")

    terminal_prices = paths.iloc[-1, :].values
    terminal_prices = terminal_prices[np.isfinite(terminal_prices)]
    last_price_val = st.session_state.get('last_price', None)

    input_mode = st.radio("Input target as", ["Absolute price", "Percent change from last price"], horizontal=True, index=0, key="target_mode")
    direction = st.radio("Condition", ["≥ target (upside)", "≤ target (downside)"], horizontal=True, index=0, key="target_direction")

    min_tp_abs = float(np.nanmin(terminal_prices)) if terminal_prices.size>0 else 0.0
    max_tp_abs = float(np.nanmax(terminal_prices)) if terminal_prices.size>0 else 0.0
    default_tp_abs = float(median_85.iloc[-1]) if (isinstance(median_85, pd.Series) and np.isfinite(median_85.iloc[-1])) else (
        float(np.nanmedian(terminal_prices)) if terminal_prices.size>0 else min_tp_abs
    )

    if input_mode == "Absolute price":
        target_abs = st.number_input(
            "Enter target price for probability analysis",
            min_value=min_tp_abs,
            max_value=max_tp_abs,
            value=default_tp_abs,
            step=max(0.01, round((max_tp_abs - min_tp_abs)/1000, 2)),
            key="target_price_input"
        )
    else:
        if last_price_val is None or not np.isfinite(last_price_val):
            st.warning("Last price unavailable; switch to Absolute price mode.")
            target_abs = default_tp_abs
        else:
            pct = st.number_input(
                "Enter percent change from last price (%)",
                min_value=-90.0,
                max_value=300.0,
                value=10.0,
                step=0.1,
                key="target_pct_input"
            )
            target_abs = float(last_price_val) * (1.0 + pct/100.0)

    # Only if there are terminal prices and the target price is valid
    if terminal_prices.size > 0 and np.isfinite(target_abs):
        prob = (terminal_prices >= target_abs).mean() if direction.startswith("≥") else (terminal_prices <= target_abs).mean()
        st.metric("Probability", f"{prob*100:.2f}%")

        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=terminal_prices,
            nbinsx=60,
            name="Simulated terminal prices",
            opacity=0.75
        ))
        hist_fig.add_vline(x=target_abs, line_color="red", line_dash="dash")
        hist_fig.update_layout(
            title="Distribution of Terminal Prices",
            xaxis_title="Price at forecast horizon",
            yaxis_title="Frequency",
            template=template,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            font=dict(color=axis_color),
            height=520,
            autosize=True,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        _c1, _c2, _c3 = st.columns([1,2,1])
        with _c2:
            st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.info("No simulations available to calculate probability. Click 'Build Forecast' first.")

if debug:
    st.markdown("---")
    st.write('Debug panel:')
    if 'paths' in st.session_state:
        st.write('Paths shape:', st.session_state['paths'].shape)
    try:
        st.write('Session keys:', list(st.session_state.keys()))
    except Exception:
        pass

# End of file

#Weaknesses

#Lack of modularity: everything is in one large Streamlit script. This is not suitable for a production or research pipeline. It should have been divided into:

#data_fetchers.py

#vola_models.py

#ui_app.py

#utils.py

#The mathematical part is superficial.

#No bootstrap/estimation of volatility confidence intervals.

#No comparison with implied vol or option surface.

#No model testing for overfitting, Ljung-Box, AIC/BIC, etc.

#Poor simulation structure: simulate_paths uses a tanh() limit on returns - this is a hack, not a theoretically justified stabilization.

#No unit tests, logging, CLI parameterization - meaning this is a research tool, not an industrial product.

#The entire RV calculation is implemented manually, without vectorization or checking data frequency - this is ok for daily, but not scalable.