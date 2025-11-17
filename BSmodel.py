import streamlit as st
import numpy as np
import scipy.stats as si
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Option Take Calculator", layout="wide")

MONTHS = {
    'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
    'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
}

def parse_deribit_instrument(name):
    try:
        parts = name.split('-')
        if len(parts) < 4:
            return {}
        expiry_str = parts[1].upper()
        day = int(''.join([c for c in expiry_str if c.isdigit()]))
        mon = ''.join([c for c in expiry_str if c.isalpha()]).upper()[:3]
        year = int(expiry_str[-2:]) + 2000
        expiry = datetime.date(year, MONTHS.get(mon, 1), day)
        strike = float(parts[2])
        opt_type = 'put' if parts[3][0].upper() == 'P' else 'call'
        return {'strike': strike, 'expiry': expiry, 'option_type': opt_type}
    except Exception:
        return {}

def fetch_deribit_ticker(instrument_name):
    try:
        url = "https://www.deribit.com/api/v2/public/ticker"
        r = requests.get(url, params={"instrument_name": instrument_name}, timeout=5)
        j = r.json()
        return j.get('result', None)
    except:
        return None

# --- Black-Scholes ---
def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(0.0, (S-K) if option_type=='call' else (K-S)), np.nan, np.nan
    if sigma <= 0:
        return max(0.0, (S-K*np.exp(-r*T)) if option_type=='call' else (K*np.exp(-r*T)-S)), np.nan, np.nan

    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type=='call':
        price = S*si.norm.cdf(d1) - K*np.exp(-r*T)*si.norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*si.norm.cdf(-d2) - S*si.norm.cdf(-d1)
    return price, d1, d2

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    price, d1, d2 = bs_price(S, K, T, r, sigma, option_type)
    if np.isnan(d1):
        return 0,0,0,0,0,0
    pdf = si.norm.pdf(d1)
    delta = si.norm.cdf(d1) if option_type=='call' else si.norm.cdf(d1)-1
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T) * 0.01             # per 1% change in IV
    theta = (-S * pdf * sigma / (2*np.sqrt(T))
             - (r*K*np.exp(-r*T)*si.norm.cdf(d2) if option_type=='call'
                else -r*K*np.exp(-r*T)*si.norm.cdf(-d2))) / 365  # per day
    rho = (K*T*np.exp(-r*T)*si.norm.cdf(d2) if option_type=='call'
           else -K*T*np.exp(-r*T)*si.norm.cdf(-d2)) * 0.01       # per 1% change in rate
    return delta, gamma, vega, theta, rho

# --- UI ---
st.title("📈 Option Take Calculator — Black–Scholes")

col1, col2 = st.columns([1, 1.4])

with col1:
    instr = st.text_input("Instrument (example: BTC_USDC-7NOV25-109000-P)", value="", key="instr")
    parsed = parse_deribit_instrument(instr) if instr else {}
    ticker = fetch_deribit_ticker(instr) if instr else None

    if ticker:
        st.success(f"Successfully fetched data from Deribit")
        S0 = ticker['underlying_price']
        sigma = ticker['mark_iv'] if ticker['mark_iv'] < 1 else ticker['mark_iv']/100
        st.write(f"Mark Price: {ticker['mark_price']:.4f}")
        st.write(f"Spot: {S0:.2f} | IV: {sigma:.2%}")
    else:
        S0 = st.number_input("Current Spot Price (S₀)", value=35000.0, key="S0")

    K = st.number_input("Strike (K)", value=parsed.get('strike', 36000.0), key="K")

    if parsed.get('expiry'):
        days_to_exp = max((parsed['expiry'] - datetime.date.today()).days, 1)
    else:
        days_to_exp = 30
    days_to_exp = st.number_input("Days to Expiration", value=days_to_exp, key="days")

    option_type = st.selectbox("Option Type", ["call", "put"],
                               index=0 if parsed.get('option_type','call')=='call' else 1, key="opt_type")

    r = st.number_input("Risk-Free Rate (decimal, e.g., 0.02 = 2%)", value=0.0, key="r")
    sigma = st.number_input("Implied Volatility (decimal, e.g., 0.45 = 45%)",
                            value=0.45 if ticker is None else sigma, key="sigma")
    S_target = st.number_input("Target Spot Price (S_target)", value=S0*1.1, key="S_target")

T = days_to_exp / 365.0

# --- Calculations ---
price_now, _, _ = bs_price(S0, K, T, r, sigma, option_type)
price_target, _, _ = bs_price(S_target, K, T, r, sigma, option_type)
greeks_now = bs_greeks(S0, K, T, r, sigma, option_type)
greeks_target = bs_greeks(S_target, K, T, r, sigma, option_type)

# --- Output ---
with col2:
    st.subheader("📊 Results")
    st.write(f"**Current Option Price:** {price_now:.4f}")
    st.write(f"**Option Price at Spot {S_target:.2f}:** {price_target:.4f}")

    df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega (per 1% IV)", "Theta (per day)", "Rho (per 1%)"],
        "Now": [f"{greeks_now[0]:.6f}", f"{greeks_now[1]:.6e}", f"{greeks_now[2]:.6f}",
                   f"{greeks_now[3]:.6f}", f"{greeks_now[4]:.6f}"],
        "At Target": [f"{greeks_target[0]:.6f}", f"{greeks_target[1]:.6e}", f"{greeks_target[2]:.6f}",
                      f"{greeks_target[3]:.6f}", f"{greeks_target[4]:.6f}"]
    })
    st.table(df)

    st.subheader("Chart: Option Price vs. Spot")
    low, high = S0*0.7, S0*1.3
    spot_range = np.linspace(low, high, 200)
    prices = [bs_price(s, K, T, r, sigma, option_type)[0] for s in spot_range]

    plt.figure(figsize=(7,4))
    plt.plot(spot_range, prices, label="BS price")
    plt.axvline(S0, color='gray', linestyle='--', label='Current spot')
    plt.axvline(S_target, color='orange', linestyle='--', label='Target spot')
    plt.xlabel("Spot")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

st.caption("All greeks are normalized: Vega is the price change per 1% change in IV, Theta is per day, and Rho is per 1% change in the interest rate.")
