import streamlit as st

# ====== KELLY CRITERION LOGIC ======
def kelly_fraction(p: float, b: float) -> float:
    """
    Calculates the optimal fraction of capital according to the Kelly Criterion.
    :param p: probability of winning, 0 < p < 1
    :param b: ratio of profit to loss (reward/risk), b > 0
    :return: fraction of capital f* in the range [0, 1]
    """
    if not (0 < p < 1):
        raise ValueError("p must be in the interval (0, 1)")
    if b <= 0:
        raise ValueError("b must be a positive number")
    f = (p * b - (1 - p)) / b
    return max(0.0, min(1.0, f))

# ====== APPLICATION INTERFACE ======
st.set_page_config(page_title="Trading Calculators", layout="wide")

st.title("📊 Trading Calculators")

# Sidebar for calculator selection
calc_type = st.sidebar.selectbox(
    "Select calculator type:",
    ["Kelly Criterion (Position Sizing)"],
)

if calc_type == "Kelly Criterion (Position Sizing)":
    st.header("💰 Kelly Criterion Calculator")

    col1, col2, col3 = st.columns(3)

    with col1:
        W = st.number_input("💵 Total Capital (W)", min_value=0.0, value=10000.0, step=100.0)
    with col2:
        p = st.number_input("🎯 Probability of Win (p)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    with col3:
        b = st.number_input("⚖️ Reward/Risk Ratio (b)", min_value=0.01, value=1.0, step=0.1)

    if st.button("Calculate Optimal Position Size"):
        try:
            f_star = kelly_fraction(p, b)
            position_size = f_star * W

            st.success(f"✅ Optimal Capital Fraction: **{f_star:.2%}**")
            st.info(f"💼 Position Size: **{position_size:,.2f}** out of {W:,.2f}")

            st.progress(min(f_star, 1.0))
        except ValueError as e:
            st.error(f"Input Error: {e}")

    st.markdown("---")
    st.markdown("""
    **Explanation:**
    - Kelly's Formula:  \n
      \\( f^* = \\frac{p \\cdot b - (1 - p)}{b} \\)
    - If the result is ≤ 0 → do not enter the trade.
    - If ≥ 1 → you can use your entire capital (but the risk is high).
    """)

