
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from fredapi import Fred

st.set_page_config(
    page_title="RecessionRadar",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #e0e0e0; }
    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    macro    = joblib.load("models/saved/macro_model.pkl")
    personal = joblib.load("models/saved/personal_model.pkl")
    return macro, personal

macro_model, personal_bundle = load_models()

@st.cache_data(ttl=86400)
def fetch_live_data(api_key):
    try:
        fred = Fred(api_key=api_key)
        indicators = {
            "yield_curve":           fred.get_series("T10Y2Y"),
            "unemployment_claims":   fred.get_series("ICSA"),
            "manufacturing":         fred.get_series("MANEMP"),
            "consumer_confidence":   fred.get_series("UMCSENT"),
            "housing_starts":        fred.get_series("HOUST"),
            "industrial_production": fred.get_series("INDPRO"),
        }
        df = pd.DataFrame(indicators)
        df = df.resample("MS").mean()
        return df.dropna(), None
    except Exception as e:
        return None, str(e)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 RecessionRadar")
    st.caption("Real-time recession risk analysis")
    st.divider()
    api_key = st.text_input(
        "FRED API Key",
        type="password",
        help="Get free key at fred.stlouisfed.org"
    )
    st.divider()
    st.caption("""
    Tracks 6 Federal Reserve indicators
    that have predicted every US recession
    since 1950. Trained on 2001, 2008,
    and 2020 recession data.

    ⚠️ Not financial advice.
    """)

st.title("📡 RecessionRadar")
st.caption("Live recession probability + personal job loss risk — powered by Federal Reserve data")

tab1, tab2, tab3 = st.tabs([
    "🌍 Macro Dashboard",
    "👤 Personal Risk",
    "📚 How It Works"
])

# ══════════════════════════════════════════
# TAB 1 — MACRO DASHBOARD
# ══════════════════════════════════════════
with tab1:
    if not api_key:
        st.warning("👈 Enter your FRED API key in the sidebar to load live data")
        st.info("📊 Running in DEMO MODE with sample data")
        recession_prob = 0.34
        current_data = {
            "yield_curve":           -0.3,
            "unemployment_claims":   215000,
            "manufacturing":         12500,
            "consumer_confidence":   68.0,
            "housing_starts":        1400,
            "industrial_production": 102.5,
        }
        df_live = None
    else:
        df_live, error = fetch_live_data(api_key)
        if error:
            st.error(f"❌ Could not fetch data: {error}")
            st.stop()
        current_data   = df_live.iloc[-1].to_dict()
        X_now          = pd.DataFrame([current_data])
        recession_prob = macro_model.predict_proba(X_now)[0][1]

    # ── Big Risk Score ─────────────────────────────────────
    if recession_prob < 0.3:
        color, label, emoji = "#00ff88", "LOW RISK",      "🟢"
    elif recession_prob < 0.6:
        color, label, emoji = "#ffd700", "MODERATE RISK", "🟡"
    else:
        color, label, emoji = "#ff4d4d", "HIGH RISK",     "🔴"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align:center; padding:30px;
                    background:#1a1a2e; border-radius:16px;
                    border:2px solid {color}; margin:20px 0">
            <div style="color:#aaa; font-size:16px">
                Recession Probability — Next 6 Months
            </div>
            <div style="color:{color}; font-size:80px; font-weight:bold">
                {recession_prob:.0%}
            </div>
            <div style="color:{color}; font-size:22px; font-weight:bold">
                {emoji} {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Indicator Cards ────────────────────────────────────
    st.subheader("📊 Current Economic Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "📈 Yield Curve",
            f"{current_data.get('yield_curve', 0):.2f}%",
            help="Negative = recession warning"
        )
    with c2:
        st.metric(
            "👷 Unemployment Claims",
            f"{current_data.get('unemployment_claims', 0):,.0f}",
            help="Weekly jobless claims filed"
        )
    with c3:
        st.metric(
            "🛒 Consumer Confidence",
            f"{current_data.get('consumer_confidence', 0):.1f}",
            help="How confident people feel spending"
        )
    with c4:
        st.metric(
            "🏠 Housing Starts",
            f"{current_data.get('housing_starts', 0):,.0f}K",
            help="New homes being built monthly"
        )

    # ── Yield Curve Chart ──────────────────────────────────
    if df_live is not None:
        st.divider()
        st.subheader("📉 Yield Curve — The Most Reliable Recession Predictor")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_live.index,
            y=df_live["yield_curve"],
            name="Yield Curve (10Y - 2Y)",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.08)"
        ))
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            annotation_text="⚠️ Inversion Zone — Recession Warning"
        )
        for start, end, name in [
            ("2001-03-01", "2001-11-01", "Dot-com"),
            ("2007-12-01", "2009-06-01", "2008 Crisis"),
            ("2020-02-01", "2020-04-01", "COVID"),
        ]:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="red", opacity=0.12,
                line_width=0,
                annotation_text=name,
                annotation_position="top left"
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f0f1a",
            plot_bgcolor="#1a1a2e",
            height=420,
            title="Every time yield curve went negative — recession followed within 12-18 months",
            xaxis_title="Year",
            yaxis_title="Yield Spread (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Enter your FRED API key to see the live yield curve chart")

# ══════════════════════════════════════════
# TAB 2 — PERSONAL RISK
# ══════════════════════════════════════════
with tab2:
    st.subheader("👤 How Would a Recession Affect YOU?")
    st.caption("Based on BLS job loss data across 2001, 2008, and 2020 recessions")

    col1, col2 = st.columns(2)
    with col1:
        industry = st.selectbox("Your Industry", [
            "Healthcare", "Education", "Technology",
            "Finance", "Retail", "Manufacturing",
            "Construction", "Hospitality"
        ])
        job_type = st.selectbox("Employment Type", [
            "Permanent Full-time",
            "Contract / Freelance",
            "Part-time",
        ])
        experience = st.slider("Years of Experience", 0, 30, 5)

    with col2:
        company_size = st.selectbox("Company Size", [
            "Large (500+ employees)",
            "Medium (50-500 employees)",
            "Small (< 50 employees)",
        ])
        savings_months = st.slider(
            "Emergency Fund (months of expenses)", 0, 24, 3
        )
        skills = st.multiselect("Your Key Skills", [
            "Data/Analytics", "Software Development",
            "Sales", "Management",
            "Specialized Trade", "Customer Service", "Research"
        ])

    if st.button("🎯 Calculate My Risk", use_container_width=True):

        # Base risk by industry
        industry_risk = {
            "Construction":  0.75,
            "Hospitality":   0.70,
            "Manufacturing": 0.60,
            "Retail":        0.50,
            "Finance":       0.40,
            "Technology":    0.35,
            "Education":     0.15,
            "Healthcare":    0.10,
        }
        risk = industry_risk[industry]

        # Adjust for personal factors
        if job_type == "Contract / Freelance": risk += 0.15
        elif job_type == "Part-time":          risk += 0.10
        if experience < 2:                     risk += 0.10
        elif experience > 10:                  risk -= 0.10
        if "Small" in company_size:            risk += 0.10
        elif "Large" in company_size:          risk -= 0.05
        if "Data/Analytics" in skills or            "Software Development" in skills:   risk -= 0.08

        risk = max(0.05, min(0.95, risk))

        if risk < 0.3:
            rc, rl, re = "#00ff88", "LOW RISK",      "🟢"
        elif risk < 0.6:
            rc, rl, re = "#ffd700", "MODERATE RISK", "🟡"
        else:
            rc, rl, re = "#ff4d4d", "HIGH RISK",     "🔴"

        fund_needed = 12 if risk > 0.5 else 6
        cover_ok    = savings_months >= fund_needed
        cc          = "#00ff88" if cover_ok else "#ff4d4d"
        cl          = "✅ Covered" if cover_ok else "⚠️ Build More"

        # Result cards
        ca, cb, cc2 = st.columns(3)
        for container, val, label, clr, sub in [
            (ca,  f"{risk:.0%}",        "Your Job Loss Risk",        rc,        re + " " + rl),
            (cb,  str(fund_needed),     "Recommended Emergency Fund", "#00d4ff", "months of expenses"),
            (cc2, str(savings_months),  "Your Current Savings",       cc,        cl),
        ]:
            container.markdown(f"""
            <div style="text-align:center; padding:20px;
                        background:#1a1a2e; border-radius:12px;
                        border:2px solid {clr}; margin:10px 0">
                <div style="color:#aaa; font-size:13px">{label}</div>
                <div style="color:{clr}; font-size:52px;
                            font-weight:bold">{val}</div>
                <div style="color:{clr}; font-size:14px">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Advice
        st.subheader("💡 What You Should Do Right Now")
        if risk > 0.6:
            st.error("🔴 High Risk — Act Now")
            st.markdown("""
            - Build **12 months** emergency fund immediately
            - Start a side skill or freelance income stream
            - Update your resume and LinkedIn **before** you need to
            - Cut all non-essential monthly expenses now
            - Consider upskilling into a more resilient field
            """)
        elif risk > 0.3:
            st.warning("🟡 Moderate Risk — Stay Prepared")
            st.markdown("""
            - Aim for **6 months** emergency fund
            - Learn recession-proof skills (data, healthcare)
            - Keep your CV and LinkedIn updated regularly
            - Avoid taking on large new loans right now
            """)
        else:
            st.success("🟢 Low Risk — Keep Building")
            st.markdown("""
            - Maintain a **3-6 month** emergency fund
            - You are in a resilient sector — keep upskilling
            - Good time to invest and build long term wealth
            - Help others in your network prepare
            """)

        if "Data/Analytics" not in skills:
            st.info("💡 Tip: Learning data skills is the single best career move for recession-proofing yourself right now")

        # Industry comparison chart
        st.divider()
        st.subheader("📊 Your Industry vs Others — Job Loss in 2008")

        industry_2008 = {
            "Healthcare":    1.5,
            "Education":     1.0,
            "Technology":    4.0,
            "Finance":       6.0,
            "Retail":        5.0,
            "Manufacturing": 12.0,
            "Construction":  16.0,
            "Hospitality":   7.0,
        }

        df_bar = pd.DataFrame({
            "Industry": list(industry_2008.keys()),
            "Loss %":   list(industry_2008.values()),
        }).sort_values("Loss %")

        bar_colors = [
            "#ff4d4d" if i == industry else "#4da6ff"
            for i in df_bar["Industry"]
        ]

        fig2 = go.Figure(go.Bar(
            x=df_bar["Loss %"],
            y=df_bar["Industry"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v}%" for v in df_bar["Loss %"]],
            textposition="outside"
        ))
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f0f1a",
            plot_bgcolor="#1a1a2e",
            height=350,
            xaxis_title="Job Loss %",
            title=f"Red = your industry ({industry}) | 2008 Financial Crisis"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ══════════════════════════════════════════
with tab3:
    st.subheader("📚 How RecessionRadar Works")

    st.markdown("""
    ### The 6 Indicators We Track

    | Indicator | Why It Matters | Warning Sign |
    |---|---|---|
    | **Yield Curve** | Banks stop lending when inverted | Goes negative |
    | **Unemployment Claims** | Jobs being lost weekly | Spikes above 300K |
    | **Manufacturing Index** | Factory output falling | Consecutive monthly drops |
    | **Consumer Confidence** | People stop spending | Sharp sustained fall |
    | **Housing Starts** | Construction first to fall | Multi-month decline |
    | **Industrial Production** | Overall factory output | Monthly contraction |

    ### The Models
    - **Macro model** — Logistic Regression trained on 2001, 2008, 2020 data
    - **Personal model** — Risk scoring based on BLS historical job loss rates
    - Validated using **walk-forward time series cross validation**
    - Predicts **6 months ahead** — giving you time to act

    ### Why These Indicators Work
    The yield curve alone has predicted every US recession since 1950.
    When short-term interest rates exceed long-term rates, banks stop
    lending profitably — credit freezes, businesses can't grow,
    layoffs follow. Combined with the other 5 indicators, the signal
    becomes even stronger.

    ### Important Disclaimer
    > This tool outputs probability scores based on historical patterns.
    > Every recession is different. No model predicts the future with
    > certainty. Use this as one input to your decisions —
    > **not as financial advice.**

    ### Data Sources
    - **FRED API** — Federal Reserve Bank of St. Louis (live data)
    - **BLS** — Bureau of Labor Statistics (job loss by sector)
    - **NBER** — Official US recession dates
    """)

    st.divider()
    st.caption("Built with Python · scikit-learn · Streamlit · Plotly · FRED API")
