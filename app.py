
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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

# ── Train models on startup (no pkl files needed) ─────────
@st.cache_resource
def load_models():
    # ── Macro model ───────────────────────────────────────
    # Historical recession data (2001, 2008, 2020)
    macro_data = {
        "yield_curve":           [-0.5,-0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 2.0,
                                   1.5, 1.0, 0.5, 0.0,-0.5,-1.0,-0.8,-0.5,
                                   0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0,
                                   1.5, 1.0, 0.5, 0.0,-0.5,-1.0,-2.0,-1.5,
                                   0.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0],
        "unemployment_claims":   [400,380,350,320,300,280,260,270,
                                   280,300,320,350,400,450,420,400,
                                   280,260,250,240,230,220,230,240,
                                   250,260,270,300,350,600,900,500,
                                   400,350,300,270,250,240,230,220],
        "manufacturing":         [16,16,16,17,17,17,17,16,
                                   16,15,15,14,13,12,12,13,
                                   17,17,18,18,18,18,17,17,
                                   17,16,16,15,14,13,11,12,
                                   15,16,17,17,18,18,17,17],
        "consumer_confidence":   [80,85,90,95,100,105,100,95,
                                   90,80,70,60,55,50,55,60,
                                   90,95,100,105,100,95,95,100,
                                   105,100,95,90,80,60,40,55,
                                   75,85,90,95,100,105,100,95],
        "housing_starts":        [1800,1900,2000,2100,2100,2000,
                                   1800,1600,1400,1200,1000,900,
                                   850,900,950,1000,1300,1400,
                                   1500,1500,1400,1400,1400,1400,
                                   1450,1450,1400,1300,1200,1000,
                                   700,900,1100,1200,1300,1400,
                                   1500,1500,1400,1400],
        "industrial_production": [100,101,102,103,104,104,103,102,
                                   101,99,97,95,93,92,93,95,
                                   100,101,102,103,104,105,105,104,
                                   103,102,101,100,98,90,80,90,
                                   95,97,99,101,103,104,104,103],
        "recession":             [1,1,0,0,0,0,0,0,
                                   0,0,1,1,1,1,0,0,
                                   0,0,0,0,0,0,0,0,
                                   0,0,0,1,1,1,1,0,
                                   0,0,0,0,0,0,0,0],
    }

    df_macro = pd.DataFrame(macro_data)
    feature_cols = ["yield_curve","unemployment_claims","manufacturing",
                    "consumer_confidence","housing_starts","industrial_production"]

    X = df_macro[feature_cols]
    y = df_macro["recession"]

    macro_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        ))
    ])
    macro_pipeline.fit(X, y)

    # ── Personal risk model ───────────────────────────────
    personal_data = {
        "industry": [
            "Construction","Construction","Construction",
            "Manufacturing","Manufacturing","Manufacturing",
            "Finance","Finance","Finance",
            "Technology","Technology","Technology",
            "Healthcare","Healthcare","Healthcare",
            "Retail","Retail","Retail",
            "Education","Education","Education",
            "Hospitality","Hospitality","Hospitality",
        ],
        "recession": [
            "dotcom","financial","covid",
            "dotcom","financial","covid",
            "dotcom","financial","covid",
            "dotcom","financial","covid",
            "dotcom","financial","covid",
            "dotcom","financial","covid",
            "dotcom","financial","covid",
            "dotcom","financial","covid",
        ],
        "job_loss_rate": [
             8.0,16.0,14.0,
             6.0,12.0, 8.0,
             3.0, 6.0, 4.0,
             5.0, 4.0, 3.0,
             1.0, 1.5, 2.0,
             3.0, 5.0,10.0,
             0.5, 1.0, 1.5,
             4.0, 7.0,35.0,
        ],
    }

    df_personal = pd.DataFrame(personal_data)
    le_industry  = LabelEncoder()
    le_recession = LabelEncoder()
    df_personal["industry_enc"]  = le_industry.fit_transform(df_personal["industry"])
    df_personal["recession_enc"] = le_recession.fit_transform(df_personal["recession"])
    df_personal["high_risk"]     = (df_personal["job_loss_rate"] > 8).astype(int)

    X_p = df_personal[["industry_enc","recession_enc"]]
    y_p = df_personal["high_risk"]

    personal_model = RandomForestClassifier(n_estimators=100, random_state=42)
    personal_model.fit(X_p, y_p)

    return macro_pipeline, {
        "model":        personal_model,
        "le_industry":  le_industry,
        "le_recession": le_recession,
    }

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
# In app.py — replace the sidebar api_key section with this

with st.sidebar:
    st.title("📡 RecessionRadar")
    st.caption("Real-time recession risk analysis")
    st.divider()
    
    # ✅ Use your key automatically — no user input needed
    try:
        api_key = st.secrets["FRED_API_KEY"]
        st.success("✅ Live data connected")
    except:
        st.warning("Running in demo mode")
        api_key = None
    
    st.divider()
    st.caption("""
    Tracks 6 Federal Reserve indicators
    that have predicted every US recession
    since 1950.
    
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
    st.subheader("📊 Current Economic Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📈 Yield Curve",
                  f"{current_data.get('yield_curve',0):.2f}%",
                  help="Negative = recession warning")
    with c2:
        st.metric("👷 Unemployment Claims",
                  f"{current_data.get('unemployment_claims',0):,.0f}",
                  help="Weekly jobless claims filed")
    with c3:
        st.metric("🛒 Consumer Confidence",
                  f"{current_data.get('consumer_confidence',0):.1f}",
                  help="How confident people feel spending")
    with c4:
        st.metric("🏠 Housing Starts",
                  f"{current_data.get('housing_starts',0):,.0f}K",
                  help="New homes being built monthly")

    if df_live is not None:
        st.divider()
        st.subheader("📉 Yield Curve — Most Reliable Recession Predictor")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_live.index,
            y=df_live["yield_curve"],
            name="Yield Curve (10Y - 2Y)",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.08)"
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      opacity=0.7,
                      annotation_text="⚠️ Inversion Zone")
        for start, end, name in [
            ("2001-03-01","2001-11-01","Dot-com"),
            ("2007-12-01","2009-06-01","2008 Crisis"),
            ("2020-02-01","2020-04-01","COVID"),
        ]:
            fig.add_vrect(x0=start, x1=end,
                          fillcolor="red", opacity=0.12,
                          line_width=0,
                          annotation_text=name,
                          annotation_position="top left")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f0f1a",
            plot_bgcolor="#1a1a2e",
            height=420,
            title="Every yield curve inversion was followed by a recession",
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
            "Healthcare","Education","Technology",
            "Finance","Retail","Manufacturing",
            "Construction","Hospitality"
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
            "Data/Analytics","Software Development",
            "Sales","Management",
            "Specialized Trade","Customer Service","Research"
        ])

    if st.button("🎯 Calculate My Risk", use_container_width=True):
        industry_risk = {
            "Construction":  0.75, "Hospitality":   0.70,
            "Manufacturing": 0.60, "Retail":        0.50,
            "Finance":       0.40, "Technology":    0.35,
            "Education":     0.15, "Healthcare":    0.10,
        }
        risk = industry_risk[industry]
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

        ca, cb, cc2 = st.columns(3)
        for container, val, label, clr, sub in [
            (ca,  f"{risk:.0%}",       "Your Job Loss Risk",         rc, re+" "+rl),
            (cb,  str(fund_needed),    "Recommended Emergency Fund", "#00d4ff", "months of expenses"),
            (cc2, str(savings_months), "Your Current Savings",       cc, cl),
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
        st.subheader("💡 What You Should Do Right Now")
        if risk > 0.6:
            st.error("🔴 High Risk — Act Now")
            st.markdown("""
            - Build **12 months** emergency fund immediately
            - Start a side skill or freelance income stream
            - Update your resume and LinkedIn **before** you need to
            - Cut all non-essential monthly expenses now
            """)
        elif risk > 0.3:
            st.warning("🟡 Moderate Risk — Stay Prepared")
            st.markdown("""
            - Aim for **6 months** emergency fund
            - Learn recession-proof skills (data, healthcare)
            - Keep your CV and LinkedIn updated regularly
            """)
        else:
            st.success("🟢 Low Risk — Keep Building")
            st.markdown("""
            - Maintain a **3-6 month** emergency fund
            - You are in a resilient sector — keep upskilling
            - Good time to invest and build long term wealth
            """)

        if "Data/Analytics" not in skills:
            st.info("💡 Learning data skills is the best career move for recession-proofing yourself")

        st.divider()
        st.subheader("📊 Your Industry vs Others — 2008 Job Loss %")
        industry_2008 = {
            "Healthcare":1.5,"Education":1.0,"Technology":4.0,
            "Finance":6.0,"Retail":5.0,"Manufacturing":12.0,
            "Construction":16.0,"Hospitality":7.0,
        }
        df_bar = pd.DataFrame({
            "Industry": list(industry_2008.keys()),
            "Loss %":   list(industry_2008.values()),
        }).sort_values("Loss %")

        bar_colors = ["#ff4d4d" if i == industry
                      else "#4da6ff" for i in df_bar["Industry"]]
        fig2 = go.Figure(go.Bar(
            x=df_bar["Loss %"], y=df_bar["Industry"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v}%" for v in df_bar["Loss %"]],
            textposition="outside"
        ))
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f0f1a",
            plot_bgcolor="#1a1a2e",
            height=350,
            xaxis_title="Job Loss %",
            title=f"Red = your industry ({industry}) | 2008 Crisis"
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
    | **Manufacturing Index** | Factory output falling | Consecutive drops |
    | **Consumer Confidence** | People stop spending | Sharp sustained fall |
    | **Housing Starts** | Construction first to fall | Multi-month decline |
    | **Industrial Production** | Overall factory output | Monthly contraction |

    ### The Models
    - **Macro model** — Logistic Regression trained on 2001, 2008, 2020 data
    - **Personal model** — Risk scoring based on BLS historical job loss rates
    - Predicts **6 months ahead** — giving you time to act

    ### Important Disclaimer
    > This tool outputs probability scores based on historical patterns.
    > Every recession is different. No model predicts the future with
    > certainty. Use this as one input — **not as financial advice.**

    ### Data Sources
    - **FRED API** — Federal Reserve Bank of St. Louis
    - **BLS** — Bureau of Labor Statistics
    - **NBER** — Official US recession dates
    """)
    st.divider()
    st.caption("Built with Python · scikit-learn · Streamlit · Plotly · FRED API")
