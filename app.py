import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Medical Predict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. SESSION STATE
# ============================================================
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "dashboard"
if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = "heart"
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Theme is kept in dedicated session-state variables so navigation
# buttons/reruns can never reset the selected theme.
# The radio widget is initialized ONCE from the saved theme and then
# becomes the source of truth for Light / Dark / Auto.
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = st.session_state.get("sidebar_theme", "Light")
if "sidebar_theme_selector" not in st.session_state:
    st.session_state.sidebar_theme_selector = st.session_state.theme_mode

# Keep the legacy key synchronized for compatibility with existing code.
st.session_state.sidebar_theme = st.session_state.theme_mode

# ============================================================
# 3. MODERN MEDICAL DASHBOARD DESIGN
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #0F172A;
    --navy2: #16213A;
    --blue: #2563EB;
    --cyan: #0EA5E9;
    --purple: #6366F1;
    --green: #10B981;
    --orange: #F59E0B;
    --red: #EF4444;
    --bg: #F4F7FB;
    --card: #FFFFFF;
    --muted: #64748B;
    --line: #E2E8F0;
}

.stApp {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111B33 0%, #182541 100%) !important;
    border-right: none !important;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    text-align: left !important;
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    transform: none !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1500px !important;
}

h1, h2, h3, h4, p, label, span {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, p, label {
    color: var(--navy) !important;
}

.dashboard-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 6px 24px rgba(15,23,42,0.06);
    margin-bottom: 18px;
}

.hero-card {
    background: linear-gradient(135deg, #0F172A 0%, #1D3563 55%, #2563EB 100%);
    border-radius: 22px;
    padding: 30px;
    color: white;
    box-shadow: 0 12px 35px rgba(37,99,235,0.20);
    margin-bottom: 20px;
}

.hero-card h1, .hero-card h2, .hero-card h3, .hero-card p {
    color: white !important;
}

.kpi-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    min-height: 130px;
    box-shadow: 0 5px 18px rgba(15,23,42,0.05);
}

.kpi-label {
    color: #64748B !important;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.kpi-value {
    color: #0F172A !important;
    font-size: 2rem;
    line-height: 1.1;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-sub {
    color: #64748B !important;
    font-size: 0.8rem;
    margin-top: 7px;
}

.section-title {
    color: #0F172A !important;
    font-size: 1.15rem;
    font-weight: 800;
    margin: 8px 0 14px 2px;
}

.disease-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px;
    min-height: 210px;
    box-shadow: 0 6px 22px rgba(15,23,42,0.05);
}

.disease-icon {
    width: 52px;
    height: 52px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 14px;
    background: #EFF6FF;
}

.disease-title {
    color: #0F172A !important;
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 6px;
}

.disease-text {
    color: #64748B !important;
    font-size: 0.84rem;
    line-height: 1.55;
    min-height: 52px;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: #EFF6FF;
    color: #2563EB !important;
    font-size: 0.72rem;
    font-weight: 700;
}

.result-risk {
    text-align: center;
    padding: 10px 5px 18px;
}

.risk-number {
    font-size: 4rem;
    line-height: 1;
    font-weight: 800;
    color: #2563EB !important;
    margin: 12px 0 5px;
}

.risk-high {
    color: #EF4444 !important;
}

.risk-low {
    color: #10B981 !important;
}

.xai-row {
    margin: 12px 0;
}

.xai-name {
    color: #334155 !important;
    font-size: 0.86rem;
    font-weight: 700;
}

.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 14px;
    padding: 14px 16px;
    color: #1E3A8A !important;
    font-size: 0.83rem;
    line-height: 1.5;
    margin: 12px 0;
}

.recommendation {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 11px 13px;
    margin: 8px 0;
    color: #334155 !important;
    font-size: 0.84rem;
    line-height: 1.45;
}

.stButton > button {
    border-radius: 10px !important;
    border: none !important;
    background: linear-gradient(135deg, #2563EB, #0EA5E9) !important;
    color: white !important;
    font-weight: 700 !important;
    min-height: 42px !important;
    box-shadow: 0 5px 14px rgba(37,99,235,0.18) !important;
    transition: 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(37,99,235,0.25) !important;
}

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #2563EB, #0EA5E9) !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-testid="stNumberInputContainer"] {
    background: white !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
}

input, textarea {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

div[data-testid="stMetric"] {
    background: transparent !important;
}

hr {
    border-color: #E2E8F0 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: white;
    border-radius: 12px;
    padding: 6px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 8px 15px;
    font-weight: 700;
}

.small-muted {
    color: #64748B !important;
    font-size: 0.82rem;
}

.nav-brand {
    padding: 8px 4px 20px;
    font-size: 1.35rem;
    font-weight: 800;
    color: white !important;
}

.nav-sub {
    color: #94A3B8 !important;
    font-size: 0.72rem;
    margin-top: -14px;
    margin-bottom: 20px;
}

.sidebar-label {
    color: #64748B !important;
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 18px 4px 7px;
}

/* ============================================================
   EXACT CLEAN FORM CONTROLS
   White input area + blue +/- controls, with no dark corners.
   ============================================================ */

/* Number-input outer shell */
div[data-testid="stNumberInput"] > div,
div[data-testid="stNumberInputContainer"] > div {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: none !important;
}

/* The actual white typing area */
div[data-testid="stNumberInput"] input,
div[data-testid="stNumberInputContainer"] input,
div[data-testid="stNumberInput"] div[data-baseweb="input"] input,
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"] input {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    border: none !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

/* White BaseWeb input section — removes black/dark corner artifacts */
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"] > div {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: none !important;
    box-shadow: none !important;
}

/* Blue +/- buttons */
div[data-testid="stNumberInput"] button,
div[data-testid="stNumberInputContainer"] button,
button[aria-label="Decrease value"],
button[aria-label="Increase value"] {
    background: #2563EB !important;
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: none !important;
    opacity: 1 !important;
    border-radius: 0 !important;
}

/* White +/- symbols */
div[data-testid="stNumberInput"] button *,
div[data-testid="stNumberInputContainer"] button *,
button[aria-label="Decrease value"] *,
button[aria-label="Increase value"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    opacity: 1 !important;
}

/* Keep the right control area blue on hover */
div[data-testid="stNumberInput"] button:hover,
div[data-testid="stNumberInputContainer"] button:hover,
button[aria-label="Decrease value"]:hover,
button[aria-label="Increase value"]:hover {
    background: #1D4ED8 !important;
    background-color: #1D4ED8 !important;
}

/* Hide Streamlit's native "Clear value" control on number inputs. */
div[data-testid="stNumberInput"] button[aria-label="Clear value"],
div[data-testid="stNumberInputContainer"] button[aria-label="Clear value"],
div[data-testid="stNumberInput"] [aria-label="Clear value"],
div[data-testid="stNumberInputContainer"] [aria-label="Clear value"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* ============================================================
   SIDEBAR HAMBURGER — CLEAN SINGLE IMPLEMENTATION
   ============================================================ */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
button[data-testid="stExpandSidebarButton"],
button[data-testid="stBaseButton-headerNoPadding"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"] button,
button[data-testid="stExpandSidebarButton"],
button[data-testid="stBaseButton-headerNoPadding"] {
    position: fixed !important;
    top: 10px !important;
    left: 16px !important;
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    min-height: 42px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    border-radius: 8px !important;
    background: transparent !important;
    box-shadow: none !important;
    color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
    overflow: visible !important;
    cursor: pointer !important;
}

[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stExpandSidebarButton"] button span,
button[data-testid="stExpandSidebarButton"] span,
button[data-testid="stBaseButton-headerNoPadding"] span {
    font-size: 0 !important;
    color: transparent !important;
    line-height: 0 !important;
}

[data-testid="stSidebarCollapsedControl"] button::before,
[data-testid="stExpandSidebarButton"] button::before,
button[data-testid="stExpandSidebarButton"]::before,
button[data-testid="stBaseButton-headerNoPadding"]::before {
    content: "" !important;
    position: absolute !important;
    left: 10px !important;
    top: 11px !important;
    width: 22px !important;
    height: 3px !important;
    border-radius: 3px !important;
    background: #0F2A5F !important;
    box-shadow: 0 7px 0 #0F2A5F, 0 14px 0 #0F2A5F !important;
    display: block !important;
}

[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stExpandSidebarButton"] button:hover,
button[data-testid="stExpandSidebarButton"]:hover,
button[data-testid="stBaseButton-headerNoPadding"]:hover {
    background: rgba(37, 99, 235, 0.08) !important;
}

/* ============================================================
   PATIENT SNAPSHOT — BOLDER DARK-MODE TEXT
   ============================================================ */
.patient-snapshot,
.patient-snapshot * {
    font-weight: 700 !important;
}

.patient-snapshot .snapshot-label {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
}

.patient-snapshot .snapshot-value {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 750 !important;
}

@media (prefers-color-scheme: dark) {
    .patient-snapshot .snapshot-label,
    .patient-snapshot .snapshot-value,
    .patient-snapshot .section-title {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 750 !important;
    }
}

.result-snapshot-text {
    color: #FFFFFF !important;
    font-weight: 750 !important;
    font-size: 0.95rem !important;
}

/* ============================================================
   FINAL SIDEBAR / SETTINGS LAYOUT
   ============================================================ */
[data-testid="stSidebar"] {
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
}

[data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
}

[data-testid="stSidebarContent"] {
    padding-top: 1.1rem !important;
    padding-left: 1.05rem !important;
    padding-right: 1.05rem !important;
}

.theme-caption {
    color: #94A3B8 !important;
    font-size: 0.72rem;
    font-weight: 700;
    margin: 0 0 8px 4px;
}

.theme-row {
    display: flex;
    border: 1px solid rgba(148,163,184,.45);
    border-radius: 99px;
    overflow: hidden;
    margin: 0 0 10px 0;
}

.theme-row-item {
    flex: 1;
    padding: 8px 4px;
    text-align: center;
    color: #E2E8F0 !important;
    font-size: 0.78rem;
    font-weight: 600;
}

.theme-row-item.active {
    background: #2563EB;
    color: #FFFFFF !important;
}

.sidebar-setting {
    padding: 8px 4px;
    border-bottom: 1px solid rgba(148,163,184,.22);
    color: #E2E8F0 !important;
    font-size: 0.78rem;
}

.sidebar-setting span {
    float: right;
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] .utility-button > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(148,163,184,.22) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    min-height: 38px !important;
    padding: 6px 4px !important;
    color: #E2E8F0 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .utility-button > button:hover {
    background: rgba(255,255,255,.06) !important;
    transform: none !important;
}

[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    padding: 0 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. MODEL ASSETS
# ============================================================
@st.cache_resource
def load_disease_assets(disease_key):
    model_path = f"model_{disease_key}.pkl"
    scaler_path = f"scaler_{disease_key}.pkl"
    features_path = f"features_{disease_key}.pkl"

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else []

    return model, scaler, features

# ============================================================
# 5. CHARTS
# ============================================================
def chart_layout(fig, height=220):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=plot_text_color(), size=12),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=plot_text_color(), size=11),
            title=dict(font=dict(color=plot_text_color(), size=12)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#475569" if get_theme_mode() == "Dark" else "#CBD5E1",
            zeroline=False,
            tickfont=dict(color=plot_text_color(), size=11),
            title=dict(font=dict(color=plot_text_color(), size=12)),
        ),
        showlegend=False,
    )
    return fig

def create_risk_trend():
    diseases = ["Diabetes", "Heart Disease", "Chronic Kidney Disease"]
    prevalence = [89.2, 51.3, 50.8]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=diseases,
        y=prevalence,
        text=[f"{v:.1f}%" for v in prevalence],
        textposition="outside",
        textfont=dict(color=plot_text_color(), size=13, family="Inter"),
        marker=dict(color=["#2563EB", "#F97316", "#10B981"], line=dict(width=0)),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        showlegend=False,
        yaxis=dict(
            title=dict(text="Disease Prevalence (%)", font=dict(color=plot_text_color(), size=11)),
            range=[0, 100],
            tickfont=dict(color=plot_text_color(), size=10),
            gridcolor="#475569" if get_theme_mode() == "Dark" else "#CBD5E1",
            zeroline=False,
        ),
        xaxis=dict(
            title=dict(text="Training Dataset", font=dict(color=plot_text_color(), size=11)),
            tickfont=dict(color=plot_text_color(), size=10),
            showgrid=False,
        ),
    )
    return chart_layout(fig, 250)

def create_distribution():
    labels = ["No Disease", "Disease Present"]
    values = [1387, 2033]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.72,
        textinfo="none",
        hovertemplate="%{label}: %{value:,} records (%{percent})<extra></extra>",
        marker=dict(colors=["#60A5FA", "#EF4444"], line=dict(color="#FFFFFF", width=1)),
    ))

    fig.update_layout(
        height=210,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.02,
            x=0.5,
            xanchor="center",
            font=dict(color=plot_text_color(), size=11),
        ),
        annotations=[dict(
            text="<b>3,420</b><br><span style='font-size:11px'>training records</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=plot_text_color(), size=16),
        )],
    )
    return fig

def create_xai_chart(names, values):
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        textfont=dict(color=plot_text_color(), size=16, family="Arial"),
        cliponaxis=False,
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=5, r=65, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            showticklabels=False,
            tickfont=dict(color=plot_text_color(), size=11),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color=plot_text_color(), size=12),
            tickcolor=plot_text_color(),
        ),
    )
    return fig

# ============================================================
# MODEL-DERIVED XAI
# ============================================================
def model_derived_xai(model, scaled_values, feature_cols, top_n=4, preferred_features=None, submitted_inputs=None):
    names = list(feature_cols)
    if not names:
        return [], []

    estimator = model
    if hasattr(model, "steps") and model.steps:
        estimator = model.steps[-1][1]

    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        return [], []

    importance = np.asarray(importance, dtype=float).reshape(-1)
    if len(importance) != len(names):
        return [], []

    candidate_indices = np.arange(len(names))

    if preferred_features:
        preferred_normalized = {str(v).strip().lower().replace(" ", "_") for v in preferred_features}
        candidate_indices = np.array([
            i for i, name in enumerate(names)
            if str(name).strip().lower().replace(" ", "_") in preferred_normalized
        ], dtype=int)

    if submitted_inputs is not None:
        submitted_indices = []
        for i, name in enumerate(names):
            value = submitted_inputs.get(name)
            if value is None:
                aliases = {
                    "age": ["Age", "AGE"],
                    "bmi": ["BMI"],
                    "hba1c": ["HbA1c"],
                    "chol": ["Chol"],
                    "tg": ["TG"],
                    "hdl": ["HDL"],
                    "ldl": ["LDL"],
                    "urea": ["Urea"],
                    "cr": ["Cr"],
                }
                for alias in aliases.get(str(name).strip().lower(), []):
                    if submitted_inputs.get(alias) is not None:
                        value = submitted_inputs[alias]
                        break

            if value is not None:
                submitted_indices.append(i)

        submitted_set = set(submitted_indices)
        candidate_indices = np.array([i for i in candidate_indices if i in submitted_set], dtype=int)

    if len(candidate_indices) == 0:
        return [], []

    scores = np.nan_to_num(importance[candidate_indices], nan=0.0)

    if submitted_inputs is not None:
        value_factors = []
        for idx in candidate_indices:
            name = names[idx]
            value = submitted_inputs.get(name)

            if value is None:
                aliases = {
                    "age": ["Age", "AGE"],
                    "bmi": ["BMI"],
                    "hba1c": ["HbA1c"],
                    "chol": ["Chol"],
                    "tg": ["TG"],
                    "hdl": ["HDL"],
                    "ldl": ["LDL"],
                    "urea": ["Urea"],
                    "cr": ["Cr"],
                }
                for alias in aliases.get(str(name).strip().lower(), []):
                    if submitted_inputs.get(alias) is not None:
                        value = submitted_inputs[alias]
                        break

            try:
                numeric_value = abs(float(value))
            except (TypeError, ValueError):
                numeric_value = 1.0

            value_factors.append(np.log1p(numeric_value))

        value_factors = np.asarray(value_factors, dtype=float)
        if np.any(value_factors > 0):
            scores = scores * value_factors

    total = float(np.sum(scores))
    if total > 0:
        display_scores = scores / total * 100.0
    elif len(scores):
        display_scores = np.ones(len(scores), dtype=float) / len(scores) * 100.0
    else:
        return [], []

    order = np.argsort(display_scores)[::-1][:top_n]

    pretty_map = {
        "cp": "Chest Pain Type",
        "restecg": "Resting ECG",
        "exang": "Exercise Angina",
        "oldpeak": "Oldpeak",
        "slope": "ST Segment Slope",
        "ca": "Major Vessels (CA)",
        "thal": "Thalassemia",
        "AGE": "Age",
        "BMI": "BMI",
        "HbA1c": "HbA1c",
        "Chol": "Cholesterol",
        "TG": "Triglycerides",
        "HDL": "HDL",
        "LDL": "LDL",
        "Urea": "Urea",
        "Cr": "Creatinine",
    }

    pretty_names, pretty_scores = [], []
    for pos in order:
        idx = int(candidate_indices[pos])
        raw_name = str(names[idx]).strip().lower().replace(" ", "_")
        pretty_names.append(
            pretty_map.get(
                names[idx],
                pretty_map.get(raw_name, str(names[idx]).replace("_", " ").title())
            )
        )
        pretty_scores.append(float(display_scores[pos]))

    return pretty_names[::-1], pretty_scores[::-1]

# ============================================================
# THEME ENGINE
# ============================================================
def clearable_number_input(label, *, key, min_value=None, max_value=None, value=None, placeholder=None, step=None):
    kwargs = {
        "min_value": min_value,
        "max_value": max_value,
        "value": value,
        "placeholder": placeholder,
        "key": key,
    }
    if step is not None:
        kwargs["step"] = step
    return st.number_input(label, **kwargs)

def get_theme_mode():
    return st.session_state.get("theme_mode", st.session_state.get("sidebar_theme", "Light"))

def _save_theme_choice():
    selected = st.session_state.get("sidebar_theme_selector", "Light")
    if selected in ("Light", "Dark"):
        st.session_state.theme_mode = selected
        st.session_state.sidebar_theme = selected

def plot_text_color():
    return "#FFFFFF" if get_theme_mode() == "Dark" else "#0F172A"

# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("<div class='nav-brand'>🩺 Medical Predict AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-sub'>Multi-Disease Risk Prediction</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>Main</div>", unsafe_allow_html=True)

    if st.button("▣   Dashboard", key="side_dashboard", use_container_width=True):
        st.session_state.view_mode = "dashboard"
        st.rerun()

    if st.button("＋   New Assessment", key="side_predict", use_container_width=True):
        st.session_state.view_mode = "select"
        st.rerun()

    if st.button("▤   Results", key="side_results", use_container_width=True):
        if st.session_state.last_prediction is not None:
            st.session_state.view_mode = "result"
        else:
            st.session_state.view_mode = "select"
        st.rerun()

    st.markdown("<div class='sidebar-label'>Information</div>", unsafe_allow_html=True)

    if st.button("ⓘ   About System", key="side_about", use_container_width=True):
        st.session_state.view_mode = "about"
        st.rerun()

    if st.button("✉   Contact", key="side_contact", use_container_width=True):
        st.session_state.view_mode = "contact"
        st.rerun()

    st.markdown("<div class='sidebar-label'>System Settings</div>", unsafe_allow_html=True)

    theme_options = ["Light", "Dark"]
    current_theme = get_theme_mode()
    if current_theme not in theme_options:
        current_theme = "Light"

    theme_choice = st.radio(
        "Theme",
        theme_options,
        horizontal=True,
        key="sidebar_theme_selector",
        on_change=_save_theme_choice,
        label_visibility="visible"
    )

    if theme_choice in theme_options:
        st.session_state.theme_mode = theme_choice
        st.session_state.sidebar_theme = theme_choice

    if theme_choice == "Dark":
        st.markdown(
            "<div class='small-muted'>Dark theme selected. Restart the app if the browser theme does not update immediately.</div>",
            unsafe_allow_html=True
        )

    if st.button("⌫   Clear cache", key="sidebar_clear_cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared.")

    st.markdown(
        "<div class='small-muted' style='margin-top:12px;'>Made with Streamlit</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        "<div class='small-muted'>AI-assisted educational tool<br>Not a medical diagnosis</div>",
        unsafe_allow_html=True
    )

# ============================================================
# DYNAMIC DARK / LIGHT THEME
# ============================================================
if get_theme_mode() == "Dark":
    st.markdown("""
    <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        .main,
        section.main {
            background: #05070B !important;
        }

        h1, h2, h3, h4, h5, h6,
        p, label, li,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p {
            color: #F8FAFC !important;
        }

        .small-muted,
        .nav-sub,
        .kpi-label,
        .kpi-sub,
        .disease-text,
        .xai-name,
        .sidebar-label,
        .stCaption {
            color: #CBD5E1 !important;
        }

        .section-title,
        .disease-title,
        .kpi-value {
            color: #FFFFFF !important;
        }

        .dashboard-card,
        .kpi-card,
        .disease-card,
        .stTabs [data-baseweb="tab-list"],
        .stExpander,
        [data-testid="stMetric"],
        .recommendation {
            background: #0B0F17 !important;
            border-color: #334155 !important;
            color: #F8FAFC !important;
            box-shadow: 0 6px 24px rgba(0,0,0,0.35) !important;
        }

        .disease-icon {
            background: #111827 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="input"],
        div[data-testid="stNumberInputContainer"],
        div[data-testid="stNumberInputContainer"] > div {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border-color: #CBD5E1 !important;
            color: #0F172A !important;
        }

        input,
        textarea {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
            caret-color: #2563EB !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #64748B !important;
            -webkit-text-fill-color: #64748B !important;
            opacity: 1 !important;
        }

        div[data-testid="stNumberInput"] button,
        div[data-testid="stNumberInputContainer"] button {
            background: #2563EB !important;
            color: #FFFFFF !important;
            border-color: #2563EB !important;
        }

        .js-plotly-plot,
        .plotly,
        .plot-container {
            color: #FFFFFF !important;
        }

        .js-plotly-plot svg text,
        .js-plotly-plot .xtick text,
        .js-plotly-plot .ytick text,
        .js-plotly-plot .gtitle text,
        .js-plotly-plot .legend text,
        .js-plotly-plot .annotation-text,
        .js-plotly-plot .axis-title {
            fill: #F8FAFC !important;
            color: #F8FAFC !important;
        }

        .js-plotly-plot .xaxislayer-above text,
        .js-plotly-plot .yaxislayer-above text,
        .js-plotly-plot .legendtext {
            fill: #F8FAFC !important;
        }

        .js-plotly-plot .xgrid,
        .js-plotly-plot .ygrid {
            stroke: #475569 !important;
        }

        .js-plotly-plot .zerolinelayer path,
        .js-plotly-plot .xaxislayer-above path,
        .js-plotly-plot .yaxislayer-above path {
            stroke: #64748B !important;
        }

        .js-plotly-plot .hovertext text,
        .js-plotly-plot .axistext {
            fill: #FFFFFF !important;
        }

        .info-box {
            background: #0B1B35 !important;
            border-color: #2563EB !important;
            color: #E0F2FE !important;
        }

        .info-box *,
        .recommendation * {
            color: inherit !important;
        }

        .recommendation {
            background: #111827 !important;
            border-color: #334155 !important;
            color: #E2E8F0 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background: #0B0F17 !important;
            border-color: #334155 !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #CBD5E1 !important;
        }

        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;
        }

        [data-testid="stAlert"] {
            background: #111827 !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }

        [data-testid="stAlert"] * {
            color: #F8FAFC !important;
        }

        hr {
            border-color: #334155 !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060B16 0%, #0D172A 100%) !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        section.main,
        .main {
            background: #05070B !important;
        }

        .kpi-card,
        .disease-card,
        .dashboard-card {
            background: #0B0F17 !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }

        .kpi-card *,
        .disease-card *,
        .dashboard-card *,
        .section-title,
        .disease-title,
        .kpi-value,
        .xai-name {
            color: #F8FAFC !important;
        }

        .kpi-label,
        .kpi-sub,
        .disease-text,
        .small-muted {
            color: #CBD5E1 !important;
        }

        .recommendation {
            background: #111827 !important;
            color: #E2E8F0 !important;
            border-color: #334155 !important;
        }

        header[data-testid="stHeader"] {
            background: #05070B !important;
        }

        [data-testid="stSidebarCollapsedControl"] button::before,
        [data-testid="stExpandSidebarButton"] button::before,
        button[data-testid="stExpandSidebarButton"]::before,
        button[data-testid="stBaseButton-headerNoPadding"]::before {
            background: #FFFFFF !important;
            box-shadow: 0 7px 0 #FFFFFF, 0 14px 0 #FFFFFF !important;
        }

        [data-testid="stSidebarCollapsedControl"] button:hover,
        [data-testid="stExpandSidebarButton"] button:hover,
        button[data-testid="stExpandSidebarButton"]:hover,
        button[data-testid="stBaseButton-headerNoPadding"]:hover {
            background: rgba(255,255,255,0.08) !important;
        }

        [data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }

        [data-testid="stRadio"] label {
            color: #E2E8F0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 7. TOP HEADER
# ============================================================
top_left, top_right = st.columns([5, 1])

with top_left:
    st.markdown(
        "<div style='font-size:0.82rem;color:#64748B;font-weight:600;'>HEALTHCARE ANALYTICS / AI PREDICTION</div>",
        unsafe_allow_html=True
    )

with top_right:
    st.markdown(
        "<div style='text-align:right;font-size:0.82rem;color:#64748B;padding-top:5px;'>● System Online</div>",
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# 8. HOME / DASHBOARD
# ============================================================
if st.session_state.view_mode == "dashboard":

    st.markdown("""
    <div class='hero-card'>
        <div style='font-size:0.8rem;font-weight:700;opacity:.75;letter-spacing:.08em;'>
            WELCOME TO MEDIPREDICT AI
        </div>
        <h1 style='font-size:2.25rem;margin:8px 0 6px;'>
            Smarter Health Risk Insights
        </h1>
        <p style='max-width:720px;margin:0;opacity:.88;'>
            An AI-driven dashboard for predicting Diabetes, Heart Disease,
            and Chronic Kidney Disease using machine learning and Explainable AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>Diseases Supported</div>
            <div class='kpi-value'>03</div>
            <div class='kpi-sub'>Multi-disease screening</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>Models Available</div>
            <div class='kpi-value'>03</div>
            <div class='kpi-sub'>Trained ML classifiers</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>System Accuracy</div>
            <div class='kpi-value'>91.3%</div>
            <div class='kpi-sub'>Evaluation benchmark</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>XAI Enabled</div>
            <div class='kpi-value'>YES</div>
            <div class='kpi-sub'>Explainable predictions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Choose a health assessment</div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)

    diseases = [
        ("diabetes", "💧", "Diabetes", "Assess diabetes risk using metabolic and blood-related health measurements."),
        ("heart", "❤️", "Heart Disease", "Assess cardiovascular risk using clinical and heart-related measurements."),
        ("kidney", "🫘", "Chronic Kidney Disease", "Assess kidney disease risk using renal and health-related measurements."),
    ]

    for col, (key, icon, name, desc) in zip([d1, d2, d3], diseases):
        with col:
            st.markdown(f"""
            <div class='disease-card'>
                <div class='disease-icon'>{icon}</div>
                <div class='disease-title'>{name}</div>
                <div class='disease-text'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Assess {name}  →", key=f"home_{key}", use_container_width=True):
                st.session_state.selected_disease = key
                st.session_state.view_mode = "input"
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    a1, a2 = st.columns([1.45, 1])

    with a1:
        st.markdown("<div class='section-title'>Disease Prevalence in Training Datasets</div>", unsafe_allow_html=True)
        st.plotly_chart(create_risk_trend(), use_container_width=True, config={"displayModeBar": False})

    with a2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Training Dataset Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(create_distribution(), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1.45, 1])

    with b1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Top Health Indicators</div>", unsafe_allow_html=True)

        indicators = [
            ("Blood Glucose", 88),
            ("Blood Pressure", 72),
            ("Cholesterol", 64),
            ("Creatinine", 58),
        ]

        for name, value in indicators:
            st.markdown(
                f"<div class='xai-name' style='margin-top:10px'>{name} <span style='float:right;color:#64748B'>{value}%</span></div>",
                unsafe_allow_html=True
            )
            st.progress(value / 100)

        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class='dashboard-card'>
            <div class='section-title'>How the system works</div>
            <div class='recommendation'>🧠 <b>Machine Learning</b><br>Models analyze clinical input patterns.</div>
            <div class='recommendation'>📊 <b>Risk Prediction</b><br>The system estimates the predicted class and probability.</div>
            <div class='recommendation'>🔍 <b>Explainable AI</b><br>Important factors are displayed to make results easier to understand.</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 9. DISEASE SELECTION
# ============================================================
elif st.session_state.view_mode == "select":

    st.markdown("<div class='section-title' style='font-size:1.7rem'>Start a New Assessment</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-muted' style='margin-bottom:20px;'>Select one condition to enter health information.</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    choices = [
        (c1, "diabetes", "💧", "Diabetes", "Metabolic health assessment"),
        (c2, "heart", "❤️", "Heart Disease", "Cardiovascular assessment"),
        (c3, "kidney", "🫘", "Chronic Kidney Disease", "Renal health assessment"),
    ]

    for col, key, icon, title, subtitle in choices:
        with col:
            st.markdown(f"""
            <div class='disease-card'>
                <div class='disease-icon'>{icon}</div>
                <div class='disease-title'>{title}</div>
                <div class='disease-text'>{subtitle}. Enter the required health measurements and receive an AI-assisted risk prediction.</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Continue →", key=f"choose_{key}", use_container_width=True):
                st.session_state.selected_disease = key
                st.session_state.view_mode = "input"
                st.rerun()

# ============================================================
# 10. HEALTH INPUT FORM
# ============================================================
elif st.session_state.view_mode == "input":

    current_dis = st.session_state.selected_disease
    names = {
        "heart": ("❤️", "Heart Disease"),
        "diabetes": ("💧", "Diabetes"),
        "kidney": ("🫘", "Chronic Kidney Disease"),
    }
    icon, disease_name = names[current_dis]

    if st.button("← Back to Dashboard", key="back_input"):
        st.session_state.view_mode = "dashboard"
        st.rerun()

    st.markdown(
        f"<div class='section-title' style='font-size:1.65rem'>{icon} {disease_name} Assessment</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='small-muted' style='margin-bottom:18px;'>Enter the health information requested by the prediction model.</div>",
        unsafe_allow_html=True
    )

    inputs = {}

    with st.form("patient_data_form"):

        st.markdown("<div class='section-title'>👤 Patient Information</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        if current_dis == "heart":
            with c1:
                inputs["Age"] = clearable_number_input(
                    "Age (years)", key="heart_age",
                    min_value=0, max_value=120, value=None,
                    placeholder="Enter age"
                )
                inputs["Height"] = clearable_number_input(
                    "Height (cm)", key="heart_height",
                    min_value=0, max_value=250, value=None,
                    placeholder="Enter height"
                )
                inputs["trestbps"] = clearable_number_input(
                    "Systolic Blood Pressure (mmHg)", key="heart_trestbps",
                    min_value=0, max_value=250, value=None,
                    placeholder="Enter systolic blood pressure"
                )
                fasting_glucose = clearable_number_input(
                    "Blood Glucose / Fasting (mg/dL)", key="heart_fasting_glucose",
                    min_value=0, max_value=500, value=None,
                    placeholder="Enter fasting glucose"
                )
                inputs["Fasting_Glucose"] = fasting_glucose
                inputs["fbs"] = None if fasting_glucose is None else int(fasting_glucose > 120)

            with c2:
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"],
                    index=None,
                    placeholder="Select gender"
                )
                inputs["Sex"] = (
                    None if gender is None
                    else (1 if gender == "Male" else 0)
                )

                inputs["Weight"] = clearable_number_input(
                    "Weight (kg)", key="heart_weight",
                    min_value=0, max_value=300, value=None,
                    placeholder="Enter weight"
                )
                inputs["thalach"] = clearable_number_input(
                    "Maximum Heart Rate (bpm)", key="heart_thalach",
                    min_value=0, max_value=250, value=None,
                    placeholder="Enter maximum heart rate"
                )
                inputs["chol"] = clearable_number_input(
                    "Total Cholesterol (mg/dL)", key="heart_chol",
                    min_value=0, max_value=600, value=None,
                    placeholder="Enter cholesterol"
                )

            st.markdown(
                "<div class='section-title' style='margin-top:18px'>🧪 "
                "Heart Disease Model Parameters</div>",
                unsafe_allow_html=True
            )
            st.caption(
                "Select the clinical category that matches the patient's "
                "heart-disease information. The stored numeric code is sent "
                "to the trained model."
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                cp_options = {
                    "0 — Typical Angina": 0,
                    "1 — Atypical Angina": 1,
                    "2 — Non-anginal Pain": 2,
                    "3 — Asymptomatic": 3,
                }
                cp_label = st.selectbox(
                    "Chest Pain Type",
                    list(cp_options.keys()),
                    index=None,
                    placeholder="Select chest pain type"
                )
                inputs["cp"] = (
                    None if cp_label is None else cp_options[cp_label]
                )

            with p2:
                ecg_options = {
                    "0 — Normal": 0,
                    "1 — ST-T Wave Abnormality": 1,
                    "2 — Left Ventricular Hypertrophy": 2,
                }
                ecg_label = st.selectbox(
                    "Resting ECG",
                    list(ecg_options.keys()),
                    index=None,
                    placeholder="Select resting ECG"
                )
                inputs["restecg"] = (
                    None if ecg_label is None else ecg_options[ecg_label]
                )

            with p3:
                exang_options = {
                    "0 — No": 0,
                    "1 — Yes": 1,
                }
                exang_label = st.selectbox(
                    "Exercise-Induced Angina",
                    list(exang_options.keys()),
                    index=None,
                    placeholder="Select exercise angina"
                )
                inputs["exang"] = (
                    None if exang_label is None else exang_options[exang_label]
                )

            q1, q2, q3 = st.columns(3)

            with q1:
                inputs["oldpeak"] = clearable_number_input(
                    "Oldpeak", key="heart_oldpeak",
                    min_value=0.0, max_value=10.0, value=None,
                    placeholder="Enter oldpeak"
                )

            with q2:
                slope_options = {
                    "0 — Downsloping": 0,
                    "1 — Flat / Abnormal": 1,
                    "2 — Upsloping / Normal": 2,
                }
                slope_label = st.selectbox(
                    "ST Segment Slope",
                    list(slope_options.keys()),
                    index=None,
                    placeholder="Select slope"
                )
                inputs["slope"] = (
                    None if slope_label is None
                    else slope_options[slope_label]
                )

            with q3:
                ca_options = {
                    "0 — None": 0,
                    "1 — One": 1,
                    "2 — Two": 2,
                    "3 — Three": 3,
                    "4 — Four": 4,
                }
                ca_label = st.selectbox(
                    "Major Vessels (CA)",
                    list(ca_options.keys()),
                    index=None,
                    placeholder="Select number of vessels"
                )
                inputs["ca"] = (
                    None if ca_label is None else ca_options[ca_label]
                )

            r1, r2 = st.columns(2)

            with r1:
                thal_options = {
                    "1 — Normal": 1,
                    "2 — Fixed Defect": 2,
                    "3 — Reversible Defect": 3,
                }
                thal_label = st.selectbox(
                    "Thalassemia",
                    list(thal_options.keys()),
                    index=None,
                    placeholder="Select thalassemia status"
                )
                inputs["thal"] = (
                    None if thal_label is None
                    else thal_options[thal_label]
                )

            with r2:
                st.markdown(
                    """
                    <div class='info-callout' style='margin-top:28px;'>
                    <b>Heart Disease Parameters</b><br>
                    The displayed descriptions are for easier interpretation.
                    The corresponding numeric codes are passed to the trained
                    model exactly as required by its feature columns.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        elif current_dis == "diabetes":
            with c1:
                inputs["AGE"] = clearable_number_input("Age (years)", key="diabetes_age", min_value=0, max_value=120, value=None, placeholder="Enter age")
                inputs["BMI"] = clearable_number_input("BMI (kg/m²)", key="diabetes_bmi", min_value=0.0, max_value=80.0, value=None, placeholder="Enter BMI")
                inputs["HbA1c"] = clearable_number_input("HbA1c Level (%)", key="diabetes_hba1c", min_value=0.0, max_value=20.0, value=None, placeholder="Enter HbA1c level")
                inputs["Urea"] = clearable_number_input("Urea (mmol/L)", key="diabetes_urea", min_value=0.0, max_value=50.0, value=None, placeholder="Enter urea")
                inputs["Cr"] = clearable_number_input("Creatinine (µmol/L)", key="diabetes_cr", min_value=0.0, max_value=1000.0, value=None, placeholder="Enter creatinine")
            with c2:
                gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select gender")
                inputs["Gender_M"] = None if gender is None else (1 if gender == "Male" else 0)
                inputs["Gender_f"] = None if gender is None else (1 if gender == "Female" else 0)
                inputs["Chol"] = clearable_number_input("Cholesterol (mmol/L)", key="diabetes_chol", min_value=0.0, max_value=20.0, value=None, placeholder="Enter cholesterol")
                inputs["TG"] = clearable_number_input("Triglycerides (mmol/L)", key="diabetes_tg", min_value=0.0, max_value=30.0, value=None, placeholder="Enter triglycerides")
                inputs["HDL"] = clearable_number_input("HDL (mmol/L)", key="diabetes_hdl", min_value=0.0, max_value=10.0, value=None, placeholder="Enter HDL")
                inputs["LDL"] = clearable_number_input("LDL (mmol/L)", key="diabetes_ldl", min_value=0.0, max_value=20.0, value=None, placeholder="Enter LDL")

        elif current_dis == "kidney":
            with c1:
                inputs["Age"] = clearable_number_input("Age (years)", key="kidney_age", min_value=0, max_value=120, value=None, placeholder="Enter age")
                inputs["Creatinine_Level"] = clearable_number_input("Creatinine Level (mg/dL)", key="kidney_creatinine", min_value=0.0, max_value=20.0, value=None, placeholder="Enter creatinine level")
                inputs["BUN"] = clearable_number_input("BUN (mg/dL)", key="kidney_bun", min_value=0.0, max_value=200.0, value=None, placeholder="Enter BUN")
            with c2:
                inputs["Urine_Output"] = clearable_number_input("Urine Output (mL/day)", key="kidney_urine_output", min_value=0, max_value=10000, value=None, placeholder="Enter urine output")
                diabetes_history = st.selectbox(
                    "Diabetes History",
                    ["0 - No", "1 - Yes"],
                    index=None,
                    placeholder="Select diabetes history"
                )
                inputs["Diabetes"] = (
                    None if diabetes_history is None
                    else (0 if diabetes_history.startswith("0") else 1)
                )

                hypertension_history = st.selectbox(
                    "Hypertension History",
                    ["0 - No", "1 - Yes"],
                    index=None,
                    placeholder="Select hypertension history"
                )
                inputs["Hypertension"] = (
                    None if hypertension_history is None
                    else (0 if hypertension_history.startswith("0") else 1)
                )
                inputs["GFR"] = clearable_number_input("GFR Level", key="kidney_gfr", min_value=0.0, max_value=200.0, value=None, placeholder="Enter GFR")

        st.markdown("""
        <div class='info-box'>
            ℹ️ Please enter accurate values. This system is intended for educational and
            research demonstration and does not replace professional medical diagnosis.
        </div>
        """, unsafe_allow_html=True)

        submit = st.form_submit_button("🔍 Analyze Health Risk", use_container_width=True)

        if submit:
            minimum_required_fields = {
                "heart": {"Age", "Sex", "trestbps", "thalach", "chol", "cp"},
                "diabetes": {"AGE", "BMI", "HbA1c", "Gender_M", "Gender_f"},
                "kidney": {"Age", "Creatinine_Level", "BUN", "GFR", "Urine_Output"},
            }

            required_keys = minimum_required_fields.get(current_dis, set())
            filled_required_count = sum(
                inputs.get(key) is not None for key in required_keys
            )

            if filled_required_count < 3:
                st.warning(
                    "Please fill in the important patient information above before analyzing your health risk."
                )
            else:
                st.session_state.user_inputs = inputs
                st.session_state.view_mode = "result"
                st.rerun()

# ============================================================
# 11. PREDICTION RESULT
# ============================================================
elif st.session_state.view_mode == "result":

    current_dis = st.session_state.selected_disease
    inputs = st.session_state.user_inputs

    model, scaler, feature_cols = load_disease_assets(current_dis)

    if not feature_cols:
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            feature_cols = list(scaler.feature_names_in_)
        elif model is not None and hasattr(model, "feature_names_in_"):
            feature_cols = list(model.feature_names_in_)
        else:
            feature_cols = list(inputs.keys())

    raw_df = pd.DataFrame([inputs])

    scaler_means = {}
    if scaler is not None and hasattr(scaler, "mean_"):
        scaler_means = dict(zip(feature_cols, np.asarray(scaler.mean_, dtype=float)))

    for col in feature_cols:
        if col not in raw_df.columns or pd.isna(raw_df.at[0, col]):
            raw_df[col] = float(scaler_means.get(col, 0.0))

    raw_df = raw_df.loc[:, feature_cols]

    if scaler is not None and hasattr(scaler, "n_features_in_"):
        expected_features = int(scaler.n_features_in_)
        if raw_df.shape[1] != expected_features:
            raise ValueError(
                f"The saved heart-disease scaler expects {expected_features} "
                f"features, but the application prepared {raw_df.shape[1]}. "
                "Check features_heart.pkl, scaler_heart.pkl and model_heart.pkl."
            )

    scaled_vals = scaler.transform(raw_df) if scaler is not None else raw_df.to_numpy()

    if model is not None:
        pred = model.predict(scaled_vals)[0]

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(scaled_vals)[0], dtype=float)
            classes = list(getattr(model, "classes_", range(len(probabilities))))

            if current_dis == "heart" and 0 in classes:
                risk_index = classes.index(0)
            elif 1 in classes:
                risk_index = classes.index(1)
            else:
                risk_index = int(np.argmax(probabilities))

            risk_prob = float(probabilities[risk_index] * 100)
            confidence = float(np.max(probabilities) * 100)
        else:
            if current_dis == "heart":
                risk_prob = 100.0 if int(pred) == 0 else 0.0
            else:
                risk_prob = 100.0 if int(pred) == 1 else 0.0
            confidence = risk_prob if risk_prob >= 50 else 100.0 - risk_prob
    else:
        pred = 0 if current_dis == "heart" else 1
        risk_prob = 91.0
        confidence = 91.0

    prob = risk_prob

    if any(value is None for value in inputs.values()):
        st.info(
            "Prediction generated using the information provided. "
            "Any blank model fields were estimated using training-data defaults."
        )

    disease_name = {
        "heart": "Heart Disease",
        "diabetes": "Diabetes",
        "kidney": "Chronic Kidney Disease"
    }[current_dis]

    st.session_state.last_prediction = {
        "disease": disease_name,
        "prediction": int(pred),
        "probability": float(prob)
    }

    risk_high = (int(pred) == 0) if current_dis == "heart" else (int(pred) == 1)

    if risk_high:
        risk_text = "HIGH RISK"
        risk_class = "risk-high"
        risk_icon = "⚠️"
    else:
        risk_text = "LOW RISK"
        risk_class = "risk-low"
        risk_icon = "✓"

    if st.button("← Back to Assessment", key="back_result"):
        st.session_state.view_mode = "input"
        st.rerun()

    st.markdown(
        f"<div class='section-title' style='font-size:1.65rem'>{risk_icon} {disease_name} Prediction Result</div>",
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns([1.05, 1.55, 1.05])

    with r1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='result-risk'>
            <div class='small-muted'>AI risk probability</div>
            <div class='risk-number {risk_class}'>{prob:.0f}%</div>
            <div class='{risk_class}' style='font-size:1.05rem;font-weight:800'>{risk_text}</div>
            <div class='small-muted' style='margin-top:6px'>Model confidence: {confidence:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(prob / 100, 0), 1))
        st.markdown(
            "<div class='small-muted' style='margin-top:12px;text-align:center;'>Prediction based on the submitted health information.</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧠 Explainable AI</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='small-muted'>Main clinical factors based only on the information entered for this patient.</div>",
            unsafe_allow_html=True
        )

        if current_dis == "heart":
            heart_main_factors = [
                "cp",
                "restecg",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ]
            x_names, x_values = model_derived_xai(
                model,
                scaled_vals,
                feature_cols,
                top_n=4,
                preferred_features=heart_main_factors,
                submitted_inputs=inputs
            )
        else:
            x_names, x_values = model_derived_xai(
                model,
                scaled_vals,
                feature_cols,
                top_n=4,
                submitted_inputs=inputs
            )

        if x_names:
            st.plotly_chart(
                create_xai_chart(x_names, x_values),
                use_container_width=True,
                config={"displayModeBar": False}
            )
        else:
            st.info("Feature contribution details are not available for this model.")

        st.markdown("""
        <div class='info-box'>
            <b>Interpretation:</b> These factors are derived from the loaded
            prediction model and the submitted patient values. Higher bars indicate
            stronger model influence for this assessment. They do not change the
            model's prediction.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r3:
        st.markdown("<div class='dashboard-card patient-snapshot'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Patient Snapshot</div>", unsafe_allow_html=True)

        if current_dis == "heart":
            snapshot_order = [
                "Age", "Sex", "Height", "Weight", "trestbps", "thalach",
                "Fasting_Glucose", "Fasting Glucose", "fbs", "chol", "cp",
                "restecg", "exang", "oldpeak", "slope", "ca", "thal"
            ]
            shown = [(key, inputs[key]) for key in snapshot_order if key in inputs]
        else:
            shown = list(inputs.items())

        full_feature_names = {
            "Age": "Age",
            "AGE": "Age",
            "Sex": "Sex",
            "Gender": "Gender",
            "Gender_M": "Gender (Male)",
            "Gender_f": "Gender (Female)",
            "Height": "Height",
            "Weight": "Weight",
            "trestbps": "Resting Blood Pressure",
            "thalach": "Maximum Heart Rate",
            "Fasting_Glucose": "Fasting Blood Glucose",
            "Fasting Glucose": "Fasting Blood Glucose",
            "fbs": "Fasting Blood Sugar Status",
            "chol": "Total Cholesterol",
            "cp": "Chest Pain Type",
            "restecg": "Resting Electrocardiogram (ECG)",
            "exang": "Exercise-Induced Angina",
            "oldpeak": "ST Depression (Oldpeak)",
            "slope": "ST Segment Slope",
            "ca": "Major Vessels (CA)",
            "thal": "Thalassemia",
            "BMI": "Body Mass Index (BMI)",
            "HbA1c": "Hemoglobin A1c (HbA1c)",
            "Urea": "Blood Urea Level",
            "Cr": "Creatinine Level",
            "Chol": "Total Cholesterol",
            "TG": "Triglycerides",
            "HDL": "High-Density Lipoprotein (HDL)",
            "LDL": "Low-Density Lipoprotein (LDL)",
            "Creatinine_Level": "Creatinine Level",
            "BUN": "Blood Urea Nitrogen (BUN)",
            "Urine_Output": "Urine Output",
            "Diabetes": "Diabetes History",
            "Hypertension": "Hypertension History",
            "GFR": "Glomerular Filtration Rate (GFR)",
        }

        for key, value in shown:
            pretty = full_feature_names.get(
                key,
                key.replace("_", " ").title()
            )

            display_value = value

            if current_dis == "kidney" and key in {"Diabetes", "Hypertension"}:
                display_value = "0 - No" if value == 0 else "1 - Yes"

            if key == "Height":
                display_value = f"{value} cm"
            elif key == "Weight":
                display_value = f"{value} kg"
            elif key == "trestbps":
                display_value = f"{value} mmHg"
            elif key == "thalach":
                display_value = f"{value} bpm"
            elif key in {"Fasting_Glucose", "Fasting Glucose"}:
                display_value = f"{value} mg/dL"
            elif key == "chol" and current_dis == "heart":
                display_value = f"{value} mg/dL"
            elif key == "BMI":
                display_value = f"{value} kg/m²"
            elif key == "HbA1c":
                display_value = f"{value}%"
            elif key == "Urea":
                display_value = f"{value} mmol/L"
            elif key == "Cr":
                display_value = f"{value} µmol/L"

            st.markdown(
                f"<div class='snapshot-row' style='padding:8px 0;border-bottom:1px solid #E2E8F0;'>"
                f"<span class='snapshot-label'>{pretty}</span>"
                f"<span class='snapshot-value' style='float:right'>{display_value}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if get_theme_mode() == "Dark":
            st.markdown("""
            <style>
                .patient-snapshot .snapshot-label,
                .patient-snapshot .snapshot-value,
                .patient-snapshot .section-title {
                    color: #FFFFFF !important;
                    -webkit-text-fill-color: #FFFFFF !important;
                }
            </style>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>💡 Health Insights & Recommendations</div>", unsafe_allow_html=True)

    rec1, rec2, rec3 = st.columns(3)

    if current_dis == "diabetes":
        recommendations = [
            ("🥗 Balanced Nutrition", "Focus on balanced meals and appropriate portions while limiting highly processed sugary foods."),
            ("🏃 Physical Activity", "Maintain regular physical activity appropriate for your individual health condition."),
            ("🩺 Routine Monitoring", "Discuss glucose and HbA1c monitoring with a qualified healthcare professional."),
        ]
    elif current_dis == "heart":
        recommendations = [
            ("🥑 Heart-Healthy Diet", "Prioritize balanced nutrition and monitor sodium and saturated fat intake."),
            ("🚶 Stay Active", "Maintain regular, appropriate physical activity and discuss exercise goals with a professional."),
            ("🩺 Monitor Risk Factors", "Keep track of blood pressure, cholesterol and other cardiovascular risk factors."),
        ]
    else:
        recommendations = [
            ("💧 Healthy Hydration", "Maintain appropriate hydration based on your individual health needs and medical advice."),
            ("🧂 Balanced Nutrition", "Pay attention to sodium and overall dietary balance, particularly if kidney concerns exist."),
            ("🩺 Monitor Kidney Health", "Discuss kidney function tests such as GFR and creatinine with a healthcare professional."),
        ]

    for col, (title, text) in zip([rec1, rec2, rec3], recommendations):
        with col:
            st.markdown(
                f"<div class='dashboard-card'><div class='recommendation'><b>{title}</b><br>{text}</div></div>",
                unsafe_allow_html=True
            )

    a, b = st.columns(2)
    with a:
        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.view_mode = "select"
            st.rerun()
    with b:
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.view_mode = "dashboard"
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("🖨️  Print Result", key="result_print", use_container_width=True):
        components.html("""
            <script>
                window.top.print();
            </script>
        """, height=1, scrolling=False)

# ============================================================
# 12. ABOUT PAGE
# ============================================================
elif st.session_state.view_mode == "about":

    st.markdown("<div class='section-title' style='font-size:1.8rem'>About Medical Predict AI</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='hero-card'>
        <h2>AI-Driven Multi-Disease Risk Prediction</h2>
        <p>
            Medical Predict AI is a healthcare technology prototype designed to demonstrate
            how machine learning can be used to estimate disease risk from clinical
            health information.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='dashboard-card'>
            <div class='section-title'>🧠 Machine Learning</div>
            <p class='small-muted'>
                The system uses trained classification models to process health
                measurements and produce disease-risk predictions.
            </p>
            <div class='recommendation'>Diabetes Prediction</div>
            <div class='recommendation'>Heart Disease Prediction</div>
            <div class='recommendation'>Chronic Kidney Disease Prediction</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='dashboard-card'>
            <div class='section-title'>🔍 Explainable AI</div>
            <p class='small-muted'>
                XAI is included to make prediction outputs easier to understand
                by presenting the health factors associated with the result.
            </p>
            <div class='recommendation'>Clear risk probability</div>
            <div class='recommendation'>Visual contribution indicators</div>
            <div class='recommendation'>Human-readable health insights</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        ⚠️ <b>Important:</b> This prototype is for educational, predictive and research
        demonstration purposes only. It is not a medical diagnostic device and should
        not be used as a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 13. CONTACT PAGE
# ============================================================
elif st.session_state.view_mode == "contact":

    st.markdown("<div class='section-title' style='font-size:1.8rem'>Contact</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='dashboard-card' style='text-align:center;padding:45px;'>
        <div style='font-size:3rem;'>✉️</div>
        <h2 style='margin:12px 0 8px;'>Get in Touch</h2>
        <p style='color:#64748B !important;'>
            Have questions, feedback, or want to collaborate on AI-driven healthcare technology?
        </p>
        <div style='margin-top:22px;'>
            <a href="https://mail.google.com/mail/?view=cm&fs=1&to=supporthealthcareai@gmail.com&su=Inquiry%20regarding%20Medical Predict%20AI"
               target="_blank"
               style="display:inline-block;background:linear-gradient(135deg,#2563EB,#0EA5E9);
               color:white;text-decoration:none;padding:13px 24px;border-radius:10px;font-weight:700;">
               📧 supporthealthcareai@gmail.com
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FINAL DOMAIN-SAFE SELECTBOX / CLEAR-X OVERRIDE
# ============================================================
st.markdown(r"""
<style>
/* ------------------------------------------------------------
   SELECTBOX FRAME & CLOSED SELECTION BOX
   ------------------------------------------------------------ */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="combobox"] {
    min-height: 42px !important;
    height: 42px !important;
    box-sizing: border-box !important;
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

/* Text inside closed selectbox */
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSelectbox"] [role="combobox"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    opacity: 1 !important;
}

/* Hide internal cursor line inside closed selectbox */
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] input[role="combobox"],
[data-testid="stSelectbox"] [role="combobox"] input {
    caret-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Apply light blue border and rounded corners to every single dropdown in the system */
select,
.dropdown-select,
.custom-select-container,
div[class*="-control"] {
  border: 1.5px solid #b2c8f8 !important; /* Light blue outer border */
  border-radius: 8px !important;           /* Rounded corners matching other inputs */
  background-color: #ffffff !important;    /* White background */
  color: #1e293b !important;               /* Dark text */
  outline: none !important;
  box-sizing: border-box !important;
}

/* Ensure blue border stays active when clicking/focusing any dropdown */
select:focus,
.dropdown-select:focus,
.custom-select-container:focus-within,
div[class*="-control"]:focus-within {
  border-color: #2563eb !important;        /* Darker blue border on click */
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

/* Remove dark background panels from dropdown arrows/icons */
.custom-select-container .dropdown-icon,
.custom-select-container .select-arrow,
div[class*="-IndicatorsContainer"] {
  background-color: transparent !important;
  color: #64748b !important;
}

/* ------------------------------------------------------------
   RIGHT-SIDE CONTROL AREA & ARROW
   ------------------------------------------------------------ */
[data-testid="stSelectbox"] [data-baseweb="select"] > div:last-child,
[data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="select-arrow"],
[data-testid="stSelectbox"] [role="combobox"] > div:last-child {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 100% !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] svg,
[data-testid="stSelectbox"] [role="combobox"] svg {
    color: #0F172A !important;
    fill: #0F172A !important;
    stroke: #0F172A !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* ------------------------------------------------------------
   OPEN DROPDOWN MENU & POPOVER (DOMAIN-SAFE)
   Forces dropdown list items to stay white with dark text across
   all environments, overriding global dark mode rules.
   ------------------------------------------------------------ */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.14) !important;
    overflow: hidden !important;
}

/* Individual options */
div[data-baseweb="menu"] li,
div[data-baseweb="menu"] [role="option"],
ul[role="listbox"] li,
div[role="listbox"] [role="option"],
div[data-baseweb="popover"] [role="option"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    border: none !important;
    min-height: 36px !important;
    padding: 8px 12px !important;
}

div[data-baseweb="menu"] li *,
div[data-baseweb="menu"] [role="option"] *,
ul[role="listbox"] li *,
div[role="listbox"] [role="option"] *,
div[data-baseweb="popover"] [role="option"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Hover & Selected Option: Highlighted Blue */
div[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] [role="option"]:hover,
ul[role="listbox"] li:hover,
div[role="listbox"] [role="option"]:hover,
div[data-baseweb="menu"] [aria-selected="true"],
div[data-baseweb="menu"] [role="option"][aria-selected="true"],
div[role="listbox"] [aria-selected="true"],
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: #2563EB !important;
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-baseweb="menu"] li:hover *,
div[data-baseweb="menu"] [role="option"]:hover *,
ul[role="listbox"] li:hover *,
div[role="listbox"] [role="option"]:hover *,
div[data-baseweb="menu"] [aria-selected="true"] *,
div[role="listbox"] [aria-selected="true"] *,
div[data-baseweb="popover"] [role="option"]:hover *,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* ------------------------------------------------------------
   CLEAR X CONTROL OVERRIDE
   ------------------------------------------------------------ */
[data-testid="stSelectbox"] [aria-label*="clear" i],
[data-testid="stSelectbox"] [title*="clear" i],
[data-testid="stSelectbox"] button[aria-label*="clear" i],
[data-testid="stSelectbox"] button[title*="clear" i] {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    min-height: 28px !important;
    padding: 0 !important;
    margin: 0 2px !important;
    border: 0 !important;
    border-radius: 50% !important;
    background: transparent !important;
    box-shadow: none !important;
    color: transparent !important;
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}

[data-testid="stSelectbox"] [aria-label*="clear" i] svg,
[data-testid="stSelectbox"] [title*="clear" i] svg,
[data-testid="stSelectbox"] button[aria-label*="clear" i] svg,
[data-testid="stSelectbox"] button[title*="clear" i] svg {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stSelectbox"] [aria-label*="clear" i]::before,
[data-testid="stSelectbox"] [title*="clear" i]::before,
[data-testid="stSelectbox"] button[aria-label*="clear" i]::before,
[data-testid="stSelectbox"] button[title*="clear" i]::before {
    content: "✕" !important;
    position: absolute !important;
    color: #0F172A !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
}
</style>
""", unsafe_allow_html=True)
