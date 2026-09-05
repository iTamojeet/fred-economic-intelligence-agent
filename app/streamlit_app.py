import streamlit as st
import pandas as pd
import joblib
import sys
import os
from google import genai

# --- Path setup so src/ imports work when deployed ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from snapshot import generate_economic_snapshot
from genai_explain import explain_economic_snapshot
from genai_report import generate_executive_report

st.set_page_config(page_title="Economic Intelligence Agent", layout="wide")

# --- Load data and model (cached so it doesn't reload on every interaction) ---
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/economic_dataset.csv'),
                      index_col='Date', parse_dates=True)
    return df

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(ROOT_DIR, 'models/recession_model.pkl'))
    features = joblib.load(os.path.join(ROOT_DIR, 'models/recession_model_features.pkl'))
    return model, features

df = load_data()
model, features = load_model()
snapshot = generate_economic_snapshot(df, model, features)

# --- Gemini client setup (API key from Streamlit secrets, not hardcoded) ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- Page: Dashboard ---
st.title("Economic Intelligence Agent")
st.caption(f"Latest data as of {snapshot['date']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Recession Risk", snapshot['recession_risk'],
            f"{snapshot['recession_probability']*100:.1f}% probability")
col2.metric("Inflation", f"{snapshot['inflation_rate']}%", snapshot['inflation_trend'])
col3.metric("Unemployment", f"{snapshot['unemployment_rate']}%", snapshot['unemployment_trend'])
col4.metric("Yield Curve", snapshot['yield_curve_status'], f"{snapshot['yield_curve_spread']} pp")

st.divider()

# --- Trends chart ---
st.subheader("Economic Trends")
chart_option = st.selectbox("Select indicator", [
    'Unemployment Rate', 'Inflation Rate', 'Industrial Production Growth', 'Yield Curve Spread'
])
st.line_chart(df[chart_option])

st.divider()

# --- GenAI Explanation ---
st.subheader("Ask the Economic Agent")
if client is None:
    st.warning("GEMINI_API_KEY not configured in Streamlit secrets. Explanation features disabled.")
else:
    if st.button("Explain current recession risk"):
        with st.spinner("Generating explanation..."):
            explanation = explain_economic_snapshot(snapshot, client)
        st.write(explanation)

    if st.button("Generate Monthly Economic Intelligence Report"):
        with st.spinner("Generating report..."):
            report = generate_executive_report(snapshot, client)
        st.markdown(report)
