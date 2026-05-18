import streamlit as st
from src.pages.single_prediction import singlePrediction
from src.pages.batch_prediction import batchPrediction
from src.pages.api_swagger import api
from src.pages.training_pipeline import trainingPipeline

st.set_page_config(
    page_title="PhishGuard - Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("# 🛡️ PhishGuard")
st.caption(
    "AI-powered phishing website detection - 30 URL features - scikit-learn + MLflow"
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Single Prediction",
        "📊 Batch Prediction",
        "⚡ API Explorer",
        "🚀 Train Pipeline",
    ]
)


with tab1:
    singlePrediction()
with tab2:
    batchPrediction()
with tab3:
    api()
with tab4:
    trainingPipeline()
