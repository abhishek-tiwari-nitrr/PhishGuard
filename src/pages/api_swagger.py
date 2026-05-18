import streamlit as st
import requests
from src.constant.streamlit_config import API_URL


def api():
    """
    Render an API Explorer section in a Streamlit app.

    Args:
        API_URL (str): Base URL of the API server.
    """
    st.subheader("API Explorer")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.json().get("model_loaded"):
            st.success(f"✅ API is live.")
        else:
            st.warning("⚠️ API is up but model not loaded")
    except Exception:
        st.warning("⚠️ API is sleeping. Open the URL once to wake it up, then refresh.")

    st.divider()

    st.iframe(
        f"{API_URL}/docs",
        height=800,
    )
