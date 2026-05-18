import streamlit as st
import requests
import pandas as pd
from src.constant.streamlit_config import API_URL, FEATURE_META


def singlePrediction():
    """
    Render a single URL phishing prediction interface in a Streamlit app.
    """
    st.subheader("Single URL Prediction")
    st.caption("Select the extracted features for the URL you want to classify.")
    feature_values = {}
    cols = st.columns(3)
    for i, (col_name, label, options) in enumerate(FEATURE_META):
        with cols[i % 3]:
            reverse_mapping_dict = {value: key for key, value in options.items()}
            select = st.selectbox(label, list(options.values()), key=f"sp_{col_name}")
            feature_values[col_name] = reverse_mapping_dict[select]

    st.markdown("")
    run_button = st.button("🔍 Predict", width="stretch", type="primary", key="sp_run")

    if run_button:
        with st.spinner("Calling model..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict", json=feature_values, timeout=10
                )
                response.raise_for_status()
                data = response.json()

                prediction = data.get("prediction")
                label = data.get("label")
                confidence = data.get("confidence")

                st.markdown("")
                m1, m2, m3 = st.columns(3)
                if prediction == 1:
                    m1.metric("Prediction", "✅ " + label + " Website")
                else:
                    m1.metric("Prediction", "🚨 " + label + " Website")
                m2.metric("Raw Output", str(prediction))
                if confidence is not None:
                    m3.metric("Confidence", f"{confidence*100:.1f}%")

            except Exception as e:
                st.error(f"Error: {e}")
