import streamlit as st
import pandas as pd
import requests, io
from src.constant.streamlit_config import API_URL, FEATURE_COLS


def batchPrediction():
    """
    Render a batch prediction interface for phishing URL detection.
    """
    st.subheader("Batch Prediction via CSV")
    col_info, col_template = st.columns([3, 1])
    with col_info:
        st.info(
            "Upload a CSV with the 30 feature columns. "
            "The app returns predictions for every row. "
            "Extra columns in your CSV are preserved."
        )
    with col_template:
        template = pd.DataFrame(columns=FEATURE_COLS)
        example_row = {col: 1 for col in FEATURE_COLS}
        example_row.update(
            {
                "having_IP_Address": -1,
                "URL_Length": -1,
                "SSLfinal_State": -1,
                "Domain_registeration_length": -1,
                "DNSRecord": -1,
                "age_of_domain": -1,
            }
        )
        template = pd.concat([template, pd.DataFrame([example_row])], ignore_index=True)
        csv_template = template.to_csv(index=False)
        st.download_button(
            "📥 Download Template CSV",
            data=csv_template,
            file_name="phishguard_template.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")

    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.write(f"**Loaded:** {len(df_input)} rows × {len(df_input.columns)} columns")
        st.dataframe(df_input.head(5), width="stretch")

        if st.button("🚀 Run Batch Prediction", type="primary", width="stretch"):
            with st.spinner(f"Predicting {len(df_input)} rows..."):
                try:
                    uploaded.seek(0)
                    response = requests.post(
                        f"{API_URL}/predict-csv",
                        files={"file": ("data.csv", uploaded.getvalue(), "text/csv")},
                        timeout=60,
                    )
                    response.raise_for_status()
                    result_df = pd.read_csv(io.StringIO(response.text))

                    phish = int((result_df["prediction"] == 0).sum())
                    legit = int((result_df["prediction"] == 1).sum())

                    st.success(
                        f"✅ Done!  Legitimate: **{legit}**  |  Phishing: **{phish}**"
                    )

                    pct = phish / len(result_df) * 100
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total URLs", len(result_df))
                    c2.metric(
                        "🚨 Phishing",
                        phish,
                        delta=f"{pct:.1f}% of total",
                        delta_color="inverse",
                    )
                    c3.metric("✅ Legitimate", legit)

                    st.download_button(
                        "📥 Download Results CSV",
                        data=response.content,
                        file_name="phishguard_predictions.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
