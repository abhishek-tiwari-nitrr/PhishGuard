import streamlit as st
from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.constant.streamlit_config import PIPELINE_STEPS
import time


def trainingPipeline():
    """
    Render and execute the end-to-end machine learning training pipeline inside a Streamlit application.

    Steps:
        1. Data Ingestion
        2. Data Validation
        3. Data Transformation
        4. Model Training
        5. Model Evaluation
        6. Model Pusher

    """
    st.subheader("Training Pipeline")
    st.info(
        "Click **Start Training** to run the full 6-step ML pipeline. "
        "The pipeline pulls fresh data from MongoDB, validates it, transforms it, trains 5 classifiers with GridSearchCV, evaluates against the production model and promotes the winner."
    )
    st.info("⚠️ Training can take 5 - 20 minutes depending on hardware")
    start = st.button("🚀 Start Training", type="primary", width="stretch")

    st.divider()

    if start:
        overall_bar = st.progress(0, text="Initialising pipeline…")
        step_placeholder = st.empty()
        log_placeholder = st.empty()
        results_placeholder = st.empty()

        try:
            pipeline_config = TrainingPipelineConfig()
            logs = []

            def do_step(index, name, desc, func):
                n = len(PIPELINE_STEPS)
                pct = int((index / n) * 100)
                overall_bar.progress(pct, text=f"Step {index+1}/{n}: {name}")
                step_placeholder.markdown(f"**⏳ Running:** `{name}` - {desc}")
                start = time.time()
                result = func()
                end = time.time() - start
                logs.append(f"✅ {name} - {end:.1f}s")
                log_placeholder.markdown("\n\n".join(logs))
                return result

            di_config = DataIngestionConfig(pipeline_config)
            di_art = do_step(
                0,
                *PIPELINE_STEPS[0],
                lambda: DataIngestion(di_config).initiate_data_ingestion(),
            )

            dv_config = DataValidationConfig(pipeline_config)
            dv_art = do_step(
                1,
                *PIPELINE_STEPS[1],
                lambda: DataValidation(di_art, dv_config).initiate_data_validation(),
            )

            dt_config = DataTransformationConfig(pipeline_config)
            dt_art = do_step(
                2,
                *PIPELINE_STEPS[2],
                lambda: DataTransformation(
                    dv_art, dt_config
                ).initiate_data_transformation(),
            )

            mt_config = ModelTrainerConfig(pipeline_config)
            mt_art = do_step(
                3,
                *PIPELINE_STEPS[3],
                lambda: ModelTrainer(dt_art, mt_config).initiate_model_trainer(),
            )

            me_config = ModelEvaluationConfig(pipeline_config)
            me_art = do_step(
                4,
                *PIPELINE_STEPS[4],
                lambda: ModelEvaluation(
                    mt_art, dt_art, me_config
                ).initiate_model_evaluation(),
            )

            pushed = do_step(
                5,
                *PIPELINE_STEPS[5],
                lambda: ModelPusher(me_art, me_config).initiate_model_pusher(),
            )

            overall_bar.progress(100, text="✅ Pipeline complete!")
            step_placeholder.empty()

            st.success("🎉 Training pipeline finished successfully!")
            st.markdown("### Results")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "Model Accepted", "✅ Yes" if me_art.is_model_accepted else "❌ No"
            )
            r2.metric("Test F1 (new)", f"{mt_art.test_metric_artifact.f1_score:.4f}")
            r3.metric(
                "ΔF1 vs production",
                (
                    f"+{me_art.improved_accuracy:.4f}"
                    if me_art.improved_accuracy >= 0
                    else f"{me_art.improved_accuracy:.4f}"
                ),
            )
            r4.metric("Pushed to prod", "✅" if pushed else "⏭️ Skipped")

            with st.expander("Detailed metrics"):
                st.json(
                    {
                        "new_model": {
                            "f1_score": round(mt_art.test_metric_artifact.f1_score, 4),
                            "precision": round(
                                mt_art.test_metric_artifact.precision_score, 4
                            ),
                            "recall": round(
                                mt_art.test_metric_artifact.recall_score, 4
                            ),
                        },
                        "is_model_accepted": me_art.is_model_accepted,
                        "delta_f1": round(me_art.improved_accuracy, 4),
                        "production_model_path": me_art.best_model_path,
                    }
                )
        except Exception as e:
            overall_bar.empty()
            step_placeholder.empty()
            st.error(f"❌ Pipeline failed: {e}")
