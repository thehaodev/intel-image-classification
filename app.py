from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.inference import predict_image
from src.model_registry import CLASS_NAMES, load_checkpoint_model


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "reports" / "model_results.json"


st.set_page_config(
    page_title="Intel Image Classification",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #475569;
            font-weight: 600;
        }
        .section-note {
            color: #64748b;
            font-size: 0.92rem;
            margin-top: -0.4rem;
            margin-bottom: 0.75rem;
        }
        .best-model-box {
            border: 1px solid #dbe3ef;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            padding: 16px 18px;
            background: #f8fbff;
            margin: 0.25rem 0 1rem 0;
        }
        .best-model-box .label {
            color: #475569;
            font-size: 0.86rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .best-model-box .name {
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 750;
            line-height: 1.2;
        }
        .best-model-box .detail {
            color: #475569;
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_results() -> pd.DataFrame:
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["checkpoint_path"] = df["checkpoint"].apply(lambda p: str(ROOT / p))
    df["history_path"] = df.get("history", pd.Series(index=df.index, dtype=str)).apply(
        lambda p: str(ROOT / p) if isinstance(p, str) and p else ""
    )
    df["confusion_matrix_path"] = df.get("confusion_matrix", pd.Series(index=df.index, dtype=str)).apply(
        lambda p: str(ROOT / p) if isinstance(p, str) and p else ""
    )
    return df


@st.cache_resource(show_spinner="Loading trained models...")
def load_models(model_rows: tuple[tuple[str, str, str], ...], device_name: str):
    device = torch.device(device_name)
    models = {}
    for model_name, display_name, checkpoint_path in model_rows:
        models[model_name] = {
            "display_name": display_name,
            "model": load_checkpoint_model(model_name, checkpoint_path, device),
        }
    return models


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_best_model_banner(best_row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="best-model-box">
            <div class="label">Best Trained Model</div>
            <div class="name">{best_row["display_name"]}</div>
            <div class="detail">
                Selected by Test F1 Macro: <strong>{format_percent(float(best_row["test_f1_macro"]))}</strong>
                &nbsp; | &nbsp;
                Test Accuracy: <strong>{format_percent(float(best_row["test_acc"]))}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_training_results(results_df: pd.DataFrame) -> None:
    best_row = results_df.sort_values("test_f1_macro", ascending=False).iloc[0]

    render_best_model_banner(best_row)

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Models", len(results_df))
    metric_b.metric("Best Test F1 Macro", format_percent(float(best_row["test_f1_macro"])))
    metric_c.metric("Best Test Accuracy", format_percent(float(best_row["test_acc"])))
    metric_d.metric("Classes", len(CLASS_NAMES))

    st.subheader("Training Summary")
    st.markdown(
        '<div class="section-note">Core comparison metrics from the completed training runs.</div>',
        unsafe_allow_html=True,
    )
    summary_df = results_df[
        [
            "display_name",
            "model_type",
            "actual_epochs",
            "best_val_acc",
            "best_val_f1_macro",
            "test_acc",
            "test_f1_macro",
        ]
    ].rename(
        columns={
            "display_name": "Model",
            "model_type": "Type",
            "actual_epochs": "Epochs",
            "best_val_acc": "Best Val Acc",
            "best_val_f1_macro": "Best Val F1 Macro",
            "test_acc": "Test Acc",
            "test_f1_macro": "Test F1 Macro",
        }
    )
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Best Val Acc": st.column_config.NumberColumn(format="%.4f"),
            "Best Val F1 Macro": st.column_config.NumberColumn(format="%.4f"),
            "Test Acc": st.column_config.NumberColumn(format="%.4f"),
            "Test F1 Macro": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    chart_df = results_df.set_index("display_name")[["test_acc", "test_f1_macro"]].rename(
        columns={"test_acc": "Test Accuracy", "test_f1_macro": "Test F1 Macro"}
    )
    st.subheader("Metric Comparison")
    st.bar_chart(chart_df, use_container_width=True)

    st.subheader("Training Curves")
    selected_model = st.selectbox(
        "Model",
        options=results_df["display_name"].tolist(),
        index=int(results_df["test_f1_macro"].astype(float).idxmax()),
        key="training_detail_model",
    )
    selected_row = results_df.loc[results_df["display_name"] == selected_model].iloc[0]
    history_path = Path(selected_row["history_path"])

    if history_path.exists():
        history_df = pd.read_csv(history_path)
        metric_choice = st.radio(
            "Curve",
            options=["Loss", "Accuracy", "F1 Macro"],
            horizontal=True,
        )
        if metric_choice == "Loss":
            curve_df = history_df.set_index("epoch")[["train_loss", "val_loss"]].rename(
                columns={"train_loss": "Train Loss", "val_loss": "Val Loss"}
            )
        elif metric_choice == "Accuracy":
            curve_df = history_df.set_index("epoch")[["train_acc", "val_acc"]].rename(
                columns={"train_acc": "Train Acc", "val_acc": "Val Acc"}
            )
        else:
            curve_df = history_df.set_index("epoch")[["train_f1_macro", "val_f1_macro"]].rename(
                columns={"train_f1_macro": "Train F1 Macro", "val_f1_macro": "Val F1 Macro"}
            )
        st.line_chart(curve_df, use_container_width=True)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No history CSV found for this model.")


def render_prediction(results_df: pd.DataFrame) -> None:
    device = get_device()
    best_row = results_df.sort_values("test_f1_macro", ascending=False).iloc[0]

    st.caption(f"Inference device: {device}")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        st.info("Upload one image to compare predictions from all five trained models.")
        return

    image = Image.open(uploaded_file)
    image_col, result_col = st.columns([1, 2])
    with image_col:
        st.image(image, caption="Uploaded image", use_container_width=True)

    missing = [
        row["checkpoint"]
        for _, row in results_df.iterrows()
        if not Path(row["checkpoint_path"]).exists()
    ]
    if missing:
        st.error("Missing checkpoint files: " + ", ".join(missing))
        return

    model_rows = tuple(
        (row["model_name"], row["display_name"], row["checkpoint_path"])
        for _, row in results_df.iterrows()
    )

    with st.spinner("Running predictions with all models..."):
        models = load_models(model_rows, str(device))
        predictions = []
        for _, row in results_df.iterrows():
            model_entry = models[row["model_name"]]
            prediction = predict_image(
                model=model_entry["model"],
                image=image,
                device=device,
                model_name=row["model_name"],
                display_name=row["display_name"],
                image_size=int(row["image_size"]),
            )
            predictions.append(prediction)

    prediction_rows = []
    for prediction in predictions:
        metric_row = results_df.loc[results_df["model_name"] == prediction.model_name].iloc[0]
        prediction_rows.append(
            {
                "Model": prediction.display_name,
                "Prediction": prediction.predicted_class,
                "Confidence": prediction.confidence,
                "Test F1 Macro": float(metric_row["test_f1_macro"]),
                "Best Model": prediction.model_name == best_row["model_name"],
            }
        )
    prediction_df = pd.DataFrame(prediction_rows).sort_values("Test F1 Macro", ascending=False)
    final_prediction = prediction_df[prediction_df["Best Model"]].iloc[0]

    with result_col:
        st.subheader("Final Prediction")
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Selected Model", str(final_prediction["Model"]))
        metric_b.metric("Predicted Class", str(final_prediction["Prediction"]))
        metric_c.metric("Confidence", format_percent(float(final_prediction["Confidence"])))
        st.caption("Final prediction is selected from the best trained model by Test F1 Macro.")

        st.subheader("Model Comparison")
        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "Test F1 Macro": st.column_config.NumberColumn(format="%.4f"),
                "Best Model": st.column_config.CheckboxColumn(),
            },
        )

    st.subheader("Class Probabilities")
    selected_model = st.selectbox(
        "Model",
        options=[prediction.display_name for prediction in predictions],
        index=[prediction.model_name for prediction in predictions].index(best_row["model_name"]),
    )
    selected_prediction = next(p for p in predictions if p.display_name == selected_model)
    probability_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Probability": [selected_prediction.probabilities[class_name] for class_name in CLASS_NAMES],
        }
    ).set_index("Class")
    st.bar_chart(probability_df, use_container_width=True)


def main() -> None:
    inject_css()
    st.title("Intel Image Classification")
    st.caption("Training result dashboard and five-model image prediction comparison.")

    results_df = load_results()

    tab_results, tab_predict = st.tabs(["Training Results", "Image Prediction"])
    with tab_results:
        render_training_results(results_df)
    with tab_predict:
        render_prediction(results_df)


if __name__ == "__main__":
    main()
