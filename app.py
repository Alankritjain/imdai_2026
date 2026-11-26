"""Explore data, inspect preprocessing and demonstrate genuine trained steering predictions."""

import io
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import CROP_BOTTOM, CROP_TOP, DATA_DIR, DATASET_URL, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.data_loader import dataset_fingerprint, find_driving_logs, load_driving_data
from src.explore_data import analyze_dataset, steering_histogram, steering_statistics
from src.model import ARCHITECTURE, build_model
from src.preprocess import crop_image, load_image, preprocess_image
from src.pipeline import file_sha256
from src.predict import load_trained_model, predict_steering
from src.dashboard_ui import apply_style, steering_gauge, input_card

st.set_page_config(page_title="Behavioral Cloning | Steering Predictor", page_icon="🚗", layout="wide")
apply_style()
st.markdown('<div class="eyebrow">DRIVE LAB / BEHAVIORAL CLONING</div>', unsafe_allow_html=True)
st.title("Self-Driving Car Behavioral Cloning")
st.caption("Explore how a lightweight CNN turns road images into steering predictions.")


@st.cache_data(ttl=60, show_spinner="Checking image paths and steering labels...")
def read_records(csv_path, modified_time):
    """Cache metadata only; refresh clears the cache after image-file changes."""
    return load_driving_data(csv_path)


def read_saved_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@st.cache_resource(show_spinner="Loading trained CNN...")
def cached_model(model_path, model_modified, metadata_modified):
    return load_trained_model(model_path)


def show_prediction(image, reference=None, caption="Uploaded simulator image", export=True):
    """Shared upload/sample path: a metric is displayed only after real inference."""
    model_path = MODELS_DIR / "best_model.keras"
    metadata_path = MODELS_DIR / "model_metadata.json"
    try:
        model, metadata = cached_model(str(model_path), model_path.stat().st_mtime_ns,
                                      metadata_path.stat().st_mtime_ns)
        started = time.perf_counter()
        steering = predict_steering(image, model)
        elapsed_ms = (time.perf_counter() - started) * 1000
        scene, result = st.columns([1.65, 1], gap="large")
        with scene:
            st.image(image, caption=caption, width="stretch")
        with result:
            st.markdown('<div class="eyebrow">CNN OUTPUT</div>', unsafe_allow_html=True)
            st.metric("Predicted Steering", f"{steering:+.6f}")
            steering_gauge(steering)
            st.caption("Native simulator units · not physical degrees")
            if reference is not None:
                actual, error = st.columns(2)
                actual.metric("Recorded steering", f"{reference:+.4f}")
                error.metric("Absolute error", f"{abs(steering - reference):.4f}")
            st.caption(f"Preprocessing + inference: {elapsed_ms:.0f} ms on this machine")
            st.caption("Left/right labels are disabled until the dataset's sign convention is verified.")
        if export:
            with st.expander("See what the model receives"):
                settings = metadata["preprocessing"]
                st.image(preprocess_image(image, settings["crop_top"], settings["crop_bottom"]),
                         caption="RGB · cropped · 160 × 80 · normalized once", width=480)
            st.download_button("Save this prediction", json.dumps({
                "predicted_steering": steering, "recorded_steering": reference,
                "units": "native simulator units", "model_sha256": metadata["model_sha256"]}, indent=2),
                file_name="steering_prediction.json", mime="application/json")
        return steering
    except (OSError, ValueError, ImportError, KeyError) as error:
        st.error(f"Cannot predict: {error}")
        return None


with st.sidebar:
    st.markdown("### Drive lab")
    st.caption("An image-to-steering experiment")
    sidebar_metadata = read_saved_json(MODELS_DIR / "model_metadata.json")
    if sidebar_metadata.get("training_status") == "complete":
        st.success("Trained CNN ready")
        st.caption(f"{sidebar_metadata['parameter_count']:,} parameters · "
                   f"{sidebar_metadata['counts']['train']:,} training images")
    with st.expander("Dataset settings"):
        dataset_directory = Path(st.text_input("Dataset folder", str(DATA_DIR))).expanduser()
        explicit_csv = st.text_input("CSV path (optional)", help="Use for a differently named log.")
    if st.button("Refresh dataset"):
        read_records.clear()
    st.link_button("Open Kaggle dataset", DATASET_URL)
    st.divider()
    st.caption("Simulation-based steering predictor; not a complete or road-safe autonomous-driving system.")

records = None
selected_csv = None
try:
    if explicit_csv.strip():
        selected_csv = Path(explicit_csv.strip()).expanduser().resolve()
    else:
        logs = find_driving_logs(dataset_directory)
        if logs:
            trained_sources = sidebar_metadata.get("split_summary", {}).get("selected_csvs", [])
            preferred = next((i for i, path in enumerate(logs) if str(path) in trained_sources), 0)
            selected_csv = st.sidebar.selectbox("Driving log", logs, index=preferred,
                                                format_func=lambda path: str(path.relative_to(dataset_directory.resolve())))
    if selected_csv is not None:
        records = read_records(str(selected_csv), selected_csv.stat().st_mtime_ns)
        if records.empty:
            st.error("No valid records remain. Inspect the validation report below.")
            st.json(records.attrs["validation"])
            records = None
except (OSError, ValueError) as error:
    st.error(str(error))

prediction_tab, performance_tab, dataset_tab, preprocessing_tab, overview, model_tab = st.tabs(
    ["Steering prediction", "Model performance", "Dataset overview",
     "Image & preprocessing explorer", "Project overview", "About the project"])

fingerprint = dataset_fingerprint(records) if records is not None else None
analysis_summary = read_saved_json(REPORTS_DIR / "dataset_summary.json")
split_summary = read_saved_json(PROCESSED_DIR / "split_summary.json")
analysis_complete = bool(fingerprint and analysis_summary.get("dataset_fingerprint") == fingerprint
                         and all((REPORTS_DIR / "figures" / name).exists() for name in
                                 ["steering_distribution.png", "sample_images.png", "preprocessing_comparison.png"]))
splitting_complete = bool(fingerprint and split_summary.get("dataset_fingerprint") == fingerprint
                          and all((PROCESSED_DIR / f"{name}.csv").exists() for name in ["train", "val", "test"]))

with overview:
    st.header("Learn steering from human driving")
    st.write("Behavioral cloning learns from examples of a person's driving. Each center-camera "
             "road image is paired with the steering command recorded by the simulator. "
             "The CNN predicts one continuous steering value: a supervised regression task.")
    st.markdown(f"Dataset: [zaynena/selfdriving-car-simulator]({DATASET_URL}). Only the center camera is used.")
    st.write("The objective is to learn an image-to-steering mapping from human demonstrations and "
             "compare it with a constant training-mean baseline on held-out records.")
    st.code("Road Image → Image Preprocessing → Trained CNN → Predicted Steering Value → Steering Visualization")
    st.caption("Python · TensorFlow/Keras · NumPy · Pandas · Pillow · Matplotlib · Streamlit")
    st.code(ARCHITECTURE)
    if selected_csv:
        st.caption(f"Selected log: {selected_csv}")
        provenance = read_saved_json(selected_csv.parent / "provenance.json")
        if provenance:
            st.warning(f"Lab subset: {provenance.get('subset_record_count', 'unknown')} records. "
                       "These results do not describe the complete Kaggle dataset.")
            with st.expander("Dataset provenance"):
                st.json(provenance)
    if records is None:
        st.warning("Select a local driving log to explore actual data.")
        st.markdown("1. Download and extract the Kaggle dataset into `data/raw`.\n"
                    "2. Keep each log beside its own `IMG` folder.\n"
                    "3. Select one log in the sidebar, then refresh.\n\n"
                    "You can also download the genuine 300-record lab subset:")
        st.code("python tools/download_demo_subset.py", language="powershell")

with dataset_tab:
    st.header("Dataset overview")
    if records is None:
        st.info("Statistics will appear after a valid dataset is selected.")
    else:
        validation = records.attrs["validation"]
        statistics = steering_statistics(records)
        cards = st.columns(4)
        cards[0].metric("CSV records", validation["total_records"])
        cards[1].metric("Valid records", len(records))
        cards[2].metric("Invalid records", validation["invalid_records"])
        cards[3].metric("Missing images", validation["missing_images"])
        cards = st.columns(3)
        cards[0].metric("Steering range", f"{statistics['minimum']:.3f} to {statistics['maximum']:.3f}")
        cards[1].metric("Mean steering", f"{statistics['mean']:.4f}")
        cards[2].metric("Near-zero steering", f"{statistics['near_zero_percent']:.1f}%")
        st.caption("Near-zero means |steering| < 0.05 in simulator units, not a universal driving category. "
                   "All statistics refer only to valid rows in this selected log.")
        figure = steering_histogram(records)
        st.pyplot(figure)
        plt.close(figure)
        with st.expander("Validation details"):
            st.json(validation)
        if st.button("Save analysis report and figures"):
            try:
                with st.spinner("Writing dataset analysis..."):
                    analyze_dataset(selected_csv)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(str(error))
        if analysis_complete:
            with st.expander("Saved image information and report"):
                st.json(analysis_summary)
        sample_rows = records.iloc[np.linspace(0, len(records) - 1, min(3, len(records)), dtype=int)]
        for column, row in zip(st.columns(3), sample_rows.itertuples(index=False)):
            try:
                column.image(load_image(row.image_path), caption=f"Recorded steering: {row.steering:.4f}")
            except (OSError, ValueError) as error:
                column.warning(str(error))

with preprocessing_tab:
    st.header("Explore an actual road image")
    if records is None:
        st.info("Add simulator images to compare original, cropped and processed views.")
    else:
        sample_index = st.number_input("Valid sample index (zero-based)", min_value=0,
                                       max_value=len(records) - 1, value=0, step=1)
        record = records.iloc[int(sample_index)]
        st.metric("Ground-truth steering label", f"{record['steering']:.5f}")
        st.caption("This is the simulator's recorded label, not a model prediction.")
        st.caption(f"{Path(record['image_path']).name} | Source row: {record['frame_index']}")
        try:
            original = load_image(record["image_path"])
            cropped = crop_image(original)
            processed = preprocess_image(original)
            columns = st.columns(3)
            columns[0].image(original, caption=f"1. Original RGB: {original.width} x {original.height}")
            columns[1].image(cropped, caption=f"2. Crop sky ({CROP_TOP:.0%}) and dashboard ({CROP_BOTTOM:.0%})")
            columns[2].image(processed, caption="3. Resize to 160 x 80; normalize to [0, 1]")
            st.code(f"Processed shape: {processed.shape}\nData type: {processed.dtype}\n"
                    f"Pixel range: [{processed.min():.3f}, {processed.max():.3f}]")
            st.write("Cropping keeps more of the road, resizing reduces computation, and normalization "
                     "puts pixel values on a consistent scale. Crop settings need checking for each camera view.")
        except (OSError, ValueError) as error:
            st.error(f"Cannot display this image: {error}. Refresh the dataset after fixing the image.")

with prediction_tab:
    st.header("Take a closer look at the road")
    st.write("Try a built-in image, upload your own simulator frame, or replay a recorded sequence.")
    if not (MODELS_DIR / "best_model.keras").is_file():
        st.warning("Trained model not found. Please complete model training before using steering prediction.")
        st.code("python -m src.train")
    else:
        source = st.radio("Choose input", ["Built-in examples", "Upload image", "Recorded replay"],
                          horizontal=True, key="prediction_source")
        # Use matching held-out results for demo selection, but recompute every displayed prediction.
        demo_records = records
        demo_scope = "Selected dataset (may include training images)"
        saved_evaluation = read_saved_json(REPORTS_DIR / "evaluation_metrics.json")
        try:
            prediction_path = REPORTS_DIR / "predictions.csv"
            if (saved_evaluation and saved_evaluation.get("model_sha256") == sidebar_metadata.get("model_sha256")
                    and prediction_path.is_file()
                    and file_sha256(prediction_path) == saved_evaluation.get("predictions_sha256")):
                demo_records = pd.read_csv(prediction_path).rename(columns={"actual_steering": "steering"})
                demo_scope = "Held-out test images · never used for fitting"
        except (OSError, ValueError):
            pass
        if source == "Upload image":
            input_card("Use a simulator road image", "JPG, JPEG or PNG from the center camera. "
                       "No upload is needed when you choose Built-in examples.")
            uploaded_image = st.file_uploader("Upload JPG, JPEG or PNG", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                try:
                    with Image.open(uploaded_image) as opened:
                        show_prediction(opened.convert("RGB"), caption=uploaded_image.name)
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    st.error(f"Invalid image: {error}")
        elif demo_records is None or demo_records.empty:
            st.info("Select a local dataset to use examples or replay, or upload an image.")
        elif source == "Built-in examples":
            st.caption(demo_scope)
            demo_index = st.select_slider("Example frame", options=list(range(len(demo_records))),
                format_func=lambda index: f"{index + 1:03d}", key="example_frame")
            demo_row = demo_records.iloc[demo_index]
            try:
                show_prediction(load_image(demo_row["image_path"]), float(demo_row["steering"]),
                                f"Frame {demo_index + 1:03d} / {len(demo_records)} · {demo_scope}")
            except (OSError, ValueError) as error:
                st.error(f"Example unavailable: {error}. Select another local dataset or upload an image.")
        else:
            st.caption(demo_scope)
            st.info("Recorded image playback with fresh CNN predictions. This does not control a simulated car.")
            controls = st.columns([1, 2, 2])
            playing = controls[0].toggle("Play recording", key="replay_playing")
            interval = controls[1].select_slider("Preview speed", options=[0.25, 0.5, 1.0, 2.0], value=0.5,
                                               format_func=lambda value: f"{1/value:g} frames/s")
            start = controls[2].number_input("Start at frame", min_value=1, max_value=len(demo_records), value=1)
            replay_identity = (str(MODELS_DIR), len(demo_records), start)
            if st.session_state.get("replay_identity") != replay_identity:
                st.session_state.replay_identity = replay_identity
                st.session_state.replay_index = int(start) - 1
                st.session_state.replay_history = []

            @st.fragment(run_every=interval if playing else None)
            def replay_frame():
                index = st.session_state.replay_index
                row = demo_records.iloc[index]
                try:
                    st.caption(f"Recorded frame {index + 1} of {len(demo_records)}")
                    prediction = show_prediction(load_image(row["image_path"]), float(row["steering"]),
                                                 f"Recorded frame {index + 1} · sampled sequence", export=False)
                    if prediction is not None:
                        history = st.session_state.replay_history
                        if not history or history[-1]["Frame"] != index + 1:
                            history.append({"Frame": index + 1, "CNN": prediction, "Recorded": float(row["steering"])})
                        st.session_state.replay_history = history[-40:]
                        st.line_chart(pd.DataFrame(history[-40:]).set_index("Frame"),
                                      color=["#2dd4bf", "#8098cb"], height=180)
                    if playing:
                        st.session_state.replay_index = (index + 1) % len(demo_records)
                except (OSError, ValueError) as error:
                    st.error(f"Replay image unavailable: {error}")

            replay_frame()
        st.caption("Predictions depend on the training camera and data. A numerical output is not a safety guarantee.")

with performance_tab:
    st.header("Held-out model performance")
    evaluation = read_saved_json(REPORTS_DIR / "evaluation_metrics.json")
    metadata = read_saved_json(MODELS_DIR / "model_metadata.json")
    if not evaluation:
        st.info("No evaluation results yet. Train the CNN, then run python -m src.evaluate.")
    else:
        try:
            if (evaluation["model_sha256"] != metadata["model_sha256"] or
                    file_sha256(MODELS_DIR / "best_model.keras") != evaluation["model_sha256"] or
                    file_sha256(REPORTS_DIR / "predictions.csv") != evaluation["predictions_sha256"]):
                raise ValueError("Saved results do not match the current model/predictions. Run evaluation again.")
            cards = st.columns(4)
            for card, (label, group, metric) in zip(cards, [
                    ("CNN MAE", "cnn", "mae"), ("CNN RMSE", "cnn", "rmse"),
                    ("Baseline MAE", "baseline", "mae"), ("Baseline RMSE", "baseline", "rmse")]):
                card.metric(label, f"{evaluation[group][metric]:.6f}")
            st.caption(f"{evaluation['test_records']} held-out records. Units: {evaluation['units']}. "
                       "Baseline always predicts the training-set mean.")
            comparison = "CNN improves both metrics over the baseline on this held-out recording." if all(
                evaluation["cnn"][metric] < evaluation["baseline"][metric] for metric in ("mae", "rmse")) else (
                "CNN does not improve both metrics over the baseline. This imbalanced recording "
                "does not provide strong evidence of useful driving behavior.")
            st.write(comparison)
            st.write("MAE is the average absolute error. RMSE gives larger errors more weight. Lower is better.")
            st.warning(evaluation["split_limitation"])
            st.caption("These reports belong to the trained model's saved splits, independently of the sidebar log selection.")
            if file_sha256(REPORTS_DIR / "training_history.csv") == metadata["history_sha256"]:
                history = pd.read_csv(REPORTS_DIR / "training_history.csv")
                st.line_chart(history.set_index("epoch")[["loss", "val_loss"]])
                st.caption("Training uses augmented images and dropout; validation uses original images. "
                           "Rising validation loss while training loss falls can indicate overfitting. "
                           "High losses on both may indicate underfitting; compare with the baseline.")
            for column, filename in zip(st.columns(2), ("actual_vs_predicted.png", "error_distribution.png")):
                path = REPORTS_DIR / "figures" / filename
                if path.exists():
                    column.image(str(path), width="stretch")
            examples_path = REPORTS_DIR / "figures/sample_predictions.png"
            if examples_path.exists():
                with st.expander("Inspect lower- and higher-error examples"):
                    st.image(str(examples_path))
            predictions = pd.read_csv(REPORTS_DIR / "predictions.csv")
            with st.expander("All held-out predictions"):
                st.dataframe(predictions, hide_index=True)
            st.download_button("Download held-out predictions", predictions.to_csv(index=False), "predictions.csv", "text/csv")
        except (OSError, ValueError, KeyError) as error:
            st.warning(f"Performance artifacts unavailable or inconsistent: {error}")

with model_tab:
    st.header("Lightweight CNN architecture")
    st.code(ARCHITECTURE)
    model_metadata = read_saved_json(MODELS_DIR / "model_metadata.json")
    if model_metadata.get("training_status") == "complete" and (MODELS_DIR / "best_model.keras").is_file():
        st.success(f"Saved trained CNN: {model_metadata['parameter_count']:,} parameters; "
                   f"{model_metadata['epochs_completed']} epochs completed.")
    else:
        st.info("Architecture defined. Train with python -m src.train.")
    if st.button("Inspect CNN architecture summary"):
        try:
            model = build_model()
            summary_text = io.StringIO()
            model.summary(print_fn=lambda line: summary_text.write(line + "\n"))
            st.code(summary_text.getvalue())
            st.caption("Architecture inspection constructs a fresh model; predictions use the saved trained checkpoint.")
        except (ImportError, OSError) as error:
            st.warning(str(error))
    st.write("Convolutions learn visual patterns. ReLU introduces nonlinearity. Global average pooling "
             "averages each feature map to keep the network small. The dense layer combines features, "
             "dropout reduces overfitting during training, and the linear output permits continuous steering values.")
    if splitting_complete:
        st.write(f"Split strategy: {split_summary['strategy']}")
        st.json(split_summary["counts"])
        st.caption(f"Boundary records excluded: {split_summary['excluded_boundary_records']}")
        st.warning(split_summary["limitation"])
    st.header("Limitations and future scope")
    st.write("The available data consists of excerpts from one drive with imbalanced steering. "
             "Neighboring scenes remain correlated despite boundary gaps. Future work needs longer, "
             "diverse recordings, independent sessions, and validation-guided improvements.")
    st.write("This model does not detect obstacles, recognize traffic signals, apply brakes, control "
             "vehicle speed, or guarantee safe navigation.")
