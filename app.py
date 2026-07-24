"""Streamlit demo for isolated Vietnamese Sign Language video classification."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from vsl_recognition.model import load_model_bundle, predict_batch
from vsl_recognition.preprocessing import prepare_video, read_video_rgb

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "videomae_olympic_best"
SUPPORTED_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]

st.set_page_config(
    page_title="Vietnamese Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
)


@st.cache_resource
def cached_model(model_dir: str):
    return load_model_bundle(model_dir)


def load_report() -> dict | None:
    path = PROJECT_ROOT / "reports" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_result(bundle, probabilities, latency_ms: float) -> None:
    top_indices = probabilities.argsort(descending=True)[:5].tolist()
    first = top_indices[0]
    st.success(
        f"Prediction: **{bundle.class_names[first]}** "
        f"({float(probabilities[first]):.1%})"
    )
    st.caption(f"Model inference latency: {latency_ms:.0f} ms")
    st.dataframe(
        pd.DataFrame(
            {
                "Rank": range(1, len(top_indices) + 1),
                "Label": [bundle.class_names[index] for index in top_indices],
                "Probability": [
                    f"{float(probabilities[index]):.2%}" for index in top_indices
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )


st.title("Vietnamese Sign Language Recognition")
st.caption("VideoMAE-Small fine-tuned for 100 isolated VSL signs.")

report = load_report()
if report:
    metrics = report["full_validation"]
    columns = st.columns(3)
    columns[0].metric("Validation accuracy", f"{metrics['top1_accuracy']:.2%}")
    columns[1].metric("Macro-F1", f"{metrics['macro_f1']:.2%}")
    columns[2].metric("Classes", "100")

demo_path = PROJECT_ROOT / "assets" / "demo.mp4"
if demo_path.exists():
    with st.expander("Project demo", expanded=True):
        st.video(str(demo_path))

with st.sidebar:
    st.header("Runtime")
    model_dir = st.text_input(
        "Model directory",
        value=os.getenv("VSL_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
    )
    st.caption("Expected files: config.json, model.safetensors, class_names.json")

try:
    bundle = cached_model(model_dir)
except (FileNotFoundError, ValueError, OSError) as error:
    st.warning("A local model bundle is required to run inference.")
    st.code(
        "models/videomae_olympic_best/\n"
        "├── config.json\n"
        "├── model.safetensors\n"
        "└── class_names.json"
    )
    st.error(str(error))
    st.stop()

st.info(
    "Upload a short video containing one complete sign. "
    "The app uniformly samples 16 frames and returns the top-5 classes."
)
uploaded = st.file_uploader(
    "Upload a sign video",
    type=SUPPORTED_EXTENSIONS,
    accept_multiple_files=False,
)

if uploaded is not None:
    st.video(uploaded)
    suffix = Path(uploaded.name).suffix.lower()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            file.write(uploaded.getbuffer())
            temp_path = Path(file.name)

        frames, video_info = read_video_rgb(temp_path)
        prepared = prepare_video(frames)
        started = time.perf_counter()
        probabilities = predict_batch(bundle, prepared.tensor.unsqueeze(0))[0]
        latency_ms = (time.perf_counter() - started) * 1000

        metadata_columns = st.columns(4)
        metadata_columns[0].metric("Decoded frames", video_info["frame_count"])
        metadata_columns[1].metric("Sampled frames", len(prepared.indices))
        metadata_columns[2].metric(
            "Resolution",
            f"{video_info['width']}×{video_info['height']}",
        )
        metadata_columns[3].metric(
            "FPS",
            f"{video_info['fps']:.1f}" if video_info["fps"] else "Unknown",
        )
        render_result(bundle, probabilities, latency_ms)
    except (RuntimeError, ValueError) as error:
        st.error(f"Could not process this video: {error}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

st.divider()
st.caption(
    "Research prototype for isolated-sign classification. "
    "It does not translate continuous sign-language sentences."
)

