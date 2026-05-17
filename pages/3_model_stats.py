"""Model Statistics page - training history, confusion matrix, metrics."""

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_model_path

st.set_page_config(page_title="Thống kê Model", page_icon="📊", layout="wide")
st.title("📊 Thống kê Model")


def load_history(model_type: str):
    """Load training history."""
    path = get_model_path(f"{model_type}_history.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_labels(model_type: str):
    """Load model labels."""
    path = get_model_path(f"{model_type}_labels.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_training_history(history: dict, title: str):
    """Plot training history (loss and accuracy)."""
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.get("loss", []),
            name="Train Loss",
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            y=history.get("val_loss", []),
            name="Val Loss",
            mode="lines",
        ))
        fig.update_layout(
            title=f"{title} - Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.get("accuracy", []),
            name="Train Accuracy",
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            y=history.get("val_accuracy", []),
            name="Val Accuracy",
            mode="lines",
        ))
        fig.update_layout(
            title=f"{title} - Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


# Tabs for different models
tab1, tab2 = st.tabs(["Static Model (CNN-1D)", "Dynamic Model (Bi-LSTM + Attention)"])

with tab1:
    st.subheader("CNN-1D - Nhận diện ký hiệu tĩnh (Bảng chữ cái)")

    static_history = load_history("static")
    static_labels = load_labels("static")

    if static_history:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            best_val_acc = max(static_history.get("val_accuracy", [0]))
            st.metric("Best Val Accuracy", f"{best_val_acc:.2%}")
        with col2:
            final_loss = static_history.get("val_loss", [0])[-1]
            st.metric("Final Val Loss", f"{final_loss:.4f}")
        with col3:
            total_epochs = len(static_history.get("loss", []))
            st.metric("Total Epochs", total_epochs)

        plot_training_history(static_history, "Static CNN-1D")

        if static_labels:
            st.markdown("### Classes")
            label_text = ", ".join([f"**{v}**" for v in static_labels.values()])
            st.markdown(f"Tổng: {len(static_labels)} classes: {label_text}")
    else:
        st.info("Chưa có dữ liệu training. Train model trước:\n\n"
                "`python training/train_static.py --data_dir data/processed/static`")

with tab2:
    st.subheader("Bi-LSTM + Attention - Nhận diện ký hiệu động (Từ/Cụm từ)")

    dynamic_history = load_history("dynamic")
    dynamic_labels = load_labels("dynamic")

    if dynamic_history:
        col1, col2, col3 = st.columns(3)
        with col1:
            best_val_acc = max(dynamic_history.get("val_accuracy", [0]))
            st.metric("Best Val Accuracy", f"{best_val_acc:.2%}")
        with col2:
            final_loss = dynamic_history.get("val_loss", [0])[-1]
            st.metric("Final Val Loss", f"{final_loss:.4f}")
        with col3:
            total_epochs = len(dynamic_history.get("loss", []))
            st.metric("Total Epochs", total_epochs)

        plot_training_history(dynamic_history, "Dynamic Bi-LSTM + Attention")

        if dynamic_labels:
            st.markdown("### Classes")
            label_text = ", ".join([f"**{v}**" for v in dynamic_labels.values()])
            st.markdown(f"Tổng: {len(dynamic_labels)} classes: {label_text}")
    else:
        st.info("Chưa có dữ liệu training. Train model trước:\n\n"
                "`python training/train_dynamic.py --data_dir data/processed/dynamic`")

# Model architecture info
st.markdown("---")
st.markdown("""
### Kiến trúc Model

**Static (CNN-1D):**
```
Input(126,1) → Conv1D(64) → BN → ReLU → Conv1D(128) → BN → ReLU → MaxPool
→ Conv1D(256) → BN → ReLU → MaxPool → GlobalAvgPool → Dense(128) → Dropout(0.3)
→ Dense(num_classes, softmax)
```

**Dynamic (Bi-LSTM + Attention):**
```
Input(30,252) → Bi-LSTM(128) → Bi-LSTM(64) → MultiHeadAttention(4 heads)
→ GlobalAvgPool → BN → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2)
→ Dense(num_classes, softmax)
```
""")
