"""VSL Recognition - Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt

Main Streamlit application.
"""

import streamlit as st

st.set_page_config(
    page_title="VSL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤟 VSL Recognition")
st.subheader("Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt")

st.markdown("""
Hệ thống nhận diện ngôn ngữ ký hiệu tiếng Việt (Vietnamese Sign Language) realtime 
từ webcam, sử dụng **MediaPipe** + **Deep Learning**.

### Tính năng

| Trang | Mô tả |
|-------|-------|
| **Nhận diện Realtime** | Webcam → MediaPipe → CNN-1D/Bi-LSTM → Kết quả |
| **Thu thập dữ liệu** | Ghi keypoints mới để mở rộng dataset |
| **Bảng chữ cái VSL** | Tham khảo 25 ký hiệu bảng chữ cái |
| **Thống kê Model** | Accuracy, Confusion Matrix, Training History |

### Pipeline

```
Webcam → MediaPipe Hands → Gesture Router → CNN-1D (static) / Bi-LSTM+Attention (dynamic) → Kết quả
```

### Cơ sở khoa học

- **Static signs (CNN-1D)**: Paper ĐH Bách Khoa HN (HUST, 2025) - >95% accuracy
- **Dynamic signs (Bi-LSTM + Attention)**: Paper ĐH GTVT (UTC, 2025) - 99.51% accuracy
- **Pipeline**: Validated by IEEE Scoping Review (2025), Springer Survey (2025)

---
👈 Chọn trang từ sidebar để bắt đầu.
""")

# Sidebar info
with st.sidebar:
    st.markdown("### Thông tin")
    st.info(
        "**Version**: 1.0\n\n"
        "**Models**:\n"
        "- Static: CNN-1D\n"
        "- Dynamic: Bi-LSTM + Attention\n\n"
        "**Feature**: MediaPipe Hands\n"
        "(42 landmarks × 3 coords = 126 features)"
    )
