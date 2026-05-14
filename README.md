# VSL Recognition - Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt

Hệ thống nhận diện ngôn ngữ ký hiệu tiếng Việt (Vietnamese Sign Language - VSL) realtime từ webcam, sử dụng **MediaPipe** + **Deep Learning** với giao diện **Streamlit**.

## Pipeline

```
Webcam → MediaPipe Hands → Gesture Router → CNN-1D (static) / Bi-LSTM+Attention (dynamic) → Streamlit UI
```

## Tính năng

- **Nhận diện realtime** qua webcam với MediaPipe hand tracking
- **Static signs**: Bảng chữ cái VSL (25 ký hiệu) → CNN-1D
- **Dynamic signs**: Từ/cụm từ thông dụng → Bi-LSTM + Multi-Head Attention
- **Gesture Router**: Tự động phân loại static/dynamic
- **Sentence builder**: Ghép ký hiệu thành câu
- **Data collection**: Thu thập thêm dữ liệu training từ webcam
- **Model stats**: Visualize training history, accuracy, confusion matrix

## Cơ sở khoa học

| Paper | Tác giả | Accuracy |
|-------|---------|----------|
| "Recognizing VSL Using Deep Neural Networks" | ĐH Giao thông Vận tải, 2025 | 99.51% |
| "A Unified Hand-Landmark-Based DL Framework" | ĐH Bình Dương, 2026 | High |
| "VSL Alphabet Recognition Using DL and MediaPipe" | ĐH Bách Khoa HN, 2025 | >95% |

## Cài đặt

```bash
# Clone repo
git clone https://github.com/king14052004-crypto/vsl-recognition.git
cd vsl-recognition

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt
```

## Sử dụng

### Chạy app Streamlit

```bash
streamlit run app.py
```

### Thu thập dữ liệu

```bash
# Static signs (chữ cái)
python training/collect_keypoints.py --label "A" --type static --num_samples 100

# Dynamic signs (từ/cụm từ)
python training/collect_keypoints.py --label "Xin chào" --type dynamic --num_sequences 30
```

### Train models

```bash
# Train static CNN-1D
python training/train_static.py --data_dir data/processed/static --epochs 100

# Train dynamic Bi-LSTM + Attention
python training/train_dynamic.py --data_dir data/processed/dynamic --epochs 80
```

## Cấu trúc project

```
vsl-recognition/
├── app.py                        # Streamlit main app
├── pages/
│   ├── 1_realtime.py             # Nhận diện realtime
│   ├── 2_collect_data.py         # Thu thập dữ liệu
│   ├── 3_reference.py            # Bảng chữ cái VSL
│   └── 4_model_stats.py          # Thống kê model
├── src/
│   ├── keypoint_extractor.py     # MediaPipe keypoint extraction
│   ├── gesture_router.py         # Phân loại static/dynamic
│   ├── static_classifier.py      # CNN-1D model + inference
│   ├── dynamic_classifier.py     # Bi-LSTM + Attention model + inference
│   ├── sentence_builder.py       # Ghép ký hiệu thành câu
│   ├── preprocessing.py          # Preprocessing utilities
│   └── utils.py                  # Helper functions
├── training/
│   ├── train_static.py           # Train CNN-1D
│   ├── train_dynamic.py          # Train Bi-LSTM + Attention
│   └── collect_keypoints.py      # Thu thập keypoints
├── models/                       # Trained models (git-ignored)
├── data/                         # Dataset (git-ignored)
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.10+**
- **Streamlit** + streamlit-webrtc (UI)
- **MediaPipe** (hand landmark extraction)
- **TensorFlow/Keras** (deep learning)
- **OpenCV** (video processing)

## Dataset

- **Multi-VSL** (WACV 2025, ĐH Bách Khoa HN) - 1000 glosses, 84000+ videos
- **QIPEDC** (Bộ GD&ĐT + World Bank) - 4000 ký hiệu chuẩn quốc gia
- **VOYA_VSL** (HuggingFace) - 161 classes, pre-extracted keypoints

## License

MIT
