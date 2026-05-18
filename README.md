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
- **Jupyter Notebooks**: Training/fine-tuning — chạy được trên **Google Colab**, **Kaggle**, hoặc **Local**
- **Model stats**: Visualize training history, accuracy, confusion matrix

## Cơ sở khoa học

| Paper | Tác giả | Accuracy |
|-------|---------|----------|
| "Recognizing VSL Using Deep Neural Networks" | ĐH Giao thông Vận tải, 2025 | 99.51% |
| "A Unified Hand-Landmark-Based DL Framework" | ĐH Bình Dương, 2026 | High |
| "VSL Alphabet Recognition Using DL and MediaPipe" | ĐH Bách Khoa HN, 2025 | >95% |

## Cài đặt (Local)

```bash
git clone https://github.com/king14052004-crypto/vsl-recognition.git
cd vsl-recognition
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Training

### Cách 1: Google Colab (Khuyến nghị - có GPU miễn phí)

1. Upload 3 notebooks từ `notebooks/` lên Google Colab (hoặc mở trực tiếp từ GitHub)
2. Chạy lần lượt:
   - `01_data_preparation.ipynb` → Download dataset
   - `02_train_static_model.ipynb` → Train CNN-1D
   - `03_train_dynamic_model.ipynb` → Train Bi-LSTM + Attention
3. Notebooks **tự động mount Google Drive** → data & models lưu tại `My Drive/vsl-recognition/`
4. Sau khi train xong, download thư mục `models/` từ Drive về local để deploy

> **Lưu ý Colab**: Vào **Runtime → Change runtime type → GPU** (T4) để train nhanh hơn.

### Cách 2: Kaggle (có GPU P100 miễn phí)

1. Tạo notebook mới trên Kaggle
2. Upload hoặc copy nội dung từ notebooks
3. Bật **GPU** trong Settings → Accelerator
4. Notebooks tự detect Kaggle và lưu output tại `/kaggle/working/vsl-recognition/`
5. Sau khi train xong, download models từ Output tab

> **Lưu ý Kaggle**: Session tối đa 12 giờ. Data lưu trong `/kaggle/working/` sẽ mất khi session kết thúc → nên download models ngay sau khi train.

### Cách 3: Local

```bash
jupyter notebook notebooks/
```

### Notebooks

| Notebook | Mô tả |
|----------|-------|
| `01_data_preparation.ipynb` | Download & xử lý VOYA_VSL dataset từ HuggingFace |
| `02_train_static_model.ipynb` | Train CNN-1D cho static signs (chữ cái) |
| `03_train_dynamic_model.ipynb` | Train Bi-LSTM + Attention cho dynamic signs (từ/cụm từ) |

Mỗi notebook có **cấu hình ở đầu** (số epochs, batch size, số classes,...) và **hiển thị log, biểu đồ, confusion matrix** trực tiếp.

### Sau khi train xong

Copy thư mục `models/` (từ Drive hoặc Kaggle output) vào thư mục `models/` của project local:

```
models/
├── static_cnn1d.keras
├── static_labels.json
├── static_history.json
├── dynamic_bilstm_att.keras
├── dynamic_labels.json
└── dynamic_history.json
```

## Deploy (Streamlit App)

```bash
streamlit run app.py
```

## Cấu trúc project

```
vsl-recognition/
├── app.py                          # Streamlit main app
├── notebooks/
│   ├── 01_data_preparation.ipynb   # Download & xử lý dataset
│   ├── 02_train_static_model.ipynb # Train CNN-1D
│   └── 03_train_dynamic_model.ipynb# Train Bi-LSTM + Attention
├── pages/
│   ├── 1_realtime.py               # Nhận diện realtime
│   ├── 2_reference.py              # Bảng chữ cái VSL
│   └── 3_model_stats.py            # Thống kê model
├── src/
│   ├── keypoint_extractor.py       # MediaPipe keypoint extraction
│   ├── gesture_router.py           # Phân loại static/dynamic
│   ├── static_classifier.py        # CNN-1D model + inference
│   ├── dynamic_classifier.py       # Bi-LSTM + Attention model + inference
│   ├── sentence_builder.py         # Ghép ký hiệu thành câu
│   ├── preprocessing.py            # Preprocessing utilities
│   └── utils.py                    # Helper functions
├── models/                         # Trained models (git-ignored)
├── data/                           # Dataset (git-ignored)
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.10+**
- **Streamlit** + streamlit-webrtc (UI)
- **MediaPipe** (hand landmark extraction)
- **TensorFlow/Keras** (deep learning)
- **OpenCV** (video processing)
- **Jupyter Notebook** (training/fine-tuning)

## Dataset

- **Multi-VSL** (WACV 2025, ĐH Bách Khoa HN) - 1000 glosses, 84000+ videos
- **QIPEDC** (Bộ GD&ĐT + World Bank) - 4000 ký hiệu chuẩn quốc gia
- **VOYA_VSL** (HuggingFace) - 161 classes, pre-extracted keypoints

## License

MIT
