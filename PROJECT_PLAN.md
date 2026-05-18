# VSL Recognition — Plan Chi Tiet Du An

## Tong Quan

Nhan dien Ngon ngu Ky hieu Tieng Viet (Vietnamese Sign Language Recognition) su dung **VideoMAEv2-Small** fine-tune tren **Multi-VSL dataset (WACV 2025)**.

- **Model chinh**: VideoMAEv2-Small (distilled, 22M params)
- **Dataset**: Multi-VSL (WACV 2025) — 50 classes, frontal view
- **Training**: Google Colab / Kaggle (T4 16GB GPU)
- **Inference**: Local GPU 4GB VRAM (~100-150ms/prediction)
- **Deploy**: Streamlit webcam demo (local only)

---

## Pipeline Tong The

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING (Colab/Kaggle T4)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Multi-VSL Dataset (Google Drive)                                   │
│  └── 50 classes x ~28 videos/class x frontal view only             │
│      = ~1400 videos (.avi)                                          │
│                                                                     │
│  Video Preprocessing:                                               │
│  ┌──────────┐   ┌───────────────┐   ┌─────────────┐               │
│  │ Load .avi │ > │ Uniform Sample│ > │ Resize+Crop │ > Tensor      │
│  │ (decord)  │   │ 16 frames     │   │ 224x224     │  (16,3,224,224)│
│  └──────────┘   └───────────────┘   └─────────────┘               │
│                                                                     │
│  Augmentation:                                                      │
│  - Random Resized Crop (scale 0.8-1.0)                             │
│  - Color Jitter (nhe, +-10%)                                       │
│  - KHONG Horizontal Flip (tay trai != tay phai)                    │
│  - Normalize (ImageNet mean/std)                                    │
│                                                                     │
│  Model: VideoMAEv2-Small (pre-trained Kinetics)                    │
│  - ViT-Small/14, 22M params                                       │
│  - Classification head (-> 50 classes)                             │
│  - Fine-tune toan bo model                                         │
│                                                                     │
│  Training config:                                                   │
│  - Optimizer: AdamW (lr=5e-4, weight_decay=0.05)                   │
│  - Scheduler: Cosine with warmup (5 epochs)                        │
│  - Epochs: 30                                                       │
│  - Batch size: 8 (fp16 mixed precision)                            │
│  - Loss: CrossEntropy + Label Smoothing (0.1)                      │
│  - Split: 80/20 stratified                                          │
│                                                                     │
│  Output: model saved to Google Drive                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE (Local, GPU 4GB)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Streamlit Webcam Demo                                              │
│  ┌──────────┐   ┌───────────────┐   ┌─────────────┐   ┌────────┐ │
│  │ Capture   │ > │ Buffer last   │ > │ Resize+Crop │ > │VideoMAE│ │
│  │ frames    │   │ 16 frames     │   │ 224x224     │   │v2-Small│ │
│  │ (30 FPS)  │   │ (~0.5 giay)   │   │ Normalize   │   │        │ │
│  └──────────┘   └───────────────┘   └─────────────┘   └───┬────┘ │
│                                                             │      │
│                                                    ┌────────▼───┐  │
│                                                    │ Prediction │  │
│                                                    │ "Xin chao" │  │
│                                                    │ Conf: 92%  │  │
│                                                    └────────────┘  │
│  - Khong can blur mat                                              │
│  - 1 gesture = 1 inference                                         │
│  - Moi 16 frames predict 1 lan                                    │
│  - VRAM: ~1.5-2 GB                                                 │
│  - Latency: ~100-150ms/prediction                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Cau Truc Project

```
vsl-recognition/
├── notebooks/
│   ├── 01_download_and_explore.ipynb   # Download & explore Multi-VSL
│   ├── 02_train_videomae.ipynb         # Fine-tune VideoMAEv2-Small
│   └── 03_inference_and_deploy.ipynb   # Benchmark & deploy
├── src/
│   ├── __init__.py
│   ├── dataset.py                       # Video dataset & transforms
│   ├── models.py                        # Model creation utilities
│   └── inference.py                     # Inference pipeline
├── app.py                               # Streamlit webcam demo (LOCAL)
├── models/                              # Trained models (git-ignored)
│   └── videomae_vsl_best/              # Best checkpoint
│       ├── config.json
│       ├── model.safetensors
│       └── class_names.json
├── data/                                # Dataset (git-ignored)
├── requirements.txt
├── PROJECT_PLAN.md                      # This file
└── README.md
```

---

## Buoc 1: Download & Explore Dataset

**Notebook**: `01_download_and_explore.ipynb`
**Chay tren**: Colab hoac Kaggle

### Setup tren Colab

1. Mo Google Colab: https://colab.research.google.com/
2. Upload file `01_download_and_explore.ipynb`
3. Chon Runtime > Change runtime type > **T4 GPU**
4. Chay tung cell theo thu tu

### Setup tren Kaggle

1. Mo Kaggle: https://www.kaggle.com/
2. New Notebook > Upload `01_download_and_explore.ipynb`
3. Settings > Accelerator > **GPU T4 x2**
4. Chay tung cell theo thu tu

### Notebook se lam gi

| Cell | Noi dung | Thoi gian |
|------|---------|-----------|
| Config | Set NUM_CLASSES=50, NUM_FRAMES=16, VAL_RATIO=0.2 | - |
| Setup | Detect Colab/Kaggle, mount Drive, install deps | ~1min |
| Download | Download Multi-VSL dataset tu Google Drive | ~10-30min |
| Explore | Thong ke: so class, so video/class, duration | ~1min |
| Visualize | Hien thi sample frames tu video | ~1min |
| Statistics | Plot distribution so video/class, duration | ~1min |
| Split | Tao train/val split (80/20), save JSON metadata | ~1min |
| Test Pipeline | Test video loading + transform pipeline | ~1min |

### Output

- `data/multi_vsl/` — Thu muc chua video theo class
- `data/metadata.json` — Thong tin dataset (class names, num_classes, ...)
- `data/train.json` — Danh sach video training
- `data/val.json` — Danh sach video validation

### Luu y quan trong

- **Google Drive link data**: https://drive.google.com/drive/folders/1yUU1m2hy_CjaXDDoR_6i9Y3T1XL2pD4C
- Dataset la paper WACV 2025: "Sign Language Recognition: A Large-Scale Multi-View Dataset"
- Chi dung **frontal view** (matched voi webcam inference)
- Neu download tu dong that bai, co huong dan download thu cong

---

## Buoc 2: Training VideoMAEv2-Small

**Notebook**: `02_train_videomae.ipynb`
**Chay tren**: Colab hoac Kaggle (yeu cau GPU T4)

### Hyperparameters

| Parameter | Value | Ghi chu |
|-----------|-------|---------|
| Model | MCG-NJU/videomae-small-finetuned-kinetics | Pre-trained tren Kinetics-400 |
| NUM_CLASSES | 50 | 50 classes dau tien cua Multi-VSL |
| NUM_FRAMES | 16 | Input frames cho VideoMAE |
| IMAGE_SIZE | 224 | Resolution |
| BATCH_SIZE | 8 | Vua voi T4 16GB khi dung fp16 |
| LEARNING_RATE | 5e-4 | Fine-tuning LR |
| WEIGHT_DECAY | 0.05 | AdamW regularization |
| EPOCHS | 30 | Co the tang len 50 neu can |
| WARMUP_EPOCHS | 5 | Linear warmup |
| LABEL_SMOOTHING | 0.1 | Regularization |
| FP16 | True | Mixed precision |
| NUM_WORKERS | 2 | DataLoader workers |

### Training Pipeline

1. **Load split** tu notebook 01 (train.json, val.json)
2. **Tao Dataset** voi augmentation:
   - Train: RandomResizedCrop + ColorJitter + Normalize
   - Val: Resize + CenterCrop + Normalize
3. **Load pre-trained model** tu HuggingFace
4. **Setup optimizer**: AdamW + Cosine scheduler + warmup
5. **Training loop** voi mixed precision (fp16)
6. **Save best model** theo val accuracy
7. **Plot** training curves, confusion matrix

### Output

- `models/videomae_vsl_best/` — Best model checkpoint
  - `config.json` — Model config
  - `model.safetensors` — Model weights
  - `class_names.json` — Ten cac class
- `models/training_curves.png` — Bieu do loss/accuracy
- `models/confusion_matrix.png` — Confusion matrix
- `models/training_history.json` — Lich su training

### Thoi gian uoc tinh

| GPU | Thoi gian / epoch | Tong (30 epochs) |
|-----|-------------------|------------------|
| T4 (Colab/Kaggle) | ~3-5 min | ~1.5-2.5h |
| A100 (Colab Pro) | ~1-2 min | ~30-60min |

### Luu y

- **VRAM**: ~6-8 GB tren T4 voi batch_size=8 + fp16
- **Augmentation**: KHONG dung Horizontal Flip (tay trai != tay phai trong ngon ngu ky hieu)
- **ColorJitter**: Apply cung params cho tat ca frames trong 1 video (temporal consistency)
- Neu Colab disconnect, co the resume tu checkpoint

---

## Buoc 3: Inference & Deploy

**Notebook**: `03_inference_and_deploy.ipynb`
**Chay tren**: Colab/Kaggle (de benchmark) hoac Local

### Benchmark

- Do latency FP32 va FP16
- Do VRAM usage
- Test tren sample videos

### Deploy Local

#### Yeu cau he thong

| Thanh phan | Yeu cau |
|------------|---------|
| GPU | NVIDIA GPU >= 4GB VRAM |
| Python | >= 3.9 |
| OS | Windows / Linux / macOS |
| Webcam | Bat ky webcam nao |

#### Cac buoc deploy

```bash
# 1. Clone repo
git clone https://github.com/gthgfuiss123-ship-it/vsl-recognition.git
cd vsl-recognition

# 2. Cai dependencies
pip install -r requirements.txt

# 3. Copy trained model tu Google Drive
# Download thu muc models/videomae_vsl_best/ tu Drive ve may
# Dat vao: vsl-recognition/models/videomae_vsl_best/
#   - config.json
#   - model.safetensors (hoac pytorch_model.bin)
#   - class_names.json

# 4. Chay Streamlit app
streamlit run app.py
```

#### Cach hoat dong cua app.py

1. Load model tu `models/videomae_vsl_best/`
2. Mo webcam (OpenCV)
3. Buffer 16 frames tu webcam
4. Preprocess: Resize + CenterCrop 224x224 + Normalize
5. Predict: VideoMAEv2-Small -> Top-5 predictions
6. Hien thi ket qua real-time

#### Performance local

| Metric | GPU 4GB | GPU 8GB+ |
|--------|---------|----------|
| Latency FP32 | ~100-150ms | ~80-120ms |
| Latency FP16 | ~60-100ms | ~50-80ms |
| VRAM usage | ~1.5-2 GB | ~1.5-2 GB |
| FPS | ~7-15 | ~10-20 |

---

## Cau Hinh Chi Tiet Cho Colab vs Kaggle

### Google Colab

```
Runtime: T4 GPU (free) hoac A100 (Pro)
RAM: 12-25 GB
Disk: ~100 GB
Session: Max 12h (free), 24h (Pro)
Mount: Google Drive (/content/drive/MyDrive/)
```

**Setup paths**:
```python
BASE_DIR = "/content/drive/MyDrive/vsl-recognition"
DATA_DIR = "/content/drive/MyDrive/vsl-recognition/data/multi_vsl"
MODEL_DIR = "/content/drive/MyDrive/vsl-recognition/models"
```

**Luu y**:
- Mount Google Drive de luu model + data persistent
- Neu session bi disconnect, data van con tren Drive
- Install dependencies moi session: `!pip install -q ...`

### Kaggle

```
Runtime: T4 x2 GPU
RAM: 30 GB
Disk: ~50 GB (/kaggle/working)
Session: Max 12h (GPU), 9h/week quota
```

**Setup paths**:
```python
BASE_DIR = "/kaggle/working/vsl-recognition"
DATA_DIR = "/kaggle/working/vsl-recognition/data/multi_vsl"
MODEL_DIR = "/kaggle/working/vsl-recognition/models"
```

**Luu y**:
- `/kaggle/working` bi xoa khi session ket thuc
- Luu model ra Output dataset de persistent
- Hoac upload model len Google Drive/HuggingFace Hub
- GPU quota gioi han: 30h/week

### So sanh

| Feature | Colab (Free) | Colab (Pro) | Kaggle |
|---------|-------------|-------------|--------|
| GPU | T4 16GB | A100 40GB | T4 x2 |
| RAM | 12 GB | 25-50 GB | 30 GB |
| Session | 12h max | 24h max | 12h max |
| Persistent | Google Drive | Google Drive | Output dataset |
| Gia | Free | ~$10/month | Free |
| GPU quota | Gioi han | Nhieu hon | 30h/week |

---

## Ket Qua Du Kien

| Metric | VideoMAEv2-Small |
|--------|-----------------|
| Top-1 Accuracy | ~90-95% |
| Model Size | ~90 MB |
| Parameters | ~22M |
| Inference Latency (4GB GPU) | ~100-150ms |
| VRAM Usage | ~1.5-2 GB |
| Training Time (T4) | ~1.5-2.5h |

---

## Tai Lieu Tham Khao

| Paper | Venue | Link |
|-------|-------|------|
| Multi-VSL Dataset | WACV 2025 | [paper](https://openaccess.thecvf.com/content/WACV2025/html/Dinh_Sign_Language_Recognition_A_Large-Scale_Multi-View_Dataset_and_Comprehensive_Evaluation_WACV_2025_paper.html) |
| VideoMAE | NeurIPS 2022 | [arxiv](https://arxiv.org/abs/2203.12602) |
| VideoMAE V2 | CVPR 2023 | [arxiv](https://arxiv.org/abs/2303.16727) |
| VideoMAE for SLR | PLoS One 2026 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12875579/) |

---

## Checklist Trien Khai

- [ ] Chay notebook 01: Download dataset
- [ ] Kiem tra dataset structure: `data/multi_vsl/<class_name>/<video>.avi`
- [ ] Kiem tra split files: `data/metadata.json`, `data/train.json`, `data/val.json`
- [ ] Chay notebook 02: Training
- [ ] Kiem tra model saved: `models/videomae_vsl_best/`
- [ ] Kiem tra training curves: `models/training_curves.png`
- [ ] Chay notebook 03: Benchmark (optional, tren Colab/Kaggle)
- [ ] Copy model ve may local
- [ ] `pip install -r requirements.txt` tren may local
- [ ] `streamlit run app.py` — Test webcam demo
