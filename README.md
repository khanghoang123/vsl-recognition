# VSL Recognition — Vietnamese Sign Language Recognition

End-to-end Vietnamese Sign Language recognition using **VideoMAEv2-Small** fine-tuned on **Multi-VSL dataset (WACV 2025)**.

## Architecture

```
Video (16 frames × 224×224) → VideoMAEv2-Small (22M params) → [CLS] → FC → Prediction
```

- **Model**: VideoMAEv2-Small (distilled from ViT-giant, CVPR 2023)
- **Dataset**: [Multi-VSL](https://github.com/Etdihatthoc/Multi-VSL_WACV_2025) (WACV 2025) — 84,000+ videos, 1000 glosses, 30 signers
- **Training**: Google Colab / Kaggle (T4 16GB GPU)
- **Inference**: Local GPU 4GB VRAM (~100-150ms/prediction)

## Quick Start

### 1. Train (Colab/Kaggle)

Upload notebooks to Colab/Kaggle with GPU enabled, then run in order:

1. `notebooks/01_download_and_explore.ipynb` — Download Multi-VSL dataset
2. `notebooks/02_train_videomae.ipynb` — Fine-tune VideoMAEv2-Small
3. `notebooks/03_inference_and_deploy.ipynb` — Benchmark & deploy

### 2. Deploy (Local)

```bash
git clone https://github.com/king14052004-crypto/vsl-recognition.git
cd vsl-recognition
pip install -r requirements.txt

# Copy trained model from Google Drive to models/videomae_vsl_best/
# Then run:
streamlit run app.py
```

## Project Structure

```
vsl-recognition/
├── notebooks/
│   ├── 01_download_and_explore.ipynb   # Download & explore Multi-VSL
│   ├── 02_train_videomae.ipynb         # Fine-tune VideoMAEv2-Small
│   └── 03_inference_and_deploy.ipynb   # Benchmark & deploy
├── src/
│   ├── dataset.py                       # Video dataset & transforms
│   ├── models.py                        # Model creation utilities
│   └── inference.py                     # Inference pipeline
├── app.py                               # Streamlit webcam demo
├── models/                              # Trained models (git-ignored)
├── data/                                # Dataset (git-ignored)
├── requirements.txt
└── README.md
```

## Model Details

| Specification | Value |
|---|---|
| Architecture | VideoMAEv2-Small (ViT-S/14) |
| Parameters | ~22M |
| Pre-trained on | Kinetics-400 |
| Fine-tuned on | Multi-VSL (50 classes, frontal view) |
| Input | 16 frames × 224 × 224 |
| VRAM (inference) | ~1.5-2 GB |
| VRAM (training) | ~6-8 GB |
| Inference latency | ~100-150ms (GPU 4GB) |

## Training Config

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (lr=5e-4, wd=0.05) |
| Scheduler | Cosine with 5-epoch warmup |
| Epochs | 30 |
| Batch size | 8 (fp16 mixed precision) |
| Label smoothing | 0.1 |
| Augmentation | RandomResizedCrop, ColorJitter (no HFlip) |
| Split | 80/20 stratified |

## References

- **Multi-VSL Dataset**: Dinh et al., "Sign Language Recognition: A Large-Scale Multi-View Dataset and Comprehensive Evaluation", WACV 2025
- **VideoMAE**: Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training", NeurIPS 2022
- **VideoMAE V2**: Wang et al., "VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking", CVPR 2023

## Why VideoMAEv2-Small?

1. **Proven for SLR** — 96.9% accuracy on BdSLW60 (PLoS One 2026)
2. **Efficient** — 22M params, fits 4GB VRAM
3. **Distilled** — Knowledge from 1B-param teacher
4. **Self-supervised** — Data-efficient fine-tuning
5. **Better than Swin** — Faster + higher accuracy than Swin Transformer used in Multi-VSL paper

## License

MIT
