# Project Plan: Olympic AI2025 + VideoMAE-Small

## Goal

Build a Vietnamese Sign Language recognition pipeline that balances accuracy and local latency:

```text
Olympic AI2025 videos -> 16 sampled frames -> VideoMAE-Small -> 100-class prediction
```

Training runs on Colab/Kaggle GPU. Google Drive stores data, metadata, checkpoints, and the final model. Local deployment only needs the final model folder.

## Drive-First Training Layout

```text
/content/drive/MyDrive/vsl-recognition/
├── data/
│   └── olympic_ai2025/
│       ├── dataset.zip
│       ├── train/
│       ├── public_test/
│       ├── private_test/
│       └── label_mapping.pkl
├── metadata/
│   ├── class_names.json
│   ├── train.json
│   ├── val.json
│   └── dataset_stats.json
├── checkpoints/
│   └── videomae_olympic/
└── models/
    └── videomae_olympic_best/
```

Notebook config:

```python
PROJECT_ROOT = "/content/drive/MyDrive/vsl-recognition"
DATA_DIR = f"{PROJECT_ROOT}/data/olympic_ai2025"
METADATA_DIR = f"{PROJECT_ROOT}/metadata"
CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints/videomae_olympic"
MODEL_DIR = f"{PROJECT_ROOT}/models/videomae_olympic_best"
```

## Local Deployment Layout

After training, download:

```text
/content/drive/MyDrive/vsl-recognition/models/videomae_olympic_best/
```

Place it locally as:

```text
vsl-recognition/
└── models/
    └── videomae_olympic_best/
```

The local Streamlit app loads only `models/videomae_olympic_best/`. It does not need the dataset and does not use Google Drive paths.

## Notebooks

1. `notebooks/01_download_and_explore.ipynb`
   - Mounts Google Drive on Colab.
   - Downloads Olympic AI2025 `dataset.zip`.
   - Extracts and discovers `train/`, `public_test/`, `private_test/`, and `label_mapping.pkl`.
   - Writes `train.json`, `val.json`, `class_names.json`, and `dataset_stats.json`.
   - Prints the evidence behind modeling choices: frame counts, duration/FPS, resolution/aspect ratio, class imbalance, unreadable/bad frames, duplicate candidates, and head/body/tail samples.

2. `notebooks/02_train_videomae.ipynb`
   - Reads metadata from Drive.
   - Keeps training constants visible in the config cell so they can be manually justified from notebook 01 EDA.
   - Fine-tunes `MCG-NJU/videomae-small-finetuned-kinetics`.
   - Uses 16 frames, 224x224 crops, label smoothing, AdamW, cosine warmup, fp16, and balanced sampling after checking notebook 01 EDA.
   - Saves periodic checkpoints to Drive and the best model to `models/videomae_olympic_best`.

3. `notebooks/03_inference_and_deploy.ipynb`
   - Loads the trained Drive model.
   - Benchmarks latency.
   - Tests one validation video.
   - Prints the exact local deploy folder structure.

## Default Training Config

| Setting | Value |
|---|---|
| Model | VideoMAE-Small |
| Classes | 100 |
| Frames | 16 |
| Image size | 224 |
| Batch size | 8 |
| Epochs | 30 |
| Optimizer | AdamW |
| LR | 5e-4 |
| Weight decay | 0.05 |
| Scheduler | cosine with 5 warmup epochs |
| Loss | CrossEntropy + label smoothing 0.1 |
| Imbalance | WeightedRandomSampler |
| Local inference | fp16, batch=1 |

## Validation Checklist

- Notebook 01 reports about 100 classes and roughly 3875 training videos.
- Notebook 01 shows enough EDA to explain why the project uses 16 frames, 224x224, no horizontal flip, and weighted sampling; it reports bad/duplicate candidates but does not delete data.
- Metadata files are saved under `/content/drive/MyDrive/vsl-recognition/metadata`.
- Notebook 02 can complete a smoke run over a few batches before full training.
- Best model folder contains `config.json`, `model.safetensors`, `preprocessor_config.json`, `class_names.json`, and `training_history.json`.
- Local `streamlit run app.py` loads `models/videomae_olympic_best/` without requiring Drive or dataset files.
