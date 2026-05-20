# VSL Recognition

Vietnamese Sign Language recognition with **VideoMAE-Small** fine-tuned on the **Olympic AI2025 preliminary dataset**.

The training workflow is Drive-first: data, metadata, checkpoints, and trained models are stored in Google Drive during Colab training. Local deployment only needs the final trained model folder.

For a guided explanation of the whole project, read [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md).
For a deeper teaching-style explanation of preprocessing, EDA, VideoMAE internals, training flow, and realtime app logic, read [PROJECT_DEEP_DIVE.md](PROJECT_DEEP_DIVE.md).

## Architecture

```text
Video/webcam segment -> 16 frames x 224x224 -> VideoMAE-Small -> 100-class prediction
```

## Training On Colab

Use the notebooks in order:

1. `notebooks/01_download_and_explore.ipynb` downloads and explores the Olympic AI2025 dataset.
2. `notebooks/02_train_videomae.ipynb` fine-tunes VideoMAE-Small and saves checkpoints/model to Drive.
3. `notebooks/03_inference_and_deploy.ipynb` benchmarks the trained model and explains local deploy.

Default Google Drive layout:

```text
/content/drive/MyDrive/vsl-recognition/
|-- data/
|   `-- olympic_ai2025/
|-- metadata/
|-- checkpoints/
|   `-- videomae_olympic/
`-- models/
    `-- videomae_olympic_best/
```

## Local Deploy

After training, download this Drive folder:

```text
/content/drive/MyDrive/vsl-recognition/models/videomae_olympic_best/
```

Place it locally as:

```text
vsl-recognition/
`-- models/
    `-- videomae_olympic_best/
```

Then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The local app does not require the dataset and does not use `/content/drive/...` paths.

## Project Structure

```text
vsl-recognition/
|-- app.py
|-- requirements.txt
|-- PROJECT_PLAN.md
|-- PROJECT_WALKTHROUGH.md
|-- PROJECT_DEEP_DIVE.md
|-- notebooks/
|   |-- 01_download_and_explore.ipynb
|   |-- 02_train_videomae.ipynb
|   `-- 03_inference_and_deploy.ipynb
|-- src/
|   |-- dataset.py
|   |-- models.py
|   `-- inference.py
`-- models/
    `-- videomae_olympic_best/   # downloaded after training, git-ignored
```

## Default Training Config

| Setting | Value |
|---|---|
| Model | `MCG-NJU/videomae-small-finetuned-kinetics` |
| Classes | 100 |
| Input | 16 frames, 224x224 |
| Optimizer | AdamW |
| LR | 5e-4 |
| Weight decay | 0.05 |
| Scheduler | cosine with warmup |
| Epochs | 30 |
| Batch size | 8 |
| Loss | CrossEntropy + label smoothing 0.1 |
| Imbalance handling | WeightedRandomSampler |
| Local inference | batch=1, fp16 on CUDA |

The defaults above should be justified from notebook 01 EDA before full training. In particular, notebook 01 prints frame-count distribution, duration/FPS, resolution distribution, class imbalance, unreadable/bad-frame checks, duplicate candidates, and head/body/tail visual samples. These checks are for explanation and inspection only; the EDA notebook does not delete or rewrite the dataset.

## References

- Olympic AI2025 preliminary Vietnamese Sign Language dataset and baseline material.
- VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.
- VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking.
