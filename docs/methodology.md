# Methodology

## Data preparation

Each metadata row stores a video filename, integer label, and class name.
Absolute Google Drive paths are treated as runtime details; the code resolves
files from a user-provided dataset root.

Videos are decoded to RGB and technical blank/solid frames are removed when at
least eight valid frames remain. Sixteen frames are then sampled uniformly
across the usable clip.

## Model input

Evaluation applies the same spatial pipeline used during training:

1. resize the short edge to 256 pixels;
2. center-crop to 224 × 224;
3. normalize with ImageNet mean and standard deviation;
4. stack frames as `(T, C, H, W)`.

Training uses one temporally consistent random crop and colour-jitter
parameter set for every frame in a clip.

## Training objective

VideoMAE-Small is fine-tuned with AdamW and cross-entropy with 0.1 label
smoothing. Inverse-frequency sampling reduces domination by high-support
classes. The best checkpoint is selected by validation macro-F1.

## Evaluation protocol

The primary metrics are top-1 accuracy, top-5 accuracy, and macro-F1. Macro-F1
is emphasized because class support ranges from 6 to 74 videos.

Evaluation also exports:

- per-class precision, recall, and F1;
- a normalized confusion matrix;
- predictions and top-5 probabilities for every validation video;
- model-only and end-to-end timing metadata;
- data-manifest and checkpoint hashes.
