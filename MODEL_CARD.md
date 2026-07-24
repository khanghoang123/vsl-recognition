# Model card: VideoMAE-Small for Vietnamese Sign Language

## Model

- Architecture: VideoMAE-Small
- Initialization: `MCG-NJU/videomae-small-finetuned-kinetics`
- Task: 100-class isolated Vietnamese Sign Language classification
- Input: 16 uniformly sampled RGB frames at 224 × 224
- Output: class probabilities and top-5 labels

## Training

The recovered run used AdamW, cosine decay with five warm-up epochs, label
smoothing, class-balanced sampling, mixed precision, and a 30-epoch budget.
The best saved checkpoint is epoch 27.

## Evaluation

Machine-readable results are stored in `reports/metrics.json`. The report
distinguishes the complete 812-video validation split from an audited subset
that excludes four validation artifacts identified during data-quality review.

| Evaluation set | Top-1 accuracy | Top-5 accuracy | Macro-F1 |
|---|---:|---:|---:|
| Full validation (812 videos) | 87.19% | 96.31% | 82.14% |
| Audited subset (808 videos) | 87.62% | 96.66% | 82.45% |

These values are held-out validation results, not an official competition test
score and not a multi-seed estimate.

## Intended use

The model is a research and portfolio prototype for recognizing one complete,
isolated sign per short video.

It is not designed for:

- continuous sentence translation;
- safety-critical accessibility services;
- unseen regional sign-language variants without further validation;
- commercial deployment without reviewing upstream licenses.

## License

The upstream checkpoint is CC BY-NC 4.0. Fine-tuned weights, if distributed,
must retain the relevant attribution and non-commercial restriction.
