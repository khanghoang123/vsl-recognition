# Dataset card

## Scope

The project uses the preliminary Olympic AI 2025 Vietnamese Sign Language
dataset for isolated-sign video classification.

- Training videos: 3,875
- Classes: 100
- Resolution: 224 × 224
- Typical clip length: 30 frames
- Internal split: 3,063 training / 812 validation videos

The raw videos, label mapping, and private competition files are intentionally
not distributed in this repository.

## Data quality audit

The recovered split has no exact filename overlap and no overlap after grouping
filename variants such as `123.mp4` and `123_1.mp4`.

A five-frame perceptual-signature audit found two cross-split groups:

1. Two training clips labelled `Chào` and one validation clip labelled
   `Chậm lại` contain the same visual sequence. This is a conflicting-label
   data issue rather than a source-ID split failure.
2. One training clip and three validation clips are solid-colour technical
   artifacts. Their identical signatures do not represent duplicated signs.

The evaluation report therefore contains both the complete legacy validation
result and an audited subset that excludes the four affected validation files.

## Known limitations

- The class distribution is imbalanced: class support ranges from 6 to 74
  videos.
- The dataset contains blank/solid technical frames.
- The split is video-level and does not claim signer-independent evaluation
  unless signer identity metadata becomes available.
- This dataset represents isolated signs, not continuous sign-language
  sentences.

## Access and usage

Obtain the dataset from the competition organizer. Verify the organizer's
terms before redistributing data, samples, or model weights trained on it.
