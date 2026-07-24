# Limitations and error analysis

## Dataset quality

The data audit found two perceptual-signature groups crossing the recovered
split. Manual frame review showed one conflicting-label duplicate and one
solid-colour artifact group. They are documented rather than silently removed.

## Domain shift

Training clips are short, centered, and visually consistent. Webcam footage
can differ in lighting, background, framing, motion blur, and gesture timing.
High validation accuracy therefore does not guarantee equally strong webcam
performance.

## Isolated-sign assumption

The classifier expects one completed sign per clip. It does not locate sign
boundaries in continuous video and cannot model grammatical context between
signs.

## Confidence

Softmax probability is not a calibrated measure of correctness. Applications
should treat low-confidence predictions as uncertain and validate thresholds
on data captured in the target environment.

## Evaluation scope

The published result is a single recovered run on an internal held-out split.
A stronger research evaluation would add signer-independent splits, multiple
seeds, an architecture baseline, and calibration analysis.
