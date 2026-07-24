import numpy as np

from vsl_recognition.evaluate import metric_summary


def test_metric_summary_for_perfect_predictions():
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.8, 0.0, 0.0, 0.0],
        ]
    )
    metrics = metric_summary([0, 1, 2], probabilities, num_classes=6)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
