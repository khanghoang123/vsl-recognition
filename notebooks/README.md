# Experiment workflow

The original exploratory notebooks were replaced by importable modules and
command-line entry points. This keeps training and evaluation logic testable
and prevents notebook output from becoming the source of truth.

Run the workflow from a Colab or Kaggle terminal after cloning the repository:

```bash
pip install -e .

python -m vsl_recognition.audit \
  --metadata-dir /path/to/metadata \
  --data-root /path/to/dataset/train \
  --model-dir /path/to/model \
  --output reports/data_audit.json

python -m vsl_recognition.train \
  --config configs/videomae_small.json \
  --metadata-dir /path/to/metadata \
  --data-root /path/to/dataset/train \
  --output-dir models/videomae_olympic_best

python -m vsl_recognition.evaluate \
  --model-dir models/videomae_olympic_best \
  --metadata /path/to/metadata/val.json \
  --data-root /path/to/dataset/train \
  --audit reports/data_audit.json \
  --output-dir reports
```

The dataset, metadata paths, and model artifacts remain outside Git.
