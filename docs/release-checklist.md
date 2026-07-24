# Portfolio release checklist

## Before publishing

- [ ] Confirm `ruff check .` and `pytest -q` pass.
- [ ] Confirm README values match `reports/metrics.json`.
- [ ] Add the existing demo video as `assets/demo.mp4`, or link to its hosted URL.
- [ ] Keep raw data, metadata with private paths, checkpoints, and model weights out of Git.
- [ ] Review competition terms before publishing a fine-tuned checkpoint.
- [ ] Add a repository social-preview image based on the demo.

## GitHub settings

Suggested description:

> VideoMAE-based recognition of 100 Vietnamese Sign Language signs with
> reproducible evaluation and a Streamlit demo.

Suggested topics:

- `computer-vision`
- `video-classification`
- `vietnamese-sign-language`
- `sign-language-recognition`
- `videomae`
- `pytorch`
- `streamlit`

The current repository is eligible for GitHub's **Leave fork network** action:
it is public, below 1 GB, and has no child forks. Back up the repository first,
then use **Settings → General → Danger Zone → Leave fork network**. Review
GitHub's warning because detaching is permanent and removes fork-network
metadata.

## Release

1. Commit the portfolio refactor with a descriptive message.
2. Push and wait for CI.
3. Verify every README link from a private browser window.
4. Create release `v1.0.0-portfolio`.
5. Attach the short demo video to the release if its publication is permitted.
