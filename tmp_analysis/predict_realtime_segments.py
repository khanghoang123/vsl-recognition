import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import CenterCrop, Normalize, Resize
from transformers import VideoMAEForVideoClassification


ROOT = Path("/mnt/d/Documents/hoc_ky_8/Thi_giac_may_tinh/Cuoi_ky/vsl-recognition")
MODEL_PATH = ROOT / "models" / "videomae_olympic_best"
NUM_FRAMES = 16
IMAGE_SIZE = 224


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    col_energy = gray.mean(axis=0)
    row_energy = gray.mean(axis=1)
    active_cols = np.where(col_energy > 8)[0]
    active_rows = np.where(row_energy > 8)[0]
    if active_cols.size > 0 and active_rows.size > 0:
        left = int(active_cols[0])
        right = int(active_cols[-1]) + 1
        top = int(active_rows[0])
        bottom = int(active_rows[-1]) + 1
        frame = frame[top:bottom, left:right]
        height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame
    size = min(height, width)
    top = max((height - size) // 2, 0)
    left = max((width - size) // 2, 0)
    return frame[top : top + size, left : left + size]


def read_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, {"fps": fps, "frame_count": count or len(frames), "width": width, "height": height}


def preprocess(frames):
    total = len(frames)
    if total >= NUM_FRAMES:
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    else:
        indices = np.concatenate([np.arange(total), np.full(NUM_FRAMES - total, total - 1, dtype=int)])

    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformed = []
    for idx in indices:
        frame_rgb = cv2.cvtColor(center_square_crop(frames[idx]), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = Resize(IMAGE_SIZE + 32, antialias=True)(tensor)
        tensor = CenterCrop(IMAGE_SIZE)(tensor)
        transformed.append(normalize(tensor))
    return torch.stack(transformed).unsqueeze(0)


def summarize_group(records):
    if not records:
        return {}
    return {
        "count": len(records),
        "mean_frames": float(np.mean([r["frame_count"] for r in records])),
        "mean_duration": float(np.mean([r["duration"] for r in records])),
        "mean_confidence": float(np.mean([r["confidence"] for r in records])),
        "max_confidence": float(np.max([r["confidence"] for r in records])),
        "thuc_an_count": sum("Th" in r["label"] and "c" in r["label"] and "a" in r["label"] for r in records),
        "labels": {label: sum(r["label"] == label for r in records) for label in sorted({r["label"] for r in records})},
    }


def main():
    with open(MODEL_PATH / "class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoMAEForVideoClassification.from_pretrained(str(MODEL_PATH), num_labels=len(class_names)).to(device)
    model.eval()
    if device == "cuda":
        model.half()

    all_records = []
    for folder_name in ["realtime_debug - Copy", "realtime_debug"]:
        folder = ROOT / "tmp_analysis" / folder_name
        for path in sorted(folder.glob("segment_*.mp4"), key=lambda p: p.stat().st_mtime):
            frames, meta = read_frames(path)
            if not frames:
                continue
            tensor = preprocess(frames).to(device)
            if device == "cuda":
                tensor = tensor.half()
            with torch.no_grad():
                probs = torch.softmax(model(pixel_values=tensor).logits[0].float(), dim=0)
                top_probs, top_idx = torch.topk(probs, 5)
            record = {
                "folder": folder_name,
                "name": path.name,
                "frame_count": len(frames),
                "fps": meta["fps"],
                "duration": len(frames) / meta["fps"] if meta["fps"] else 0,
                "width": meta["width"],
                "height": meta["height"],
                "label": class_names[top_idx[0].item()],
                "confidence": float(top_probs[0].item()),
                "top5": [[class_names[i.item()], float(p.item())] for p, i in zip(top_probs, top_idx)],
            }
            all_records.append(record)

    copy_names = {r["name"] for r in all_records if r["folder"] == "realtime_debug - Copy"}
    copy_records = [r for r in all_records if r["folder"] == "realtime_debug - Copy"]
    later_records = [r for r in all_records if r["folder"] == "realtime_debug" and r["name"] not in copy_names]

    output = {
        "summary": {
            "copy_run": summarize_group(copy_records),
            "later_run": summarize_group(later_records),
        },
        "copy_run": copy_records,
        "later_run": later_records,
    }
    out_path = ROOT / "tmp_analysis" / "realtime_prediction_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
