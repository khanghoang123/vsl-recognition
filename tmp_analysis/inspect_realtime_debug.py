import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


DEBUG_DIR = Path("/mnt/d/Documents/hoc_ky_8/Thi_giac_may_tinh/Cuoi_ky/vsl-recognition/tmp_analysis/realtime_debug")
OUT_DIR = Path("/mnt/d/Documents/hoc_ky_8/Thi_giac_may_tinh/Cuoi_ky/vsl-recognition/tmp_analysis/realtime_debug_review")


def inspect_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "cannot_open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps and frame_count else None

    positions = np.linspace(0, max(frame_count - 1, 0), min(6, max(frame_count, 1)), dtype=int) if frame_count else np.array([], dtype=int)
    frames = []
    brightness = []
    blur = []

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((int(pos), rgb))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    cap.release()

    if frames:
        thumb_w = 220
        thumb_h = int(thumb_w * height / max(width, 1)) if width and height else 124
        canvas = Image.new("RGB", (thumb_w * len(frames), thumb_h + 34), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, (pos, rgb) in enumerate(frames):
            img = Image.fromarray(rgb).resize((thumb_w, thumb_h))
            x = idx * thumb_w
            canvas.paste(img, (x, 0))
            draw.text((x + 8, thumb_h + 8), f"f{pos}", fill="black")
        canvas.save(OUT_DIR / f"{video_path.stem}_contact.png")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
        "sample_brightness_mean": float(np.mean(brightness)) if brightness else None,
        "sample_blur_var_mean": float(np.mean(blur)) if blur else None,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DEBUG_DIR.glob("segment_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    report = {file.name: inspect_video(file) for file in files}
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
