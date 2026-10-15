"""Runs detection over an image or a video and writes the outputs."""

import json
from pathlib import Path
from typing import Optional

import cv2

from . import annotate
from .occupancy import OccupancyTracker

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_image(detector, source: Path, out_dir: Path, capacity: Optional[int] = None) -> dict:
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    detections = detector.detect(frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{source.stem}_annotated.jpg"), annotate.draw(frame, detections, capacity))
    summary = {"source": source.name, "people_detected": len(detections)}
    if capacity is not None:
        summary["capacity"] = capacity
        summary["over_capacity"] = len(detections) > capacity
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def process_video(
    detector,
    source,
    out_dir: Path,
    capacity: Optional[int] = None,
    every_n_frames: int = 5,
    save_video: bool = True,
) -> dict:
    """Detect people in every Nth frame of a video (or webcam index).

    Sampling every few frames keeps processing fast; people rarely appear and
    vanish within a fraction of a second, so counts stay accurate.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(str(source)).stem if not isinstance(source, int) else f"camera{source}"

    writer = None
    if save_video:
        out_fps = max(fps / every_n_frames, 1.0)
        writer = cv2.VideoWriter(
            str(out_dir / f"{name}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
        )

    tracker = OccupancyTracker(capacity=capacity)
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % every_n_frames == 0:
            detections = detector.detect(frame)
            tracker.record(frame_index, frame_index / fps, len(detections))
            if writer is not None:
                writer.write(annotate.draw(frame, detections, capacity))
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    summary = {"source": name, "fps": round(fps, 2), "total_frames": frame_index, **tracker.summary()}
    tracker.save_csv(out_dir / f"{name}_occupancy.csv")
    if summary.get("frames_analyzed", 0) > 0:
        tracker.save_chart(out_dir / f"{name}_occupancy.png", title=f"Occupancy: {name}")
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS
