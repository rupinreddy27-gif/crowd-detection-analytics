"""Command-line entry point.

Examples:
    python detect.py --source samples/street.jpg
    python detect.py --source samples/mall.mp4 --capacity 20
    python detect.py --source 0 --capacity 10          # webcam
"""

import argparse
import json
from pathlib import Path

from crowd.detector import PeopleDetector
from crowd.pipeline import is_image, process_image, process_video


def parse_args():
    p = argparse.ArgumentParser(description="Detect and count people in images or video, and track occupancy.")
    p.add_argument("--source", required=True, help="Image path, video path, or webcam index (e.g. 0)")
    p.add_argument("--out", default="outputs", help="Folder for results (default: outputs)")
    p.add_argument("--capacity", type=int, default=None, help="Optional occupancy limit to flag")
    p.add_argument("--every", type=int, default=5, help="Analyze every Nth video frame (default: 5)")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.4, help="Minimum detection confidence (default: 0.4)")
    p.add_argument("--no-video", action="store_true", help="Skip writing the annotated video")
    return p.parse_args()


def main():
    args = parse_args()
    detector = PeopleDetector(model_name=args.model, confidence=args.conf)
    out_dir = Path(args.out)

    source = int(args.source) if args.source.isdigit() else Path(args.source)
    if isinstance(source, Path) and is_image(source):
        summary = process_image(detector, source, out_dir, args.capacity)
    else:
        summary = process_video(
            detector, source if isinstance(source, int) else str(source), out_dir,
            capacity=args.capacity, every_n_frames=args.every, save_video=not args.no_video,
        )
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
