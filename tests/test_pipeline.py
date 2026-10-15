"""Tests for counting, occupancy statistics, and the video pipeline.

They use a fake detector, so they run without downloading a YOLO model:
    python -m unittest discover tests
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from crowd.detector import Detection
from crowd.occupancy import OccupancyTracker
from crowd.pipeline import process_image, process_video


class ScriptedDetector:
    """Returns a preset number of people for each call, in order."""

    def __init__(self, counts):
        self.counts = list(counts)
        self.calls = 0

    def detect(self, frame):
        n = self.counts[self.calls % len(self.counts)]
        self.calls += 1
        return [Detection(10 + 30 * i, 10, 30 + 30 * i, 60, 0.9) for i in range(n)]


def make_video(path: Path, frames: int, fps: int = 10, size=(160, 120)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestOccupancyTracker(unittest.TestCase):
    def test_summary_stats(self):
        t = OccupancyTracker(capacity=3)
        for i, c in enumerate([1, 2, 5, 4, 2]):
            t.record(i, i * 1.0, c)
        s = t.summary()
        self.assertEqual(s["peak_count"], 5)
        self.assertEqual(s["peak_time_s"], 2.0)
        self.assertEqual(s["average_count"], 2.8)
        self.assertEqual(s["frames_over_capacity"], 2)
        self.assertEqual(s["seconds_over_capacity"], 2.0)
        self.assertEqual(s["pct_time_over_capacity"], 40.0)

    def test_smoothing(self):
        t = OccupancyTracker(smoothing_window=2)
        for i, c in enumerate([0, 4, 4]):
            t.record(i, float(i), c)
        self.assertEqual(t.to_dataframe()["smoothed"].tolist(), [0.0, 2.0, 4.0])

    def test_empty(self):
        self.assertEqual(OccupancyTracker().summary(), {"frames_analyzed": 0})


class TestPipeline(unittest.TestCase):
    def test_video_sampling_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp / "clip.mp4"
            make_video(video, frames=50, fps=10)
            detector = ScriptedDetector([1, 3, 6, 2])

            summary = process_video(detector, str(video), tmp / "out", capacity=4, every_n_frames=5)

            self.assertEqual(summary["total_frames"], 50)
            self.assertEqual(summary["frames_analyzed"], 10)  # every 5th of 50 frames
            self.assertEqual(summary["peak_count"], 6)
            for suffix in ["_annotated.mp4", "_occupancy.csv", "_occupancy.png", "_summary.json"]:
                self.assertTrue((tmp / "out" / f"clip{suffix}").exists(), suffix)
            saved = json.loads((tmp / "out" / "clip_summary.json").read_text())
            self.assertEqual(saved["peak_count"], 6)

    def test_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            img = tmp / "room.jpg"
            cv2.imwrite(str(img), np.zeros((100, 200, 3), dtype=np.uint8))
            summary = process_image(ScriptedDetector([3]), img, tmp / "out", capacity=2)
            self.assertEqual(summary["people_detected"], 3)
            self.assertTrue(summary["over_capacity"])
            self.assertTrue((tmp / "out" / "room_annotated.jpg").exists())


if __name__ == "__main__":
    unittest.main()
