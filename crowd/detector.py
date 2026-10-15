"""People detection with a YOLO model (Ultralytics)."""

from dataclasses import dataclass
from typing import List

import numpy as np

PERSON_CLASS_ID = 0  # "person" in the COCO dataset that YOLO models are trained on


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PeopleDetector:
    """Detects people in a single frame.

    Uses a pretrained YOLO model and keeps only the "person" class.
    The model weights (e.g. yolov8n.pt) download automatically on first use.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4, image_size: int = 640):
        from ultralytics import YOLO  # imported here so the rest of the package works without it

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]
        detections = []
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            detections.append(Detection(int(x1), int(y1), int(x2), int(y2), float(conf)))
        return detections
