"""Drawing detections and the live count onto frames."""

from typing import List, Optional

import cv2
import numpy as np

from .detector import Detection

GREEN = (60, 200, 60)
RED = (40, 40, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw(frame: np.ndarray, detections: List[Detection], capacity: Optional[int] = None) -> np.ndarray:
    out = frame.copy()
    count = len(detections)
    over = capacity is not None and count > capacity
    box_color = RED if over else GREEN

    for d in detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), box_color, 2)
        label = f"{d.confidence:.2f}"
        cv2.putText(out, label, (d.x1, max(d.y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    banner = f"People: {count}" + (f" / {capacity}" if capacity is not None else "")
    if over:
        banner += "  OVER CAPACITY"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (10, 10), (20 + tw, 22 + th), RED if over else BLACK, -1)
    cv2.putText(out, banner, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    return out
