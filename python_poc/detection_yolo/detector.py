"""YOLO inference adapter for bullet-hole detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_model_cache: dict[str, Any] = {}


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = "bullet_hole"


def _load_model(model_path: str | Path) -> Any:
    key = str(model_path)
    if key not in _model_cache:
        from ultralytics import YOLO  # lazy import so the module loads even without ultralytics
        _model_cache[key] = YOLO(key)
    return _model_cache[key]


def detect(
    image: np.ndarray,
    *,
    model_path: str | Path | None = None,
    confidence: float = 0.40,  # val 기준 가장 정확한 개수가 나온 값
    draw: bool = True,
    **_: object,
) -> tuple[np.ndarray, list[Detection]]:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if model_path is None:
        raise ValueError("model_path is required for YOLO detection")

    model = _load_model(model_path)
    # imgsz=960 학습값과 동일. BGR np.ndarray를 그대로 넘겨도 ultralytics가 처리함
    results = model.predict(image, imgsz=960, conf=confidence, verbose=False)

    detections: list[Detection] = []
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf))

    if draw and detections:
        out = image.copy()
        for det in detections:
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            cv2.putText(
                out, f"{det.confidence:.2f}",
                (det.x1, max(det.y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
            )
        return out, detections

    return image.copy(), detections
