"""Dummy detection adapter for the python POC.

Detection implementation is intentionally left empty for now. The function
keeps the pipeline interface stable and returns the input image unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = "dummy"


def detect(image: np.ndarray, **_: object) -> tuple[np.ndarray, list[Detection]]:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    return image.copy(), []
