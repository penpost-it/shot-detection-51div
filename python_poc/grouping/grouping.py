"""Shot-grouping (탄착군 형성) analysis for warped target images.

The analysis core is pure Python (no numpy / cv2) so it is trivially testable
and reusable by both the CLI pipeline and the Streamlit ``front_poc``. Only
``draw_grouping`` needs cv2, which is imported lazily inside the function.

Given bullet-hole detections (bounding boxes on the warped target canvas), we
take their center points, compute the smallest circle enclosing them, and call
the group "formed" when that circle is tight enough::

    formed := minEnclosingCircle.diameter <= threshold

By default the threshold is a fraction of the warped-canvas diagonal, so it
stays comparable across reference frames of different sizes. Pass
``threshold_px`` to use an absolute pixel threshold instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

Point = Tuple[float, float]

# status values
FORMED = "formed"
NOT_FORMED = "not_formed"
INSUFFICIENT = "insufficient_shots"
NO_SHOTS = "no_shots"

_EPS = 1e-9
_CONTAINS_TOL = 1e-7


@dataclass(frozen=True)
class GroupingResult:
    """Outcome of a single shot-grouping evaluation."""

    status: str  # FORMED | NOT_FORMED | INSUFFICIENT | NO_SHOTS
    formed: Optional[bool]  # None when not judgeable (no/insufficient shots)
    n_shots: int
    threshold_px: float
    canvas_diagonal_px: float
    diameter_px: Optional[float] = None
    diameter_frac: Optional[float] = None  # diameter_px / canvas_diagonal_px
    circle_center: Optional[Point] = None
    circle_radius: Optional[float] = None
    centroid: Optional[Point] = None


def _center(detection: Any) -> Point:
    """Center of a detection bbox given as a dict, an object (x1..y2), or a sequence."""
    if isinstance(detection, dict):
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    elif all(hasattr(detection, attr) for attr in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
    else:
        x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _circle_from_two(a: Point, b: Point) -> Tuple[Point, float]:
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return center, _dist(center, a)


def _circumcircle(a: Point, b: Point, c: Point) -> Optional[Tuple[Point, float]]:
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < _EPS:
        return None  # collinear; the enclosing circle comes from the farthest pair
    a2 = a[0] * a[0] + a[1] * a[1]
    b2 = b[0] * b[0] + b[1] * b[1]
    c2 = c[0] * c[0] + c[1] * c[1]
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    center = (ux, uy)
    return center, _dist(center, a)


def min_enclosing_circle(points: Sequence[Point]) -> Tuple[Optional[Point], float]:
    """Smallest circle enclosing all points.

    Exact O(n^3) brute force over circles defined by point pairs (as a diameter)
    and point triples (circumcircle). n is the number of bullet holes per target
    (a few dozen at most), so this is plenty fast and fully deterministic.
    Returns ``(center, radius)`` with ``center=None`` only for an empty input.
    """
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    if n == 0:
        return None, 0.0
    if n == 1:
        return pts[0], 0.0

    def encloses(center: Point, radius: float) -> bool:
        return all(_dist(center, p) <= radius + _CONTAINS_TOL for p in pts)

    best_center: Optional[Point] = None
    best_radius = float("inf")

    for i in range(n):
        for j in range(i + 1, n):
            center, radius = _circle_from_two(pts[i], pts[j])
            if radius < best_radius and encloses(center, radius):
                best_center, best_radius = center, radius

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                circle = _circumcircle(pts[i], pts[j], pts[k])
                if circle is None:
                    continue
                center, radius = circle
                if radius < best_radius and encloses(center, radius):
                    best_center, best_radius = center, radius

    return best_center, best_radius


def analyze_grouping(
    detections: Sequence[Any],
    canvas_size: Tuple[int, int],
    *,
    threshold_frac: float = 0.12,
    threshold_px: Optional[float] = None,
    min_shots: int = 2,
) -> GroupingResult:
    """Decide whether the detected shots form a group.

    Args:
        detections: bullet-hole detections (dicts or objects with x1,y1,x2,y2).
        canvas_size: ``(width, height)`` of the warped image the detections live in.
        threshold_frac: group "diameter" limit as a fraction of the canvas diagonal.
        threshold_px: absolute pixel limit; overrides ``threshold_frac`` when given.
        min_shots: minimum number of shots required to render a formed/not-formed verdict.
    """
    width, height = canvas_size
    diagonal = math.hypot(float(width), float(height))
    resolved_threshold = (
        float(threshold_px) if threshold_px is not None else threshold_frac * diagonal
    )

    centers = [_center(d) for d in detections]
    n = len(centers)

    if n == 0:
        return GroupingResult(NO_SHOTS, None, 0, resolved_threshold, diagonal)

    centroid = (sum(p[0] for p in centers) / n, sum(p[1] for p in centers) / n)

    if n < min_shots:
        return GroupingResult(
            INSUFFICIENT, None, n, resolved_threshold, diagonal, centroid=centroid
        )

    center, radius = min_enclosing_circle(centers)
    diameter = 2.0 * radius
    diameter_frac = diameter / diagonal if diagonal > _EPS else 0.0
    formed = diameter <= resolved_threshold

    return GroupingResult(
        status=FORMED if formed else NOT_FORMED,
        formed=formed,
        n_shots=n,
        threshold_px=resolved_threshold,
        canvas_diagonal_px=diagonal,
        diameter_px=diameter,
        diameter_frac=diameter_frac,
        circle_center=center,
        circle_radius=radius,
        centroid=centroid,
    )


# --- Korean status labels for UI presentation ------------------------------

STATUS_LABELS_KO = {
    FORMED: "탄착군 형성",
    NOT_FORMED: "탄착군 미형성",
    INSUFFICIENT: "판정 불가 (탄착 수 부족)",
    NO_SHOTS: "판정 불가 (탄착 없음)",
}

# cv2.putText 는 한글 글리프를 그리지 못해 "????" 로 깨진다. 임의의 PC에서 폰트
# 의존성 없이 시연하기 위해, 이미지에 그려 넣는 판정 라벨은 영어로 표기한다.
# (프런트 화면의 한글 합/불 판정은 HTML 로 렌더되어 그대로 정상 표시된다.)
STATUS_LABELS_EN = {
    FORMED: "Grouped (PASS)",
    NOT_FORMED: "Not grouped (FAIL)",
    INSUFFICIENT: "N/A (too few shots)",
    NO_SHOTS: "N/A (no shots)",
}


def _draw_verdict_label(out: "Any", status: str, suffix: str, color) -> None:
    """판정 라벨을 좌상단에 영어로 그린다 (폰트 의존성 없이 어디서나 동작)."""
    import cv2

    text = STATUS_LABELS_EN.get(status, status) + suffix
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_grouping(
    image: "Any",
    result: GroupingResult,
    detections: Optional[Sequence[Any]] = None,
    *,
    draw_boxes: bool = True,
) -> "Any":
    """Overlay bounding boxes, the enclosing circle, the centroid, and a verdict.

    Returns a new BGR image (the input is not modified). cv2 is imported lazily
    so importing this module never requires OpenCV.
    """
    import cv2  # lazy: only needed when actually rendering

    out = image.copy()
    if result.formed is True:
        color = (0, 200, 0)  # green
    elif result.formed is False:
        color = (0, 0, 230)  # red
    else:
        color = (180, 180, 180)  # gray (not judgeable)

    if draw_boxes and detections:
        for d in detections:
            cx, cy = _center(d)
            x1, y1, x2, y2 = _bbox(d)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
            cv2.circle(out, (int(round(cx)), int(round(cy))), 2, (0, 200, 255), -1)

    if result.circle_center is not None and result.circle_radius:
        c = (int(round(result.circle_center[0])), int(round(result.circle_center[1])))
        cv2.circle(out, c, max(int(round(result.circle_radius)), 1), color, 2)

    if result.centroid is not None:
        ctr = (int(round(result.centroid[0])), int(round(result.centroid[1])))
        cv2.drawMarker(out, ctr, color, cv2.MARKER_CROSS, 18, 2)

    suffix = ""
    if result.diameter_px is not None:
        suffix = f"  d={result.diameter_px:.0f}px ({result.diameter_frac * 100:.1f}%)"
    _draw_verdict_label(out, result.status, suffix, color)
    return out


def _bbox(detection: Any) -> Tuple[float, float, float, float]:
    if isinstance(detection, dict):
        return (
            float(detection["x1"]),
            float(detection["y1"]),
            float(detection["x2"]),
            float(detection["y2"]),
        )
    if all(hasattr(detection, attr) for attr in ("x1", "y1", "x2", "y2")):
        return (
            float(detection.x1),
            float(detection.y1),
            float(detection.x2),
            float(detection.y2),
        )
    return (float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3]))
