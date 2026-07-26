"""Shot-grouping (탄착군 형성) analysis for warped target images.

The analysis core is pure Python (no numpy / cv2) so it is trivially testable
and reusable by both the CLI pipeline and the Streamlit ``front_poc``. Only
``draw_grouping`` needs cv2, which is imported lazily inside the function.

Given bullet-hole detections (bounding boxes on the warped target canvas), we
take their center points and search for the fixed-size circle that covers the
largest number of shots. The group is "formed" when that best circle contains
at least ``min_shots`` detections::

    formed := max_points_in_circle(radius=threshold / 2) >= min_shots

By default the threshold is a circle diameter expressed as a fraction of the
warped-canvas diagonal, so it stays comparable across reference frames of
different sizes. Pass ``threshold_px`` to use an absolute pixel diameter.
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
    n_shots: int  # number of shots inside the selected fixed circle
    threshold_px: float
    canvas_diagonal_px: float
    diameter_px: Optional[float] = None
    diameter_frac: Optional[float] = None  # diameter_px / canvas_diagonal_px
    circle_center: Optional[Point] = None
    circle_radius: Optional[float] = None
    centroid: Optional[Point] = None
    total_detection_count: int = 0
    included_shot_count: int = 0
    included_indices: Tuple[int, ...] = ()


@dataclass(frozen=True)
class FixedCircleResult:
    """Best fixed-radius circle for a set of points."""

    formed: bool
    center: Optional[Point]
    radius: float
    included_indices: Tuple[int, ...]

    @property
    def shot_count(self) -> int:
        return len(self.included_indices)


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


def distance_sq(a: Point, b: Point) -> float:
    """Squared distance between two points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _mean(points: Sequence[Point]) -> Point:
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def circle_centers_from_pair(p1: Point, p2: Point, radius: float) -> list[Point]:
    """Centers of fixed-radius circles whose boundary passes through two points."""
    if radius <= 0:
        raise ValueError("radius must be greater than zero")

    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)

    if distance < _EPS:
        return []
    if distance > 2.0 * radius + _CONTAINS_TOL:
        return []

    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0
    half_distance = distance / 2.0
    height_sq = radius * radius - half_distance * half_distance
    height = math.sqrt(max(0.0, height_sq))

    perpendicular_x = -dy / distance
    perpendicular_y = dx / distance
    center1 = (
        midpoint_x + height * perpendicular_x,
        midpoint_y + height * perpendicular_y,
    )
    center2 = (
        midpoint_x - height * perpendicular_x,
        midpoint_y - height * perpendicular_y,
    )

    if distance >= 2.0 * radius - _CONTAINS_TOL:
        return [center1]
    return [center1, center2]


def points_inside_circle(points: Sequence[Point], center: Point, radius: float) -> list[int]:
    """Indices of points inside or on the boundary of a circle."""
    radius_sq = (radius + _CONTAINS_TOL) * (radius + _CONTAINS_TOL)
    return [
        index
        for index, point in enumerate(points)
        if distance_sq(point, center) <= radius_sq
    ]


def find_best_fixed_circle(
    points: Sequence[Point],
    radius: float,
    min_shots: int,
) -> FixedCircleResult:
    """Find the fixed-radius circle that contains the most points."""
    if radius <= 0:
        raise ValueError("radius must be greater than zero")
    if min_shots <= 0:
        raise ValueError("min_shots must be at least 1")

    pts = [(float(x), float(y)) for x, y in points]
    if not pts:
        return FixedCircleResult(
            formed=False,
            center=None,
            radius=radius,
            included_indices=(),
        )

    best_center: Optional[Point] = None
    best_indices: tuple[int, ...] = ()

    def consider(center: Point) -> None:
        nonlocal best_center, best_indices
        included = tuple(points_inside_circle(pts, center, radius))
        if len(included) > len(best_indices):
            best_center = center
            best_indices = included

    # Point-centered candidates cover single-shot and duplicate-coordinate cases.
    for point in pts:
        consider(point)

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for center in circle_centers_from_pair(pts[i], pts[j], radius):
                consider(center)

    return FixedCircleResult(
        formed=len(best_indices) >= min_shots,
        center=best_center,
        radius=radius,
        included_indices=best_indices,
    )


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
        threshold_frac: allowed fixed-circle diameter as a fraction of the canvas diagonal.
        threshold_px: absolute fixed-circle diameter; overrides ``threshold_frac`` when given.
        min_shots: minimum shots required inside the fixed circle.
    """
    if min_shots <= 0:
        raise ValueError("min_shots must be at least 1")

    width, height = canvas_size
    diagonal = math.hypot(float(width), float(height))
    resolved_threshold = (
        float(threshold_px) if threshold_px is not None else threshold_frac * diagonal
    )
    if resolved_threshold <= 0:
        raise ValueError("threshold diameter must be greater than zero")

    all_centers = [_center(d) for d in detections]
    total_detection_count = len(all_centers)
    if not all_centers:
        return GroupingResult(
            NO_SHOTS,
            None,
            0,
            resolved_threshold,
            diagonal,
            total_detection_count=0,
            included_shot_count=0,
        )

    radius = resolved_threshold / 2.0
    circle_result = find_best_fixed_circle(
        points=all_centers,
        radius=radius,
        min_shots=min_shots,
    )
    included_centers = [all_centers[i] for i in circle_result.included_indices]
    centroid = _mean(included_centers) if included_centers else _mean(all_centers)
    diameter = 2.0 * circle_result.radius
    diameter_frac = diameter / diagonal if diagonal > _EPS else 0.0
    if total_detection_count < min_shots:
        return GroupingResult(
            status=INSUFFICIENT,
            formed=None,
            n_shots=circle_result.shot_count,
            threshold_px=resolved_threshold,
            canvas_diagonal_px=diagonal,
            diameter_px=diameter,
            diameter_frac=diameter_frac,
            circle_center=circle_result.center,
            circle_radius=circle_result.radius,
            centroid=centroid,
            total_detection_count=total_detection_count,
            included_shot_count=circle_result.shot_count,
            included_indices=circle_result.included_indices,
        )

    return GroupingResult(
        status=FORMED if circle_result.formed else NOT_FORMED,
        formed=circle_result.formed,
        n_shots=circle_result.shot_count,
        threshold_px=resolved_threshold,
        canvas_diagonal_px=diagonal,
        diameter_px=diameter,
        diameter_frac=diameter_frac,
        circle_center=circle_result.center,
        circle_radius=circle_result.radius,
        centroid=centroid,
        total_detection_count=total_detection_count,
        included_shot_count=circle_result.shot_count,
        included_indices=circle_result.included_indices,
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
    """Overlay bounding boxes, the fixed judgment circle, the centroid, and a verdict.

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
        included_indices = set(result.included_indices)
        has_inclusion_info = result.total_detection_count > 0
        for index, d in enumerate(detections):
            cx, cy = _center(d)
            x1, y1, x2, y2 = _bbox(d)
            if has_inclusion_info and index not in included_indices:
                box_color = (145, 145, 145)
                thickness = 1
            else:
                box_color = (0, 200, 255)
                thickness = 2
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), box_color, thickness)
            cv2.circle(out, (int(round(cx)), int(round(cy))), 2, box_color, -1)

    if result.circle_center is not None and result.circle_radius:
        c = (int(round(result.circle_center[0])), int(round(result.circle_center[1])))
        cv2.circle(out, c, max(int(round(result.circle_radius)), 1), color, 2)

    if result.centroid is not None:
        ctr = (int(round(result.centroid[0])), int(round(result.centroid[1])))
        cv2.drawMarker(out, ctr, color, cv2.MARKER_CROSS, 18, 2)

    suffix = ""
    if result.diameter_px is not None:
        suffix = f"  D={result.diameter_px:.0f}px ({result.diameter_frac * 100:.1f}%)"
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
