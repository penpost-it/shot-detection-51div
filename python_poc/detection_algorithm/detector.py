"""Reference-difference bullet-hole detector (classical, no ML).

Pipeline (matches main.py: the input is already warped into the reference frame):
1. Load the fixed clean reference target (warping/top_view_reference/target_reference.png).
2. ECC sub-pixel re-alignment of the (already warped) input onto the reference.
3. Black-hat morphology to isolate small dark spots.
   - Bullet holes are absolutely dark spots -> they light up in the black-hat image.
   - Ring lines are bright -> they do NOT appear in the black-hat (this is why a naive
     gray subtraction leaks ring residue but black-hat does not).
4. Subtract the reference black-hat so printed dark marks (e.g. the green "10" digits)
   are removed, leaving only holes that exist in the shot target but not the clean one.
5. Absolute-darkness gate: keep only genuinely dark spots, which drops the bright ring /
   number residue that survives sub-pixel misalignment.
6. Suppress dark printed marks (reference black-hat) and restrict to the target silhouette.
7. Threshold and filter blobs by area, circularity, fill (extent) and aspect ratio so that
   compact holes are kept while elongated ring-arc fragments are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
# The reference used for difference-based detection is fixed to this file.
REFERENCE_PATH = _PACKAGE_DIR.parent / "warping" / "top_view_reference" / "target_reference.png"

_reference_cache: dict[str, np.ndarray] = {}


def _load_reference(target_shape: tuple[int, int]) -> np.ndarray:
    key = str(REFERENCE_PATH.resolve())
    if key not in _reference_cache:
        loaded = cv2.imread(key, cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"Could not read fixed reference image: {REFERENCE_PATH}")
        _reference_cache[key] = loaded
    ref = _reference_cache[key]
    if ref.shape[:2] != target_shape:
        ref = cv2.resize(ref, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    return ref


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    cx: int = 0
    cy: int = 0
    radius: int = 0
    area: float = 0.0
    class_name: str = "bullet_hole"


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _build_silhouette(reference_bgr: np.ndarray, erode: int = 9) -> np.ndarray:
    """Mask of the target plate (green silhouette, filled) where holes can occur."""
    b, g, r = (channel.astype(np.int16) for channel in cv2.split(reference_bgr))
    greenish = ((g - r) > 12) & ((g - b) > 12)
    mask = greenish.astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(silhouette, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        silhouette = mask
    if erode > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        silhouette = cv2.erode(silhouette, erode_kernel)
    return silhouette


def _ecc_align(
    image_bgr: np.ndarray,
    image_gray: np.ndarray,
    reference_gray: np.ndarray,
    max_side: int = 720,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sub-pixel align the input onto the reference frame (affine ECC)."""
    height, width = reference_gray.shape[:2]
    long_side = max(height, width)
    scale = max_side / float(long_side) if long_side > max_side else 1.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_eq = clahe.apply(reference_gray)
    img_eq = clahe.apply(image_gray)
    if scale != 1.0:
        size = (round(width * scale), round(height * scale))
        ref_small = cv2.resize(ref_eq, size, interpolation=cv2.INTER_AREA)
        img_small = cv2.resize(img_eq, size, interpolation=cv2.INTER_AREA)
    else:
        ref_small, img_small = ref_eq, img_eq

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_small.astype(np.float32),
            img_small.astype(np.float32),
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
        if not np.isfinite(cc):
            raise cv2.error("ECC produced a non-finite correlation")
    except cv2.error:
        return image_bgr, image_gray, False

    if scale != 1.0:
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale

    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    aligned_bgr = cv2.warpAffine(image_bgr, warp_matrix, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    aligned_gray = cv2.warpAffine(image_gray, warp_matrix, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    return aligned_bgr, aligned_gray, True


def _printed_structure_mask(
    reference_gray: np.ndarray,
    reference_blackhat: np.ndarray,
    silhouette: np.ndarray,
    dark_threshold: int = 16,
    dilate: int = 3,
) -> np.ndarray:
    """Mask of *dark* printed marks (e.g. the center "10" digits) to suppress.

    Bright printed marks (ring lines, white numbers) are handled by the absolute-darkness
    gate in ``detect`` (they are not dark), so suppressing them here too would also erase
    real holes that happen to lie on a ring. We therefore only suppress dark printed marks,
    taken from the reference black-hat.
    """
    dark_printed = ((reference_blackhat >= dark_threshold).astype(np.uint8) * 255)
    dark_printed = cv2.bitwise_and(dark_printed, silhouette)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    return cv2.dilate(dark_printed, dilate_kernel)


def detect(
    image: np.ndarray,
    *,
    min_area: float = 20.0,
    max_area: float = 5000.0,
    min_circularity: float = 0.25,
    blackhat_kernel: int = 41,
    response_threshold: float = 14.0,
    abs_dark_threshold: float = 26.0,
    align: bool = True,
    suppress_structure: bool = True,
    draw: bool = True,
    **_: object,
) -> tuple[np.ndarray, list[Detection]]:
    """Find bullet holes by differencing the warped input against the fixed reference.

    The reference is always ``REFERENCE_PATH`` (target_reference.png). The input image
    is expected to already be warped into that reference's coordinate frame (as produced
    by the warping stage of the pipeline).
    """
    if image is None or image.size == 0:
        raise ValueError("image is empty")

    reference_bgr = _load_reference(image.shape[:2])
    reference_gray = _to_gray(reference_bgr)
    image_gray = _to_gray(image)

    work_bgr, work_gray = image, image_gray
    if align:
        work_bgr, work_gray, _ = _ecc_align(image, image_gray, reference_gray)

    silhouette = _build_silhouette(reference_bgr)

    kernel_size = max(3, int(blackhat_kernel) | 1)
    bh_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image_blackhat = cv2.morphologyEx(work_gray, cv2.MORPH_BLACKHAT, bh_struct)
    reference_blackhat = cv2.morphologyEx(reference_gray, cv2.MORPH_BLACKHAT, bh_struct)

    # Dark spots present in the shot target but not in the clean reference.
    response = np.clip(image_blackhat.astype(np.int16) - reference_blackhat.astype(np.int16), 0, 255).astype(np.uint8)
    response = cv2.bitwise_and(response, silhouette)

    # Absolute-darkness gate: a real hole is genuinely dark in the warped image, while
    # ring/number residue is only a green-vs-white edge (not absolutely dark). This kills
    # the printed-structure leftovers that survive the difference.
    dark_gate = (image_blackhat > float(abs_dark_threshold)).astype(np.uint8) * 255
    response = cv2.bitwise_and(response, dark_gate)

    if suppress_structure:
        structure = _printed_structure_mask(reference_gray, reference_blackhat, silhouette)
        response = cv2.bitwise_and(response, cv2.bitwise_not(structure))

    response = cv2.GaussianBlur(response, (3, 3), 0)

    _, mask = cv2.threshold(response, float(response_threshold), 255, cv2.THRESH_BINARY)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Shape gates that reject elongated ring-arc fragments while keeping compact holes.
        extent = area / float(w * h) if w > 0 and h > 0 else 0.0
        if extent < 0.32:
            continue
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_side = max(rect_w, rect_h)
        short_side = min(rect_w, rect_h)
        aspect_ratio = long_side / short_side if short_side > 0 else 999.0
        if aspect_ratio > 4.0:
            continue
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_response = float(response[blob_mask > 0].mean()) if np.any(blob_mask > 0) else 0.0
        confidence = float(np.clip(mean_response / (float(response_threshold) * 3.0), 0.0, 1.0))

        (cx, cy), enclosing_radius = cv2.minEnclosingCircle(contour)
        detections.append(
            Detection(
                x1=int(x),
                y1=int(y),
                x2=int(x + w),
                y2=int(y + h),
                confidence=confidence,
                cx=int(round(cx)),
                cy=int(round(cy)),
                radius=int(round(enclosing_radius)),
                area=area,
                class_name="bullet_hole",
            )
        )

    detections.sort(key=lambda d: (d.cy, d.cx))

    output = work_bgr.copy()
    if draw:
        for det in detections:
            draw_radius = max(det.radius + 4, 6)
            cv2.circle(output, (det.cx, det.cy), draw_radius, (0, 0, 255), 2)
        cv2.putText(
            output,
            f"holes: {len(detections)}",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output, detections
