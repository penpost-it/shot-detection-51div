"""Single-image warping helpers for the python POC pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .warping import (
    MatchResult,
    choose_best,
    create_feature,
    create_matcher,
    enhance_gray,
    ensure_uint8_gray,
    evaluate_reference,
    list_images,
    load_reference,
    scale_for_features,
)


DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent / "top_view_reference"


@dataclass(frozen=True)
class WarpMetadata:
    best_reference: Path
    score: float
    good_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error: float
    warp_coverage: float
    output_size: tuple[int, int]


def load_top_view_references(
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
    *,
    feature: str = "sift",
    max_features: int = 3000,
    max_feature_side: int = 1000,
):
    reference_dir = Path(reference_dir)
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {reference_dir}")

    detector = create_feature(feature, max_features)
    reference_paths = list_images(reference_dir)
    if not reference_paths:
        raise RuntimeError(f"No reference images found in {reference_dir}")

    references = [load_reference(path, detector, max_feature_side) for path in reference_paths]
    references = [reference for reference in references if reference.descriptors is not None and len(reference.keypoints) >= 4]
    if not references:
        raise RuntimeError(f"No reference image produced enough features: {reference_dir}")
    return references, detector


def warp_image_with_metadata(
    image_path: str | Path,
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
    feature: str = "sift",
    max_features: int = 3000,
    ratio: float = 0.75,
    ransac_threshold: float = 5.0,
    min_good_matches: int = 10,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.12,
    min_warp_coverage: float = 0.10,
    max_feature_side: int = 1000,
) -> tuple[np.ndarray, WarpMetadata]:
    """Warp one image into the best top-view reference coordinate frame.

    The returned image is a BGR numpy array, matching OpenCV conventions.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read input image: {image_path}")

    references, detector = load_top_view_references(
        reference_dir,
        feature=feature,
        max_features=max_features,
        max_feature_side=max_feature_side,
    )
    matcher = create_matcher(feature)

    gray = ensure_uint8_gray(image)
    feature_gray, query_scale_matrix, _ = scale_for_features(enhance_gray(gray), max_feature_side)
    query_keypoints, query_desc = detector.detectAndCompute(feature_gray, None)

    match_args = argparse.Namespace(
        ratio=ratio,
        ransac_threshold=ransac_threshold,
        min_good_matches=min_good_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        min_warp_coverage=min_warp_coverage,
    )
    results: list[MatchResult] = [
        evaluate_reference(
            reference,
            query_keypoints,
            query_desc,
            query_scale_matrix,
            image.shape[:2],
            matcher,
            match_args,
        )
        for reference in references
    ]
    best = choose_best(results)
    if best is None or best.homography is None:
        best_candidate = max(results, key=lambda item: (item.score, item.inliers, item.inlier_ratio), default=None)
        if best_candidate is None:
            raise RuntimeError("No reference could be evaluated.")
        raise RuntimeError(
            "No valid homography found. "
            f"Best candidate={best_candidate.reference.path.name}, "
            f"status={best_candidate.status}, "
            f"good_matches={best_candidate.good_matches}, "
            f"inliers={best_candidate.inliers}, "
            f"inlier_ratio={best_candidate.inlier_ratio:.4f}, "
            f"coverage={best_candidate.warp_coverage:.4f}"
        )

    ref_height, ref_width = best.reference.image.shape[:2]
    warped = cv2.warpPerspective(image, best.homography, (ref_width, ref_height))
    metadata = WarpMetadata(
        best_reference=best.reference.path,
        score=best.score,
        good_matches=best.good_matches,
        inliers=best.inliers,
        inlier_ratio=best.inlier_ratio,
        reprojection_error=best.reprojection_error,
        warp_coverage=best.warp_coverage,
        output_size=(ref_width, ref_height),
    )
    return warped, metadata


def warp_image(image_path: str | Path, **kwargs) -> np.ndarray:
    warped, _ = warp_image_with_metadata(image_path, **kwargs)
    return warped
