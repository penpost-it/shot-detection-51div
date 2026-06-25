#!/usr/bin/env python3
"""Warp test target plates into the best top-view reference frame.

Pipeline:
1. Crop the target plate from each query image using YOLO labels when available.
2. Match the query crop against every top-view reference with SIFT or ORB.
3. Estimate query -> reference homographies with RANSAC.
4. Select the best reference using inlier count, inlier ratio, and reprojection error.
5. Save cv2.warpPerspective(query_crop, best_H, reference_size).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Reference:
    path: Path
    image: np.ndarray
    gray: np.ndarray
    feature_gray: np.ndarray
    scale_matrix: np.ndarray
    scale_matrix_inv: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None


@dataclass
class MatchResult:
    reference: Reference
    homography: np.ndarray | None
    good_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error: float
    warp_coverage: float
    score: float
    status: str


@dataclass
class CropResult:
    image: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    used_label: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warp test target plates using top-view references.")
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=Path("datasets/test/images"),
        help="Directory containing query test images.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("datasets/test/labels"),
        help="Directory containing YOLO labels for target-plate crops.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("top_view_reference"),
        help="Directory containing top-view reference images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/test/warping_results"),
        help="Directory where warped query images will be saved.",
    )
    parser.add_argument(
        "--feature",
        choices=["sift", "orb"],
        default="sift",
        help="Feature detector/descriptor used for matching.",
    )
    parser.add_argument("--max-features", type=int, default=6000, help="Maximum SIFT/ORB features per image.")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio threshold for descriptor matching.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold in feature-image pixels.",
    )
    parser.add_argument("--min-good-matches", type=int, default=10, help="Minimum Lowe-filtered matches.")
    parser.add_argument("--min-inliers", type=int, default=8, help="Minimum homography inliers.")
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.12,
        help="Minimum inlier ratio among good matches.",
    )
    parser.add_argument(
        "--min-warp-coverage",
        type=float,
        default=0.10,
        help="Minimum fraction of the reference canvas covered by the warped query crop.",
    )
    parser.add_argument(
        "--max-feature-side",
        type=int,
        default=1600,
        help="Downscale images only for feature extraction when the long side is larger than this.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.04,
        help="Extra margin around the YOLO bbox as a fraction of bbox width/height.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Ignore labels and match/warp the full query image.",
    )
    parser.add_argument(
        "--fallback",
        choices=["resize", "copy", "none"],
        default="resize",
        help="What to save when no valid homography is found.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--save-debug-crops",
        action="store_true",
        help="Also save query crops used for matching under output-dir/debug_crops.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split query images into this many deterministic shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Run only this zero-based shard index.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def ensure_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def scale_for_features(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = gray.shape[:2]
    long_side = max(width, height)
    if max_side <= 0 or long_side <= max_side:
        matrix = np.eye(3, dtype=np.float64)
        return gray, matrix, matrix

    scale = max_side / float(long_side)
    resized = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    matrix = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix_inv = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return resized, matrix, matrix_inv


def create_feature(feature: str, max_features: int):
    if feature == "sift":
        return cv2.SIFT_create(nfeatures=max_features)
    return cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)


def create_matcher(feature: str):
    if feature == "sift":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def load_reference(path: Path, detector, max_feature_side: int) -> Reference:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read reference image: {path}")
    gray = ensure_uint8_gray(image)
    feature_gray, scale_matrix, scale_matrix_inv = scale_for_features(enhance_gray(gray), max_feature_side)
    keypoints, descriptors = detector.detectAndCompute(feature_gray, None)
    return Reference(
        path=path,
        image=image,
        gray=gray,
        feature_gray=feature_gray,
        scale_matrix=scale_matrix,
        scale_matrix_inv=scale_matrix_inv,
        keypoints=keypoints,
        descriptors=descriptors,
    )


def parse_yolo_bbox(label_path: Path, width: int, height: int, margin: float) -> tuple[int, int, int, int] | None:
    if not label_path.exists():
        return None

    best_bbox = None
    best_area = -1.0
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            x_center, y_center, box_width, box_height = map(float, parts[1:5])
        except ValueError:
            continue
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            continue
        if box_width <= 0.0 or box_height <= 0.0:
            continue

        box_w_px = box_width * width
        box_h_px = box_height * height
        x1 = (x_center * width) - (box_w_px / 2.0)
        y1 = (y_center * height) - (box_h_px / 2.0)
        x2 = x1 + box_w_px
        y2 = y1 + box_h_px

        pad_x = box_w_px * margin
        pad_y = box_h_px * margin
        x1 = max(0, math.floor(x1 - pad_x))
        y1 = max(0, math.floor(y1 - pad_y))
        x2 = min(width, math.ceil(x2 + pad_x))
        y2 = min(height, math.ceil(y2 + pad_y))

        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area and x2 > x1 and y2 > y1:
            best_bbox = (x1, y1, x2, y2)
            best_area = area

    return best_bbox


def crop_query(image: np.ndarray, image_path: Path, label_dir: Path, margin: float, use_crop: bool) -> CropResult:
    height, width = image.shape[:2]
    full_bbox = (0, 0, width, height)
    if not use_crop:
        return CropResult(image=image, bbox_xyxy=full_bbox, used_label=False)

    label_path = label_dir / f"{image_path.stem}.txt"
    bbox = parse_yolo_bbox(label_path, width, height, margin)
    if bbox is None:
        return CropResult(image=image, bbox_xyxy=full_bbox, used_label=False)

    x1, y1, x2, y2 = bbox
    return CropResult(image=image[y1:y2, x1:x2].copy(), bbox_xyxy=bbox, used_label=True)


def lowe_matches(matcher, query_desc: np.ndarray | None, ref_desc: np.ndarray | None, ratio: float) -> list[cv2.DMatch]:
    if query_desc is None or ref_desc is None or len(query_desc) < 2 or len(ref_desc) < 2:
        return []
    pairs = matcher.knnMatch(query_desc, ref_desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def reprojection_error(src: np.ndarray, dst: np.ndarray, homography: np.ndarray, mask: np.ndarray) -> float:
    inlier_mask = mask.ravel().astype(bool)
    if not np.any(inlier_mask):
        return float("inf")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected[inlier_mask] - dst[inlier_mask], axis=1)
    return float(np.mean(errors)) if len(errors) else float("inf")


def warp_coverage(query_shape: tuple[int, int], reference_shape: tuple[int, int], homography: np.ndarray) -> float:
    query_height, query_width = query_shape
    ref_height, ref_width = reference_shape
    mask = np.full((query_height, query_width), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, homography, (ref_width, ref_height), flags=cv2.INTER_NEAREST)
    return float(np.count_nonzero(warped_mask)) / float(ref_width * ref_height)


def evaluate_reference(
    reference: Reference,
    query_keypoints: list[cv2.KeyPoint],
    query_desc: np.ndarray | None,
    query_scale_matrix: np.ndarray,
    query_shape: tuple[int, int],
    matcher,
    args: argparse.Namespace,
) -> MatchResult:
    good = lowe_matches(matcher, query_desc, reference.descriptors, args.ratio)
    if len(good) < args.min_good_matches:
        return MatchResult(reference, None, len(good), 0, 0.0, float("inf"), 0.0, -1.0, "not_enough_matches")

    src_points = np.float32([query_keypoints[m.queryIdx].pt for m in good])
    dst_points = np.float32([reference.keypoints[m.trainIdx].pt for m in good])
    homography_small, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, args.ransac_threshold)
    if homography_small is None or mask is None:
        return MatchResult(reference, None, len(good), 0, 0.0, float("inf"), 0.0, -1.0, "homography_failed")

    inliers = int(mask.sum())
    inlier_ratio = inliers / float(len(good)) if good else 0.0
    error = reprojection_error(src_points, dst_points, homography_small, mask)
    # Convert H from feature-image coordinates back to original crop/reference coordinates.
    homography = reference.scale_matrix_inv @ homography_small @ query_scale_matrix
    homography = homography / homography[2, 2] if abs(homography[2, 2]) > 1e-12 else homography

    reference_shape = reference.image.shape[:2]
    coverage = warp_coverage(query_shape, reference_shape, homography)

    if inliers < args.min_inliers or inlier_ratio < args.min_inlier_ratio:
        status = "weak_homography"
    elif coverage < args.min_warp_coverage:
        status = "low_warp_coverage"
    else:
        status = "ok"

    score = (inliers * inlier_ratio * math.sqrt(max(coverage, 1e-6))) / (1.0 + error) if np.isfinite(error) else -1.0
    return MatchResult(reference, homography, len(good), inliers, inlier_ratio, error, coverage, score, status)


def choose_best(results: list[MatchResult]) -> MatchResult | None:
    valid = [result for result in results if result.homography is not None and result.status == "ok"]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.score, r.inliers, r.inlier_ratio, -r.reprojection_error))


def save_fallback(crop: np.ndarray, output_path: Path, reference: Reference | None, mode: str) -> None:
    if mode == "none":
        return
    if mode == "copy" or reference is None:
        cv2.imwrite(str(output_path), crop)
        return
    ref_height, ref_width = reference.image.shape[:2]
    resized = cv2.resize(crop, (ref_width, ref_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), resized)


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    query_dir = args.query_dir.resolve()
    label_dir = args.label_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not query_dir.is_dir():
        raise FileNotFoundError(f"Query directory does not exist: {query_dir}")
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {reference_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_crop_dir = output_dir / "debug_crops"
    if args.save_debug_crops:
        debug_crop_dir.mkdir(parents=True, exist_ok=True)

    detector = create_feature(args.feature, args.max_features)
    matcher = create_matcher(args.feature)

    reference_paths = list_images(reference_dir)
    if not reference_paths:
        raise RuntimeError(f"No reference images found in {reference_dir}")
    references = [load_reference(path, detector, args.max_feature_side) for path in reference_paths]
    references = [reference for reference in references if reference.descriptors is not None and len(reference.keypoints) >= 4]
    if not references:
        raise RuntimeError("No reference image produced enough features.")

    all_query_paths = list_images(query_dir)
    if not all_query_paths:
        raise RuntimeError(f"No query images found in {query_dir}")
    query_paths = [
        path for offset, path in enumerate(all_query_paths) if offset % args.num_shards == args.shard_index
    ]
    if not query_paths:
        raise RuntimeError(
            f"Shard {args.shard_index}/{args.num_shards} has no query images from {query_dir}"
        )

    if args.num_shards == 1:
        manifest_path = output_dir / "warping_manifest.csv"
        details_path = output_dir / "reference_scores.csv"
    else:
        suffix = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
        manifest_path = output_dir / f"warping_manifest_{suffix}.csv"
        details_path = output_dir / f"reference_scores_{suffix}.csv"
    processed = 0
    failed = 0
    fallback_count = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file, details_path.open(
        "w", newline="", encoding="utf-8"
    ) as details_file:
        manifest = csv.writer(manifest_file)
        details = csv.writer(details_file)
        manifest.writerow(
            [
                "query",
                "output",
                "status",
                "best_reference",
                "score",
                "good_matches",
                "inliers",
                "inlier_ratio",
                "reprojection_error",
                "warp_coverage",
                "crop_x1",
                "crop_y1",
                "crop_x2",
                "crop_y2",
                "used_label_crop",
            ]
        )
        details.writerow(
            [
                "query",
                "reference",
                "status",
                "score",
                "good_matches",
                "inliers",
                "inlier_ratio",
                "reprojection_error",
                "warp_coverage",
            ]
        )

        for idx, query_path in enumerate(query_paths, start=1):
            output_path = output_dir / f"{query_path.stem}_warped.png"
            if output_path.exists() and not args.overwrite:
                processed += 1
                manifest.writerow([query_path.name, output_path.name, "skipped_existing", "", "", "", "", "", "", "", "", "", "", "", ""])
                continue

            image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
            if image is None:
                failed += 1
                manifest.writerow([query_path.name, output_path.name, "read_failed", "", "", "", "", "", "", "", "", "", "", "", ""])
                continue

            crop = crop_query(image, query_path, label_dir, args.crop_margin, not args.no_crop)
            crop_gray = ensure_uint8_gray(crop.image)
            feature_gray, query_scale_matrix, _ = scale_for_features(enhance_gray(crop_gray), args.max_feature_side)
            query_keypoints, query_desc = detector.detectAndCompute(feature_gray, None)

            if args.save_debug_crops:
                cv2.imwrite(str(debug_crop_dir / f"{query_path.stem}_crop.png"), crop.image)

            results = [
                evaluate_reference(
                    reference,
                    query_keypoints,
                    query_desc,
                    query_scale_matrix,
                    crop.image.shape[:2],
                    matcher,
                    args,
                )
                for reference in references
            ]
            for result in results:
                details.writerow(
                    [
                        query_path.name,
                        result.reference.path.name,
                        result.status,
                        f"{result.score:.6f}",
                        result.good_matches,
                        result.inliers,
                        f"{result.inlier_ratio:.6f}",
                        f"{result.reprojection_error:.6f}" if np.isfinite(result.reprojection_error) else "inf",
                        f"{result.warp_coverage:.6f}",
                    ]
                )

            best = choose_best(results)
            if best is None:
                fallback_count += 1
                fallback_ref = max(results, key=lambda r: (r.score, r.inliers, r.inlier_ratio), default=None)
                save_fallback(crop.image, output_path, fallback_ref.reference if fallback_ref else None, args.fallback)
                x1, y1, x2, y2 = crop.bbox_xyxy
                manifest.writerow(
                    [
                        query_path.name,
                        output_path.name if args.fallback != "none" else "",
                        "fallback_no_valid_homography" if args.fallback != "none" else "failed_no_valid_homography",
                        fallback_ref.reference.path.name if fallback_ref else "",
                        f"{fallback_ref.score:.6f}" if fallback_ref else "",
                        fallback_ref.good_matches if fallback_ref else "",
                        fallback_ref.inliers if fallback_ref else "",
                        f"{fallback_ref.inlier_ratio:.6f}" if fallback_ref else "",
                        f"{fallback_ref.reprojection_error:.6f}" if fallback_ref and np.isfinite(fallback_ref.reprojection_error) else "inf",
                        f"{fallback_ref.warp_coverage:.6f}" if fallback_ref else "",
                        x1,
                        y1,
                        x2,
                        y2,
                        crop.used_label,
                    ]
                )
                if args.fallback == "none":
                    failed += 1
                else:
                    processed += 1
            else:
                ref_height, ref_width = best.reference.image.shape[:2]
                warped = cv2.warpPerspective(crop.image, best.homography, (ref_width, ref_height))
                cv2.imwrite(str(output_path), warped)
                x1, y1, x2, y2 = crop.bbox_xyxy
                manifest.writerow(
                    [
                        query_path.name,
                        output_path.name,
                        "ok",
                        best.reference.path.name,
                        f"{best.score:.6f}",
                        best.good_matches,
                        best.inliers,
                        f"{best.inlier_ratio:.6f}",
                        f"{best.reprojection_error:.6f}",
                        f"{best.warp_coverage:.6f}",
                        x1,
                        y1,
                        x2,
                        y2,
                        crop.used_label,
                    ]
                )
                processed += 1

            if idx == 1 or idx % 10 == 0 or idx == len(query_paths):
                print(
                    f"shard {args.shard_index + 1}/{args.num_shards}: "
                    f"{idx}/{len(query_paths)} done: processed={processed}, "
                    f"fallback={fallback_count}, failed={failed}",
                    flush=True,
                )

    if failed:
        print(f"Finished with failures: processed={processed}, fallback={fallback_count}, failed={failed}")
        return 1
    print(f"Finished: processed={processed}, fallback={fallback_count}, failed={failed}")
    print(f"Manifest: {manifest_path}")
    print(f"Scores: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
