#!/usr/bin/env python3
"""Python POC pipeline: optional warping + optional shot detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detection_algorithm import detect as detect_algorithm
from detection_yolo import detect as detect_yolo
from warping import DEFAULT_REFERENCE_DIR, warp_image_with_metadata


PROJECT_DIR = Path(__file__).resolve().parent


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = args.input_path_opt or args.input_path
    output_path = args.output_path_opt or args.output_path
    if input_path is None:
        raise ValueError("input path is required. Use positional INPUT_PATH or --input-path.")
    if output_path is None:
        raise ValueError("output path is required. Use positional OUTPUT_PATH or --output-path.")
    return Path(input_path), Path(output_path)


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    warp: bool = True,
    detection: str = "algorithm",
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
    yolo_model: str | Path | None = None,
    yolo_confidence: float = 0.40,
    algorithm_min_area: float = 20.0,
    algorithm_max_area: float = 5000.0,
    algorithm_min_circularity: float = 0.25,
    draw: bool = True,
    warp_feature: str = "sift",
    warp_max_features: int = 3000,
    warp_max_feature_side: int = 1000,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    warp_metadata = None
    if warp:
        image, warp_metadata = warp_image_with_metadata(
            input_path,
            reference_dir=reference_dir,
            feature=warp_feature,
            max_features=warp_max_features,
            max_feature_side=warp_max_feature_side,
        )
    else:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read input image: {input_path}")

    detections = []
    if detection == "none":
        output_image = image
    elif detection == "algorithm":
        output_image, detections = detect_algorithm(
            image,
            min_area=algorithm_min_area,
            max_area=algorithm_max_area,
            min_circularity=algorithm_min_circularity,
            draw=draw,
        )
    elif detection == "yolo":
        output_image, detections = detect_yolo(image, model_path=yolo_model, confidence=yolo_confidence, draw=draw)
    else:
        raise ValueError(f"Unsupported detection mode: {detection}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output_image):
        raise RuntimeError(f"Failed to save output image: {output_path}")

    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "warp": warp,
        "detection": detection,
        "detection_count": len(detections),
        "image_shape": [int(output_image.shape[0]), int(output_image.shape[1]), int(output_image.shape[2]) if output_image.ndim == 3 else 1],
    }
    if warp_metadata is not None:
        summary["warping"] = {
            "best_reference": str(warp_metadata.best_reference),
            "score": warp_metadata.score,
            "good_matches": warp_metadata.good_matches,
            "inliers": warp_metadata.inliers,
            "inlier_ratio": warp_metadata.inlier_ratio,
            "reprojection_error": warp_metadata.reprojection_error,
            "warp_coverage": warp_metadata.warp_coverage,
            "output_size": list(warp_metadata.output_size),
        }
    summary["detections"] = [det.__dict__ for det in detections]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional warping and shot detection on one image.")
    parser.add_argument("input_path", nargs="?", type=Path, help="Input image path.")
    parser.add_argument("output_path", nargs="?", type=Path, help="Output image path.")
    parser.add_argument("--input-path", dest="input_path_opt", type=Path, default=None, help="Input image path.")
    parser.add_argument("--output-path", dest="output_path_opt", type=Path, default=None, help="Output image path.")
    parser.add_argument("--warp", type=parse_bool, default=True, help="Whether to apply perspective warping.")
    parser.add_argument("--detection", choices=["none", "yolo", "algorithm"], default="algorithm", help="Detection mode.")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR, help="Top-view reference directory.")
    parser.add_argument("--draw", type=parse_bool, default=True, help="Draw detections on the saved image.")
    parser.add_argument("--yolo-model", type=Path, default=None, help="YOLO weight path for --detection yolo.")
    parser.add_argument("--yolo-confidence", type=float, default=0.40, help="YOLO confidence threshold (val-tuned optimum).")
    parser.add_argument("--algorithm-min-area", type=float, default=20.0, help="Minimum contour area for algorithm detection.")
    parser.add_argument("--algorithm-max-area", type=float, default=5000.0, help="Maximum contour area for algorithm detection.")
    parser.add_argument("--algorithm-min-circularity", type=float, default=0.25, help="Minimum contour circularity for algorithm detection.")
    parser.add_argument("--warp-feature", choices=["sift", "orb"], default="sift", help="Warping feature type.")
    parser.add_argument("--warp-max-features", type=int, default=3000, help="Maximum features for warping.")
    parser.add_argument("--warp-max-feature-side", type=int, default=1000, help="Feature extraction long-side size for warping.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path, output_path = resolve_paths(args)
    summary = run_pipeline(
        input_path,
        output_path,
        warp=args.warp,
        detection=args.detection,
        reference_dir=args.reference_dir,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        algorithm_min_area=args.algorithm_min_area,
        algorithm_max_area=args.algorithm_max_area,
        algorithm_min_circularity=args.algorithm_min_circularity,
        draw=args.draw,
        warp_feature=args.warp_feature,
        warp_max_features=args.warp_max_features,
        warp_max_feature_side=args.warp_max_feature_side,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"saved={summary['output_path']}")
        print(f"warp={summary['warp']}")
        print(f"detection={summary['detection']}")
        print(f"detection_count={summary['detection_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
