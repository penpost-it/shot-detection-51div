"""사격 탄착군 분석 PoC (Streamlit front-end).

업로드한 표적지 이미지를 python_poc 의 기존 기능으로 처리해서
  1) reference 로 워핑하고
  2) 탄착(총알구멍)을 detection 하고
  3) 탄착군 형성(합/불)을 판정
한 결과를 보여 준다.

실행:
    cd front_poc && streamlit run app.py
    # 또는 repo 루트에서: streamlit run front_poc/app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# --- python_poc 백엔드를 import 경로에 올린다 -------------------------------
PYTHON_POC = Path(__file__).resolve().parent.parent / "python_poc"
if str(PYTHON_POC) not in sys.path:
    sys.path.insert(0, str(PYTHON_POC))

from grouping import STATUS_LABELS_KO, analyze_grouping, draw_grouping  # noqa: E402
from warping import warp_image_with_metadata  # noqa: E402

DETECTION_MODES = {
    "yolo": "YOLO (학습형)",
    "algorithm": "OpenCV 알고리즘 (비학습)",
    "manual": "수동 좌표 입력 (디버그)",
    "none": "detection 없음",
}

# 워핑·detection 이 잘 되는 것만 추려 둔 PoC 샘플 (detection_algorithm/test_fig).
SAMPLE_DIR = PYTHON_POC / "detection_algorithm" / "test_fig"
SAMPLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_samples() -> list[Path]:
    """test_fig 최상위의 표적 샘플 이미지 목록 (origin/ 등 하위 폴더는 제외)."""
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(
        p for p in SAMPLE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_SUFFIXES
    )


def discover_yolo_weights() -> str:
    """리포에는 가중치(.pt)를 커밋하지 않으므로, 로컬 관례 경로에서 자동 탐색한다.

    우선순위: 환경변수 YOLO_WEIGHTS → front_poc/weights → detection_yolo →
    repo 루트 → python_poc 트리(best.pt 우선). 못 찾으면 빈 문자열.
    """
    env = os.environ.get("YOLO_WEIGHTS", "").strip()
    if env and Path(env).is_file():
        return env

    repo_root = PYTHON_POC.parent
    search_dirs = [
        Path(__file__).resolve().parent / "weights",  # front_poc/weights/
        PYTHON_POC / "detection_yolo",
        repo_root,
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        # best.pt 를 먼저, 그다음 그 외 .pt
        for name in sorted(d.glob("*.pt"), key=lambda p: (p.name != "best.pt", p.name)):
            return str(name)
    # 마지막 수단: python_poc 트리 전체 탐색 (best.pt 우선)
    candidates = sorted(PYTHON_POC.rglob("*.pt"), key=lambda p: (p.name != "best.pt", str(p)))
    return str(candidates[0]) if candidates else ""


# --------------------------------------------------------------------------- #
# 백엔드 호출 헬퍼
# --------------------------------------------------------------------------- #
def read_image_bgr(uploaded) -> np.ndarray:
    """업로드된 파일 바이트를 BGR ndarray 로 읽는다 (워핑 미적용 경로)."""
    buffer = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 디코딩하지 못했습니다.")
    return image


def read_image_bgr_from_path(path) -> np.ndarray:
    """파일 경로(샘플 이미지)를 BGR ndarray 로 읽는다."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    return image


def warp_path(path) -> tuple[np.ndarray, object]:
    """파일 경로를 그대로 warp_image_with_metadata 에 넘긴다 (샘플 경로용)."""
    return warp_image_with_metadata(str(path))


def warp_uploaded(uploaded) -> tuple[np.ndarray, object]:
    """업로드 파일을 임시 경로에 저장하고 warp_image_with_metadata 를 호출한다."""
    suffix = Path(uploaded.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        tmp.close()
        return warp_image_with_metadata(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def parse_manual_points(text: str, box: int = 8) -> list[dict]:
    """'x,y' 줄들을 탄착 bbox dict 리스트로 변환 (디버그용)."""
    dets: list[dict] = []
    half = box / 2.0
    for line in text.splitlines():
        line = line.strip().replace("(", "").replace(")", "")
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        dets.append(
            {"x1": x - half, "y1": y - half, "x2": x + half, "y2": y + half,
             "confidence": 1.0, "class_name": "manual"}
        )
    return dets


def run_detection(image, mode, *, weights_path, confidence, manual_text):
    """선택한 모드로 detection 을 수행한다. (detections, error_message) 반환."""
    if mode == "none":
        return [], None
    if mode == "manual":
        return parse_manual_points(manual_text), None
    if mode == "algorithm":
        from detection_algorithm import detect as detect_algorithm

        _, dets = detect_algorithm(image, draw=False)
        if not dets:
            return [], "OpenCV 알고리즘이 탄착을 찾지 못했습니다 (워핑/정렬 품질을 확인하세요)."
        return dets, None
    if mode == "yolo":
        if not weights_path or not Path(weights_path).is_file():
            return [], "YOLO 가중치(.pt) 경로가 비었거나 파일이 아닙니다."
        try:
            from detection_yolo import detect as detect_yolo

            _, dets = detect_yolo(image, model_path=weights_path, confidence=confidence, draw=False)
            return dets, None
        except Exception as exc:  # ultralytics 미설치/가중치 오류 등
            return [], f"YOLO 추론 실패: {exc}"
    return [], f"알 수 없는 detection 모드: {mode}"


def det_to_dict(det) -> dict:
    if isinstance(det, dict):
        return det
    if is_dataclass(det):
        return asdict(det)
    return {k: getattr(det, k) for k in ("x1", "y1", "x2", "y2", "confidence", "class_name") if hasattr(det, k)}


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="사격 탄착군 분석 PoC", page_icon="🎯", layout="wide")
st.markdown(
    """
    <style>
      .verdict {font-size:1.6rem; font-weight:700; padding:0.6rem 1rem; border-radius:0.6rem;}
      .verdict.ok  {background:#e7f7ec; color:#0f7a32;}
      .verdict.bad {background:#fdecec; color:#c0252b;}
      .verdict.na  {background:#f1f1f3; color:#5f6368;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎯 사격 탄착군 분석")
st.caption("표적지 업로드 → 워핑 → 탄착 detection → 탄착군 형성(합/불) 판정")

samples = list_samples()

with st.sidebar:
    st.header("입력")
    source_mode = st.radio(
        "입력 소스",
        ["샘플 (test_fig)", "직접 업로드"],
        index=0 if samples else 1,
        help="test_fig 는 워핑·detection 이 잘 되도록 추려 둔 PoC 샘플입니다.",
    )
    uploaded, sample_path = None, None
    if source_mode == "샘플 (test_fig)":
        if samples:
            sample_path = st.selectbox(
                "샘플 이미지", samples, format_func=lambda p: p.name,
            )
        else:
            st.warning(f"샘플이 없습니다: {SAMPLE_DIR}")
    else:
        uploaded = st.file_uploader("표적지 이미지", type=["jpg", "jpeg", "png", "bmp", "webp"])

    # 워핑은 파이프라인 전제(기준 좌표계 정렬)라 항상 적용 — 사용자 토글 없음.
    warp_on = True

    # detection 은 자동: 학습된 YOLO 가중치가 로컬에 있으면 YOLO, 없으면 비학습
    # OpenCV 알고리즘으로 자동 fallback. 사용자는 아무것도 설정할 필요 없다.
    discovered_weights = discover_yolo_weights()
    auto_mode = "yolo" if discovered_weights else "algorithm"

    # 개발용 컨트롤 — 평소엔 접혀 있어 시연 화면을 깔끔하게 유지한다.
    weights_path, confidence, manual_text = discovered_weights, 0.40, ""
    with st.expander("⚙️ 고급 (개발용)", expanded=False):
        mode = st.radio(
            "탄착 detection 방식", list(DETECTION_MODES.keys()),
            format_func=lambda m: DETECTION_MODES[m],
            index=list(DETECTION_MODES).index(auto_mode),
            help="기본은 자동(YOLO 가중치가 있으면 YOLO, 없으면 OpenCV 알고리즘).",
        )
        # YOLO 관련 설정은 YOLO 모드일 때만 노출 (다른 모드에선 무의미)
        if mode == "yolo":
            weights_path = st.text_input(
                "YOLO 가중치 경로 (.pt)", value=discovered_weights, placeholder="(자동 탐색)",
                help="비워 두면 front_poc/weights · detection_yolo · 환경변수 YOLO_WEIGHTS 에서 "
                     "자동으로 찾습니다. 가중치는 용량이 커서 git 에 올리지 않습니다.",
            )
            confidence = st.slider(
                "YOLO confidence", 0.05, 0.95, 0.40, 0.05,
                help="이 점수 미만의 검출은 버립니다. 높이면 확실한 것만 남아 오검출↓(놓침↑), "
                     "낮추면 더 많이 잡지만 오검출↑.",
            )
        elif mode == "manual":
            manual_text = st.text_area("탄착 좌표 (워핑 캔버스 기준, 'x,y' 줄단위)",
                                       value="", height=120, placeholder="120,140\n160,150")
        st.divider()
        thr_mode = st.radio("탄착군 임계값 기준", ["캔버스 대각선 %", "픽셀(px)"], index=0)
        if thr_mode == "캔버스 대각선 %":
            # max 100%: PoC 표적 사진은 탄착이 표적 전체에 퍼져 지름이 대각선의 50%+ 인
            # 경우가 많아, 상한이 낮으면 판정이 범위 안에서 절대 안 뒤집혀 '먹통'처럼 보인다.
            threshold_frac = st.slider("임계값 (지름 ≤ 대각선의 %)", 1, 100, 12) / 100.0
            threshold_px = None
        else:
            threshold_frac = 0.12
            threshold_px = float(st.number_input("임계값 지름 (px)", min_value=1.0, value=150.0, step=1.0))
        min_shots = int(st.number_input("최소 탄착 수", min_value=1, value=2, step=1))

if uploaded is None and sample_path is None:
    st.info("⬅️ 사이드바에서 샘플을 고르거나 표적지 이미지를 업로드하세요.")
    st.stop()

# --- 1) 원본 읽기 + 워핑 ---------------------------------------------------- #
# 워핑은 파일 경로를 입력으로 받으므로, 업로드 파일은 임시 경로로 떨군다.
if sample_path is not None:
    original = read_image_bgr_from_path(sample_path)
    warp_source = warp_path
    warp_arg = sample_path
else:
    original = read_image_bgr(uploaded)
    warp_source = warp_uploaded
    warp_arg = uploaded

warp_meta = None
work_image = original
if warp_on:
    with st.spinner("워핑 중... (reference 매칭)"):
        try:
            work_image, warp_meta = warp_source(warp_arg)
        except Exception as exc:
            st.warning(f"워핑 실패 → 원본으로 진행합니다. ({exc})")
            work_image = original

# --- 2) detection --------------------------------------------------------- #
with st.spinner("탄착 detection 중..."):
    detections, det_error = run_detection(
        work_image, mode, weights_path=weights_path, confidence=confidence, manual_text=manual_text
    )
if det_error:
    st.warning(det_error)

# --- 3) 탄착군 판정 -------------------------------------------------------- #
height, width = work_image.shape[:2]
result = analyze_grouping(
    detections, (width, height),
    threshold_frac=threshold_frac, threshold_px=threshold_px, min_shots=min_shots,
)
annotated = draw_grouping(work_image, result, detections, draw_boxes=True)

# --- 4) 결과 표시 ---------------------------------------------------------- #
verdict_class = {True: "ok", False: "bad", None: "na"}[result.formed]
verdict_icon = {True: "✅", False: "❌", None: "⚠️"}[result.formed]
st.markdown(
    f'<div class="verdict {verdict_class}">{verdict_icon} {STATUS_LABELS_KO[result.status]}</div>',
    unsafe_allow_html=True,
)
st.caption(f"검출기: {DETECTION_MODES.get(mode, mode)} · 워핑: {'적용됨' if warp_meta else '실패→원본'}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("탄착 수", result.n_shots)
m2.metric("탄착군 지름", f"{result.diameter_px:.0f} px" if result.diameter_px is not None else "—")
m3.metric("지름 / 대각선", f"{result.diameter_frac * 100:.1f} %" if result.diameter_frac is not None else "—")
m4.metric("임계값", f"{result.threshold_px:.0f} px")

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("원본")
    st.image(original, channels="BGR", use_container_width=True)
with col_b:
    st.subheader("워핑 + 탄착군 결과")
    st.image(annotated, channels="BGR", use_container_width=True)

with st.expander("상세 정보 / 다운로드"):
    if warp_meta is not None:
        st.write(
            {
                "best_reference": Path(str(warp_meta.best_reference)).name,
                "score": round(warp_meta.score, 4),
                "inliers": warp_meta.inliers,
                "inlier_ratio": round(warp_meta.inlier_ratio, 4),
                "warp_coverage": round(warp_meta.warp_coverage, 4),
                "output_size": list(warp_meta.output_size),
            }
        )
    det_dicts = [det_to_dict(d) for d in detections]
    if det_dicts:
        st.dataframe(det_dicts, use_container_width=True)

    summary = {
        "warp": warp_on,
        "detection_mode": mode,
        "grouping": asdict(result),
        "detections": det_dicts,
    }
    ok, png = cv2.imencode(".png", annotated)
    if ok:
        st.download_button("결과 이미지(PNG) 다운로드", data=png.tobytes(),
                           file_name="grouping_result.png", mime="image/png")
    st.download_button("판정 요약(JSON) 다운로드",
                       data=json.dumps(summary, ensure_ascii=False, indent=2),
                       file_name="grouping_summary.json", mime="application/json")
