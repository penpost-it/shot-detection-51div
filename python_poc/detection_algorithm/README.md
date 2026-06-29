# detection_algorithm — 기준 차분 기반 탄착 검출 (No-ML)

워핑된 표적 이미지를 **깨끗한 기준 표적(`target_reference.png`)과 차분**하여, 머신러닝 없이 고전 영상처리(OpenCV)만으로 총알 구멍을 검출하는 모듈입니다.

`detection_yolo`(YOLO 기반)와 동일한 인터페이스(`detect(image, ...) -> (output_image, detections)`)를 제공하여, `main.py` 파이프라인에서 `--detection algorithm`으로 바로 교체해 쓸 수 있습니다.

---

## 파이프라인 내 위치

```
입력 이미지 ──▶ warping(투시 보정) ──▶ detection_algorithm.detect ──▶ 결과 이미지 + 검출 목록
                  (top-view 정렬)         (기준과 차분)
```

`detect()`에 들어오는 `image`는 **이미 워핑되어 기준 좌표계로 정렬된 상태**라고 가정합니다(`main.py`에서 `--warp true`로 수행). 검출 단계는 그 위에서 ECC 미세 정렬을 한 번 더 수행해 잔차를 보정합니다.

---

## 디렉터리 구성

```
detection_algorithm/
├── __init__.py          # Detection, detect 공개
├── detector.py          # 검출 알고리즘 본체
├── README.md            # 이 문서
└── test_fig/            # 샘플 입력 이미지
    ├── 1.jpg, 29.jpg, 117.jpg, 124.jpg   # 탄착이 있는 표적 사진
    └── origin/target_origin.png          # 탄착 없는 깨끗한 표적 샘플
```

> 검출/워핑 시각화 산출물(`_detect_check/`, `_warp_check/`)은 생성물이며 `.gitignore`에 등록되어 커밋되지 않습니다.

---

## 기준 이미지 (고정)

차분의 기준은 **항상** 아래 파일로 고정되어 있습니다.

```
python_poc/warping/top_view_reference/target_reference.png
```

코드 상수는 `detector.REFERENCE_PATH` 이며, 호출 인자로 바꿀 수 없습니다(의도된 고정). 기준을 교체하려면 이 파일을 바꾸면 됩니다. 기준 이미지는 **탄착이 없는 깨끗한 정면(top-view) 표적**이어야 합니다.

---

## 검출 알고리즘 (단계)

핵심 아이디어: **총알 구멍은 절대적으로 어두운 작은 점**이고, **링/숫자는 밝은 인쇄물**이라는 차이를 이용합니다.

1. **기준 로드** — `target_reference.png`를 읽어 입력과 동일 크기로 맞춤(캐시).
2. **ECC 미세 정렬** — `cv2.findTransformECC`(affine)로 입력을 기준에 서브픽셀 정렬. 워핑 잔차(1~2px)를 줄여 링 잔재를 억제.
3. **Black-hat 차분** — `blackhat(입력) − blackhat(기준)`.
   - 구멍(어두운 점)은 black-hat에서 밝게 부각됨.
   - 밝은 링/숫자는 black-hat에 거의 안 잡혀 단순 차분 대비 잔재가 크게 줄어듦.
4. **절대 암점 게이트** — 입력 black-hat이 충분히 큰(진짜로 어두운) 화소만 통과. 밝은 링·숫자 잔재를 제거하는 핵심 단계.
5. **어두운 인쇄물 억제** — 기준 black-hat에서 어두운 인쇄 요소(중앙 "10" 글자 등)를 마스킹해 제거. 밝은 링은 4단계가 처리하므로 여기서 건드리지 않아 **링 위의 실제 구멍**을 보존.
6. **표적 실루엣 제한** — 기준의 녹색 표적 영역(중앙 흰 원판 포함, 가장자리 침식) 안으로 검출을 한정.
7. **형태 필터** — 컨투어를 면적(`min/max_area`)·원형도(`min_circularity`)·채움률(`extent ≥ 0.32`)·종횡비(`aspect ≤ 4.0`)로 필터링. 둥근 구멍은 살리고 가늘고 긴 링 호(arc) 조각은 제거.

---

## API

### `detect(image, **params) -> (output_image, detections)`

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `image` | (필수) | BGR `np.ndarray`. 워핑되어 기준 좌표계로 정렬된 이미지 가정. |
| `min_area` | `20.0` | 검출 blob 최소 면적(px²). |
| `max_area` | `5000.0` | 검출 blob 최대 면적(px²). |
| `min_circularity` | `0.25` | 최소 원형도 `4πA/P²`. |
| `blackhat_kernel` | `41` | black-hat 타원 커널 크기(홀수). 가장 큰 구멍보다 커야 함. |
| `response_threshold` | `14.0` | 차분 응답 이진화 임계값. |
| `abs_dark_threshold` | `26.0` | 절대 암점 게이트 임계값(클수록 엄격 → 오검출↓, 재현율↓). |
| `align` | `True` | ECC 미세 정렬 사용 여부. |
| `suppress_structure` | `True` | 어두운 인쇄물 억제 사용 여부. |
| `draw` | `True` | 결과 이미지에 검출 원/개수 표기. |

`min_area`, `max_area`, `min_circularity`는 `main.py`가 그대로 전달합니다. 나머지(`response_threshold`, `abs_dark_threshold` 등)는 함수 기본값으로 동작하며, 직접 호출 시 조정 가능합니다.

### `Detection` (dataclass)

| 필드 | 설명 |
|------|------|
| `x1, y1, x2, y2` | 바운딩 박스 좌표 |
| `cx, cy` | 중심 좌표 |
| `radius` | 최소 외접원 반지름 |
| `area` | 컨투어 면적 |
| `confidence` | 응답 세기 기반 신뢰도(0~1) |
| `class_name` | `"bullet_hole"` |

---

## 사용법

### 1) 파이프라인(main.py)으로 실행

```bash
cd python_poc
python main.py detection_algorithm/test_fig/1.jpg out.png \
  --warp true --detection algorithm --json
```

### 2) 코드에서 직접 호출

```python
import cv2
from warping import warp_image_with_metadata
from detection_algorithm import detect

warped, _ = warp_image_with_metadata("detection_algorithm/test_fig/1.jpg")
output, detections = detect(warped)
print(f"검출된 탄착 수: {len(detections)}")
cv2.imwrite("out.png", output)
```

---

## 의존성

- Python 3.10+
- `opencv-python`, `numpy`

```bash
pip install opencv-python numpy
```

---

## 한계 및 튜닝

- **종이 주름/접힌 자국**처럼 물리적으로 어두운 영역은 구멍과 구분이 어려워 일부 오검출될 수 있습니다.
  - `abs_dark_threshold`를 올리면 오검출이 줄지만 흐릿한 구멍을 놓칠 수 있습니다.
- **군집 탄착**은 하나의 큰 blob으로 합쳐질 수 있습니다. 개별 분리가 필요하면 distance-transform + watershed를 추가하세요.
- **링 위 구멍**은 보존되도록 설계했으나, 정렬 품질이 나쁘면 링 잔재가 늘 수 있습니다. 워핑 정합 품질이 결과에 직접적인 영향을 줍니다.
- 검출은 입력이 기준 좌표계로 **워핑된 상태**임을 전제로 합니다. `--warp false`로 정렬되지 않은 원본을 넣으면 차분이 무의미해집니다.
