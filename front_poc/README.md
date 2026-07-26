# front_poc — 사격 탄착군 분석 PoC

표적지 이미지(샘플 갤러리에서 선택하거나 직접 업로드)를 넣으면
**워핑 → 탄착 detection → 탄착군 형성(합/불) 판정**까지 한 화면에서 보여 주는 Streamlit 데모.
담당 범위는 파이프라인 (3) 탄착군 형성 + (4) 프론트 종합.

```
업로드 ──▶ warp_image_with_metadata ──▶ detect_{yolo|algorithm} ──▶ analyze_grouping ──▶ draw_grouping ──▶ 화면
            (warping/)            (detection_*)     
            (grouping/)           (grouping/)
```

## 실행

```bash
# 1) 의존성 (가상환경 권장)
python3 -m venv .venv && source .venv/bin/activate
pip install -r front_poc/requirements.txt      # streamlit, opencv, numpy (+yolo면 ultralytics)

# 2) 실행 (repo 루트에서)
streamlit run front_poc/app.py
```

## 담당 파트 ①: 탄착군 형성 로직 — `python_poc/grouping/`

탄착 detection 결과(bbox)와 **무관하게** 동작하는 순수 함수. cv2/numpy 의존이 없어
(그리기 헬퍼만 cv2 lazy import) 테스트가 빠르고 CLI/프론트 공용으로 재사용된다.

```python
from grouping import analyze_grouping, draw_grouping

result = analyze_grouping(
    detections,            # x1,y1,x2,y2 가진 dict 또는 객체들 (YOLO Detection 호환)
    canvas_size=(w, h),    # 워핑된 이미지 크기
    threshold_frac=0.12,   # 판정 원 지름 = 캔버스 대각선의 %
    threshold_px=None,     # 주면 판정 원 지름을 px 절대값으로 지정
    min_shots=2,           # 합격에 필요한 원 내부 탄착 수
)
annotated = draw_grouping(image_bgr, result, detections)  # 고정 원+중심+판정 오버레이
```

**판정식**: 임계값을 **고정 원 지름**으로 보고, 그 원을 움직였을 때 들어가는 최대 탄착 수가
`min_shots` 이상이면 `formed=True`(합격).
- 고정 원 탐색: `find_best_fixed_circle()` — 모든 탄착점 쌍에서 가능한 반지름 `threshold / 2`
  원 중심 후보를 만들고, 후보마다 포함 탄착 수를 센다.
- 먼 탄착은 원을 키우지 않고 자동으로 제외된다.
- 임계값을 **대각선 %로 정규화**한 이유: reference 캔버스 크기가 서로 달라도 비교가 일관됨.

**`GroupingResult`**: `status`(`formed`/`not_formed`/`insufficient_shots`/`no_shots`),
`formed`(bool|None), `total_detection_count`, `included_shot_count`, `included_indices`,
`n_shots`(원 내부 탄착 수), `diameter_px`, `diameter_frac`, `circle_center/radius`,
`centroid`, `threshold_px`, `canvas_diagonal_px`.

**엣지케이스**: 0발→`no_shots`, `min_shots` 미만→`insufficient_shots` (둘 다 `formed=None`=판정불가).

**테스트** (설치 0개, 표준 라이브러리만):
```bash
cd python_poc && python3 -m unittest grouping.test_grouping -v   # 18 tests
```

## 담당 파트 ②: 프론트 — `front_poc/app.py`

시연 화면을 깔끔하게 — 사용자는 입력만 고르면 나머지는 자동이다.

- **사이드바(기본)**: 입력 소스만 노출 — `샘플 (test_fig)` 갤러리 ↔ `직접 업로드`.
  - **샘플 갤러리**: `detection_algorithm/test_fig/` 의 추려둔 PoC 이미지를 골라 바로 실행.
- **워핑**: 파이프라인 전제라 **항상 적용**(토글 없음). 실패 시 원본으로 자동 fallback.
- **detection 자동**: 학습된 YOLO 가중치(`*.pt`)가 로컬에 있으면 **YOLO**, 없으면 비학습
  **OpenCV 알고리즘**으로 자동 전환. 가중치 경로를 사용자가 입력할 필요 없음.
- **고급(개발용) expander**: detection 방식 수동 선택 · YOLO 가중치 경로 override · confidence ·
  manual 좌표 · 탄착군 임계값(% 또는 px) · 최소 탄착 수. 평소엔 접혀 있다.
- **메인**: 합/불 배지(한글, HTML) · 활성 검출기/워핑 상태 캡션 ·
  지표(검출된 전체 탄착·판정 원 내부 탄착·판정 원 밖 탄착·판정 원 지름·합격 필요 탄착) ·
  원본/결과 2단 이미지 · 상세+다운로드(PNG/JSON).
- **이미지 위 판정 라벨은 영어**(`PASS`/`FAIL` 등): `cv2.putText` 가 한글 글리프를 못 그려 `????` 가
  되므로, 임의 PC 시연에서 폰트 의존성 없이 동작하도록 영어로 표기. 화면 배지는 HTML 이라 한글 정상.

## 설계 결정 (요약)

- **Streamlit**: 백엔드가 Python이라 `run_pipeline` 계열 함수를 in-process로 바로 호출. PoC 속도+가독성.
- **경계 — 백엔드 무수정**: `warping/`, `detection_*/`, `main.py`를 건드리지 않고 기존 seam만 조립.
  detector가 stub이든 완성이든 프론트는 그대로 동작하고, 근형/학명 작업과 충돌하지 않는다.
- **manual 모드**(고급): detector 없이도 탄착군 로직/시각화를 좌표 직접 입력으로 데모·검증. 디버그용.

## 팀 통합 노트

- **detection 모델**: 학습형은 **YOLO26n**(`detection_yolo`, imgsz=960). 가중치 `*.pt`는 용량이 커서
  `.gitignore` 대상이라 repo 에 없다. 학습 가중치(`best.pt`)를 `front_poc/weights/` ·
  `python_poc/detection_yolo/` 중 한 곳에 두거나 환경변수 `YOLO_WEIGHTS=/경로/best.pt` 로 지정하면
  프론트가 자동으로 찾아 YOLO 모드로 동작한다. 없으면 OpenCV 알고리즘으로 자동 fallback.
- **algorithm**: `detection_algorithm` 은 기준 차분 기반으로 **구현 완료**(학명) — 가중치 없이도 동작.
- detection 좌표는 **워핑된 캔버스 기준**이어야 한다 (프론트는 워핑 후 detection 실행).

## 향후 확장 (현재 범위 밖)

- flyer(이상치) 1~2발 제외 옵션, 다발 그룹 별도 판정
- 정확도/점수: 점수링 중심 대비 군 중심 거리 → 점수 환산
- reference 캐싱(`st.cache_resource`)으로 워핑 반복 호출 가속
