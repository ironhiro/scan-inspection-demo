# 스캔 이미지 손상 검출 · 여백 측정 · 검사 화면 데모

**데모:** https://ironhiro.github.io/scan-inspection-demo/ (브라우저에서 모델이 직접 실행되며 이미지는 서버로 전송되지 않습니다)

카드 형태 인쇄물의 스캔 이미지에서

1. 대상물을 찾아 원근 보정하고
2. 인쇄 영역의 상하좌우 여백(센터링)을 측정하고
3. 모서리 마모 · 테두리 마모 · 표면 긁힘 · 구김을 픽셀 단위로 검출해
4. 등급 기준표로 품질 등급을 계산한 뒤
5. 검사자가 웹 화면에서 결과를 확인 · 수정 · 완료 처리하는

전체 흐름을 보여 주는 데모입니다.

> **모델은 합성 데이터로만 학습했습니다.** 실제 스캔 이미지에 대한 성능 수치가 아닙니다.
> 시험 결과는 웹의 “모델 · 검증 결과” 페이지와 `web/public/metrics.json`에 있습니다.

## 시험 결과 (합성 시험 스캔 500장, 원본 스캔부터 전 과정)

| 항목 | 결과 |
|---|---|
| 손상 검출 (유형 무관) | 재현율 93.5% · 정밀도 91.8% |
| 손상 유형별 재현율 | 모서리 97.8% · 테두리 84.5% · 긁힘 98.7% · 구김 96.1% |
| 여백 오차 (검출 성공 건) | 중앙값 1.0px (0.1mm) · 95% 2.05px |
| 대상물 검출 실패 | 11 / 500 (대부분 밝은 테두리를 밝은 배경에서 스캔한 경우) |
| 등급 일치 (정답 라벨에 같은 기준표 적용 시) | 정확 일치 80.6% · ±1등급 98.8% |

전체 지표는 `web/public/metrics.json`.

## 구성

```
ml/        합성 데이터 생성, 학습, 평가 (Python, PyTorch, OpenCV)
  synth.py         카드 형태 인쇄물 + 손상 4종 + 스캔 조건 합성 (픽셀 단위 정답 포함)
  pipeline.py      대상물 검출 · 보정 · 여백 측정 · 손상 영역 분리 · 등급 계산
  gen_dataset.py   학습 5,000 / 검증 300 / 시험 500장 생성
  train.py         U-Net 학습 후 ONNX 내보내기
  eval.py          시험 스캔을 원본부터 끝까지 처리해 지표 계산 → web/public/metrics.json
  grade_analysis.py  등급 불일치를 감점 항목별로 분해 → ml/out/grade_analysis.json
web/       검사자 화면 (React, TypeScript, Vite)
  src/analysis/    pipeline.py와 같은 로직의 TypeScript 구현 + ONNX Runtime Web 추론 (Web Worker)
  src/data/        검사 건 저장소 인터페이스 (데모는 IndexedDB 구현)
  public/grading_rules.json  등급 기준표 (Python · 웹 공용)
server/    FastAPI 참조 구현 (기존 시스템 이미지는 읽기 전용, 결과·상태는 별도 테이블)
```

## 설계 메모

- **여백 측정은 딥러닝을 쓰지 않습니다.** 테두리 색을 추정하고 바깥에서 안쪽으로 한 줄씩 보며 인쇄 경계를 찾습니다. 결과가 항상 같고 근거를 설명할 수 있습니다.
- **대상물 검출은 네 변에 직선을 맞춰** 꼭짓점을 구합니다. 둥근 모서리나 모서리 마모에 흔들리지 않도록 각 변의 가운데 60%만 씁니다.
- **등급은 규칙 기반**입니다. 손상 크기(mm, mm²)와 센터링 비율을 `grading_rules.json`에 적용하므로, 같은 결과는 항상 같은 등급이 되고 감점 내역을 그대로 보여 줄 수 있습니다.
- **테두리 마모는 카드 전체 길이 합계로 한 번만 감점합니다.** 조각마다 감점하면 같은 마모가 몇 조각으로 나뉘어 검출되느냐에 따라 등급이 바뀌었습니다(`grade_analysis.py` 기준 등급 불일치의 절반 이상). 합산으로 바꾼 뒤 정확 일치가 73.8% → 80.6%가 됐습니다.
- **라이선스.** U-Net은 직접 구현했고 사전학습 가중치나 AGPL 라이브러리를 쓰지 않습니다.
- **검사자 수정 결과는 라벨 JSON으로 내보내** 재학습 데이터로 쓸 수 있습니다.

## 실행

```bash
# 웹 (모델과 샘플 이미지는 web/public에 포함)
cd web && npm ci && npm run dev

# 학습부터 다시
python -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/pip install numpy opencv-python-headless onnx onnxruntime
cd ml && python gen_dataset.py && python train.py 30 8 && python eval.py
cp out/damage_unet.onnx ../web/public/models/

# 서버 (참조 구현)
pip install -r server/requirements.txt && uvicorn server.app:app --reload
```
