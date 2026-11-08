import { useEffect, useState } from 'react';
import { DAMAGE_LABEL, DAMAGE_TYPES } from '../analysis/types';

interface ClassM { gt: number; pred: number; recall: number | null; precision: number | null; pixel_iou: number | null }
interface Metrics {
  generated_at: string;
  data: { train: number; val: number; test: number };
  model: { input: [number, number]; onnx_mb: number };
  per_class: Record<string, ClassM>;
  any_damage: { recall: number; precision: number; gt: number; pred: number };
  clean_items: { n: number; false_alarm_rate: number | null };
  detection: { fail_threshold_px: number; failures: number; n: number; failures_low_contrast: number };
  corners_px: { median: number; p95: number };
  margins: { err_px_median: number; err_px_p95: number; ratio_err_pp_median: number; ratio_err_pp_p95: number; px_per_mm: number };
  grade: { exact: number; within1: number; mean_signed: number };
  cpu_ms_per_image: number;
  train_log: { epoch: number; loss: number; val_miou_fg: number }[];
}

const pct = (v: number | null | undefined) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`);

const SYNTH: Record<string, string> = {
  corner_wear: '모서리 둥근 부분에서 안쪽으로 불규칙하게 하얗게 일어나거나 떨어져 나간 영역',
  edge_wear: '모서리를 제외한 네 변을 따라 생긴 찍힘·마모. 밝은 테두리에서는 회색 오염으로 나타남',
  scratch: '표면을 가로지르는 1~2px 폭의 밝거나 어두운 곡선',
  crease: '한쪽은 밝고 반대쪽은 어두운 음영이 생기는 접힘·눌림 선',
};

export default function About() {
  const [m, setM] = useState<Metrics | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}metrics.json`).then((r) => r.json()).then(setM).catch(() => setErr(true));
  }, []);
  const base = import.meta.env.BASE_URL;

  return (
    <div className="doc">
      <h1>모델 · 검증 결과</h1>
      <p className="muted">
        이 데모의 모델은 <strong>합성 데이터로만 학습</strong>했습니다. 실제 스캔 이미지에 대한 성능을 뜻하지 않고,
        검사 시스템 전체 흐름과 검증 방식을 보여 주기 위한 것입니다.
      </p>

      <h2>처리 흐름</h2>
      <ol>
        <li><strong>대상물 검출·보정</strong> — 스캔 배경색과의 차이로 대상물을 찾고, 네 변에 직선을 맞춰 꼭짓점을 구한 뒤 원근 보정합니다. 모서리 마모나 둥근 모서리에 흔들리지 않도록 변의 가운데 60% 구간만 씁니다.</li>
        <li><strong>인쇄 여백 측정</strong> — 테두리 색을 추정한 뒤 바깥에서 안쪽으로 한 줄씩 보며 테두리 색이 아닌 픽셀이 절반을 넘는 첫 위치를 인쇄 경계로 잡습니다. 딥러닝을 쓰지 않아 결과가 항상 같고 근거를 설명할 수 있습니다.</li>
        <li><strong>손상 분할</strong> — 보정된 이미지를 {m ? `${m.model.input[0]}×${m.model.input[1]}` : '448×624'}로 줄여 U-Net이 픽셀마다 손상 유형을 판정하고, 이어진 영역을 손상 1건으로 묶습니다.</li>
        <li><strong>등급 산정</strong> — 손상 크기(mm, mm²)와 센터링 비율을 <a href={`${base}grading_rules.json`} target="_blank">등급 기준표(JSON)</a>에 적용합니다. 기준표만 바꾸면 등급 체계가 바뀝니다.</li>
        <li><strong>검사자 확인</strong> — 검사자가 손상 삭제·추가·유형 변경, 여백선 조정, 등급 지정을 하면 등급이 다시 계산되고 이력이 남습니다. 수정 결과는 라벨 JSON으로 내보내 재학습에 씁니다.</li>
      </ol>
      <p className="hint">모든 처리는 브라우저 안(Web Worker + ONNX Runtime Web)에서 실행되며, 이미지는 서버로 전송되지 않습니다.</p>

      <h2>학습 데이터 (합성)</h2>
      <p>
        카드 형태의 인쇄물(63×88mm, 약 254dpi)을 절차적으로 생성하고, 손상 4종을 픽셀 단위 정답과 함께 합성한 뒤
        스캔 조건(배경, ±2.5° 기울기, 배율, 흐림, 노이즈, JPEG 압축)을 흉내 냈습니다.
        {m && <> 학습 {m.data.train.toLocaleString()}장 · 검증 {m.data.val}장 · 시험 {m.data.test}장.</>}
      </p>
      <table className="grid">
        <thead><tr><th>유형</th><th>합성 방식 (= 라벨링 기준 초안)</th></tr></thead>
        <tbody>
          {DAMAGE_TYPES.map((t) => (
            <tr key={t}><td style={{ whiteSpace: 'nowrap' }}>{DAMAGE_LABEL[t]}</td><td>{SYNTH[t]}</td></tr>
          ))}
        </tbody>
      </table>

      <h2>시험 결과</h2>
      {err && <p className="err">metrics.json을 불러오지 못했습니다.</p>}
      {m && (
        <>
          <p>
            학습·검증에 쓰지 않은 시험 스캔 {m.data.test}장을 <strong>원본 스캔부터</strong> 앱과 같은 경로(검출 → 보정 → 추론)로
            처리해 측정했습니다. 손상 1건은 예측 영역의 30% 이상이 정답 영역(3px 여유) 안에 있으면 맞힌 것으로 셉니다.
          </p>
          <table className="grid">
            <thead>
              <tr><th>손상 유형</th><th className="r">정답 수</th><th className="r">검출 수</th><th className="r">재현율</th><th className="r">정밀도</th><th className="r">픽셀 IoU</th></tr>
            </thead>
            <tbody>
              {DAMAGE_TYPES.map((t) => {
                const c = m.per_class[t];
                return (
                  <tr key={t}>
                    <td>{DAMAGE_LABEL[t]}</td>
                    <td className="r num">{c.gt}</td><td className="r num">{c.pred}</td>
                    <td className="r num">{pct(c.recall)}</td><td className="r num">{pct(c.precision)}</td><td className="r num">{pct(c.pixel_iou)}</td>
                  </tr>
                );
              })}
              <tr>
                <td><strong>유형 무관</strong></td>
                <td className="r num">{m.any_damage.gt}</td><td className="r num">{m.any_damage.pred}</td>
                <td className="r num"><strong>{pct(m.any_damage.recall)}</strong></td><td className="r num"><strong>{pct(m.any_damage.precision)}</strong></td><td />
              </tr>
            </tbody>
          </table>

          <table className="grid" style={{ marginTop: 12 }}>
            <tbody>
              <tr>
                <td>대상물 검출 실패 (꼭짓점 오차 {m.detection.fail_threshold_px}px 초과)</td>
                <td className="r num">{m.detection.n}장 중 {m.detection.failures}장 ({pct(m.detection.failures / m.detection.n)}) · 이 중 {m.detection.failures_low_contrast}장은 밝은 테두리를 밝은 배경에서 스캔</td>
              </tr>
              <tr><td>대상물 꼭짓점 오차 (검출 성공 건)</td><td className="r num">중앙값 {m.corners_px.median.toFixed(2)}px · 95% {m.corners_px.p95.toFixed(2)}px</td></tr>
              <tr><td>여백 오차 (네 변 중 최대, 검출 성공 건)</td><td className="r num">중앙값 {m.margins.err_px_median.toFixed(2)}px ({(m.margins.err_px_median / m.margins.px_per_mm).toFixed(2)}mm) · 95% {m.margins.err_px_p95.toFixed(2)}px</td></tr>
              <tr><td>센터링 비율 오차 (검출 성공 건)</td><td className="r num">중앙값 {m.margins.ratio_err_pp_median.toFixed(2)}%p · 95% {m.margins.ratio_err_pp_p95.toFixed(2)}%p</td></tr>
              <tr><td>등급 일치 (정답 라벨에 같은 기준표 적용 시)</td><td className="r num">정확 일치 {pct(m.grade.exact)} · ±1등급 {pct(m.grade.within1)}</td></tr>
              <tr><td>무손상 대상물 오경보</td><td className="r num">{m.clean_items.n}장 중 {pct(m.clean_items.false_alarm_rate)} (표본이 적어 참고용)</td></tr>
              <tr><td>처리 시간 (CPU 1장, Python)</td><td className="r num">{Math.round(m.cpu_ms_per_image)}ms · 모델 {m.model.onnx_mb}MB</td></tr>
            </tbody>
          </table>
          <p className="hint">
            등급 일치는 사람 검사자가 아닌 “정답 라벨에 같은 기준표를 적용한 등급”과 비교한 값입니다.
            실데이터에서는 검사자 2명이 같은 이미지를 각각 판정해 사람끼리의 일치율을 먼저 재고, 그 수준을 목표로 삼습니다.
          </p>

          <h2>예시 — 보정 이미지 / 정답 / 예측</h2>
          <div className="figure">
            {[0, 1, 2, 3].map((i) => (
              <figure key={i}><img src={`${base}figs/compare_${i}.jpg`} alt={`비교 ${i + 1}`} loading="lazy" /></figure>
            ))}
          </div>

          <h2>학습 경과</h2>
          <table className="grid">
            <thead><tr><th className="r">epoch</th><th className="r">학습 loss</th><th className="r">검증 mIoU (손상 4종)</th></tr></thead>
            <tbody>
              {m.train_log.filter((r, i, a) => r.epoch % 5 === 0 || i === 0 || i === a.length - 1).map((r) => (
                <tr key={r.epoch}><td className="r num">{r.epoch}</td><td className="r num">{r.loss.toFixed(4)}</td><td className="r num">{pct(r.val_miou_fg)}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h2>한계와 실데이터 적용 시 할 일</h2>
      <ul>
        <li>합성 손상은 실제 손상보다 형태가 단순합니다. 이 모델을 실제 스캔에 그대로 쓰면 성능이 크게 떨어집니다.</li>
        <li>실데이터 1단계에서는 스캔 이미지 현황을 분석해 손상 유형 정의와 라벨링 기준서를 만들고, 초기 라벨링 후 이 구조 그대로 재학습해 실제 성능을 측정합니다.</li>
        <li>밝은 테두리를 밝은 배경에서 스캔하면 테두리와 배경이 구분되지 않아 인쇄 영역을 대상물로 잘못 잡습니다(위 검출 실패의 대부분). 스캔 배경을 표준화하거나, 실데이터 단계에서 경계 검출을 학습 모델로 보완합니다. 검사 화면의 ‘원본 스캔’ 보기에 검출 외곽선이 그려져 검사자가 바로 확인할 수 있습니다.</li>
        <li>등급 기준표는 예시입니다. 발주사 내부 등급 기준이 있으면 그대로 옮기고, 검사자 확정 등급이 쌓이면 기준표를 보정합니다.</li>
      </ul>

      <h2>구성</h2>
      <table className="grid">
        <tbody>
          <tr><td>학습</td><td>Python, PyTorch (U-Net 직접 구현, 외부 사전학습 가중치 없음), OpenCV</td></tr>
          <tr><td>추론</td><td>ONNX Runtime Web (WebGPU, 미지원 시 WASM) · Web Worker</td></tr>
          <tr><td>화면</td><td>React, TypeScript, Vite · 저장은 IndexedDB (서버 연동 시 같은 인터페이스로 교체)</td></tr>
          <tr><td>서버 (저장소 포함)</td><td>FastAPI + SQLite 참조 구현 · 같은 분석 로직(Python)과 ONNX 모델 사용 · Docker</td></tr>
        </tbody>
      </table>
    </div>
  );
}
