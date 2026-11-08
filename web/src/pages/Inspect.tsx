import { useCallback, useEffect, useMemo, useState } from 'react';
import { centering, grade, loadRules, measureOf, type Rules } from '../analysis/grade';
import { DAMAGE_LABEL, DAMAGE_TYPES, TYPE_COLOR, type Box, type Damage, type DamageType, type Margins } from '../analysis/types';
import Viewer, { type Layers } from '../components/Viewer';
import { aiGrade, effective, fmtTime, regionOf, runAnalysis } from '../data/caseOps';
import { now, repo, STATUS_LABEL, type CaseRecord, type Review } from '../data/repo';
import { useObjectUrl } from '../util';

const swatch = (t: DamageType) => ({ background: `rgb(${TYPE_COLOR[t].join(',')})` });
const SIDES: { key: keyof Margins; label: string }[] = [
  { key: 'left', label: '좌' }, { key: 'right', label: '우' }, { key: 'top', label: '상' }, { key: 'bottom', label: '하' },
];

export default function Inspect({ id }: { id: string }) {
  const [rec, setRec] = useState<CaseRecord | null>(null);
  const [all, setAll] = useState<CaseRecord[]>([]);
  const [rules, setRules] = useState<Rules | null>(null);
  const [mode, setMode] = useState<'rect' | 'scan'>('rect');
  const [layers, setLayers] = useState<Layers>({ mask: true, boxes: true, margins: true });
  const [tool, setTool] = useState<'select' | 'add'>('select');
  const [addType, setAddType] = useState<DamageType>('scratch');
  const [selected, setSelected] = useState<string | null>(null);
  const [fitSignal, setFitSignal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanSize, setScanSize] = useState<[number, number] | null>(null);
  const [liveMargins, setLiveMargins] = useState<Margins | null>(null);

  const rectUrl = useObjectUrl(rec?.ai?.rectified);
  const scanUrl = useObjectUrl(rec?.image);

  useEffect(() => {
    loadRules().then(setRules);
    repo.list().then(setAll);
    repo.get(id).then(async (r) => {
      if (!r) { setError('검사 건을 찾을 수 없습니다'); return; }
      setRec(r);
      if (r.status === 'new') {
        setBusy(true);
        try { setRec(await runAnalysis(r)); } catch (e) { setError((e as Error).message); }
        setBusy(false);
        repo.list().then(setAll);
      }
    });
  }, [id]);

  useEffect(() => {
    if (!scanUrl) return;
    const im = new Image();
    im.onload = () => setScanSize([im.naturalWidth, im.naturalHeight]);
    im.src = scanUrl;
  }, [scanUrl]);

  const save = useCallback(async (next: CaseRecord) => {
    setRec(next);
    await repo.put(next);
    setAll((a) => a.map((r) => (r.id === next.id ? { ...r, status: next.status } : r)));
  }, []);

  /** apply an edit to the review copy, starting the review on first edit */
  const edit = useCallback(
    (fn: (r: Review) => Review, logText: string) => {
      if (!rec?.ai) return;
      const base: Review = rec.review ?? {
        damages: rec.ai.damages.map((d) => ({ ...d })),
        margins: { ...rec.ai.margins },
        gradeOverride: null,
        note: '',
      };
      const log = [...rec.log];
      if (!rec.review) log.push({ at: now(), text: '검토 시작' });
      if (logText) log.push({ at: now(), text: logText });
      save({ ...rec, review: fn(base), status: rec.status === 'done' ? 'done' : 'reviewing', log });
    },
    [rec, save],
  );

  const eff = useMemo(() => (rec ? effective(rec) : null), [rec]);
  const margins = liveMargins ?? eff?.margins ?? null;
  const damages = useMemo(() => eff?.damages ?? [], [eff]);
  const locked = rec?.status === 'done';

  const g = useMemo(
    () => (rec?.ai && rules && margins ? grade(damages, margins, rec.ai.rectW, rules) : null),
    [rec, rules, margins, damages],
  );
  const ag = rec && rules ? aiGrade(rec, rules) : null;
  const finalG = rec?.review?.gradeOverride ?? g?.grade ?? null;

  const removeDamage = useCallback(
    (did: string) => {
      const d = damages.find((x) => x.id === did);
      if (!d || locked) return;
      edit((r) => ({ ...r, damages: r.damages.filter((x) => x.id !== did) }), `${d.id} ${DAMAGE_LABEL[d.type]} 삭제`);
      setSelected(null);
    },
    [damages, edit, locked],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, select')) return;
      if ((e.key === 'Delete' || e.key === 'Backspace') && selected) removeDamage(selected);
      else if (e.key === 'Escape') { setSelected(null); setTool('select'); }
      else if (e.key === 'a' && !locked) setTool((t) => (t === 'add' ? 'select' : 'add'));
      else if (e.key === 'f') setFitSignal((n) => n + 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected, removeDamage, locked]);

  if (error && !rec) return <div className="page"><p className="err">{error}</p></div>;
  if (!rec) return <div className="page muted">불러오는 중</div>;

  const ai = rec.ai;
  const idx = all.findIndex((r) => r.id === rec.id);
  const nextOpen = all.slice(idx + 1).concat(all.slice(0, idx)).find((r) => r.status !== 'done');

  function onMargins(m: Margins, commit: boolean) {
    if (!commit) { setLiveMargins(m); return; }
    setLiveMargins(null);
    const before = eff!.margins;
    const changed = SIDES.filter((s) => before[s.key] !== m[s.key]).map((s) => `${s.label} ${before[s.key]}→${m[s.key]}px`);
    if (changed.length) edit((r) => ({ ...r, margins: m }), `여백 수정 ${changed.join(', ')}`);
  }

  function onAdd(box: Box) {
    const n = damages.reduce((mx, x) => (x.id.startsWith('m-') ? Math.max(mx, Number(x.id.slice(2)) || 0) : mx), 0) + 1;
    const d: Damage = { id: `m-${n}`, type: addType, bbox: box, area: box[2] * box[3], conf: null, source: 'inspector' };
    edit((r) => ({ ...r, damages: [...r.damages, d] }), `${DAMAGE_LABEL[addType]} 영역 추가`);
    setSelected(d.id);
    setTool('select');
  }

  function changeType(d: Damage, t: DamageType) {
    edit(
      (r) => ({ ...r, damages: r.damages.map((x) => (x.id === d.id ? { ...x, type: t } : x)) }),
      `${d.id} 유형 변경 ${DAMAGE_LABEL[d.type]}→${DAMAGE_LABEL[t]}`,
    );
  }

  function complete() {
    if (!rec || finalG === null) return;
    const base = rec.review ?? { damages: rec.ai!.damages, margins: rec.ai!.margins, gradeOverride: null, note: '' };
    save({
      ...rec,
      review: base,
      status: 'done',
      completedAt: now(),
      log: [...rec.log, { at: now(), text: `검사 완료 · 확정 등급 ${finalG} (AI ${ag?.grade ?? '–'})` }],
    });
  }

  function reopen() {
    save({ ...rec!, status: 'reviewing', completedAt: null, log: [...rec!.log, { at: now(), text: '검사 다시 열기' }] });
  }

  function revert() {
    save({ ...rec!, review: null, status: 'analyzed', log: [...rec!.log, { at: now(), text: '수정 내용 취소, AI 결과로 되돌림' }] });
    setSelected(null);
  }

  async function reanalyze() {
    setBusy(true);
    setError(null);
    try { setRec(await runAnalysis(rec!)); } catch (e) { setError((e as Error).message); }
    setBusy(false);
  }

  function exportLabels() {
    if (!ai || !eff) return;
    const out = {
      case_id: rec!.id,
      file_name: rec!.fileName,
      status: rec!.status,
      rectified_size: [ai.rectW, ai.rectH],
      corners_in_scan: ai.corners,
      model: ai.modelVersion,
      margins_px: eff.margins,
      grade: { ai: ag?.grade ?? null, final: finalG },
      annotations: eff.damages.map((d) => ({
        id: d.id, category: d.type, bbox_xywh: d.bbox.map((v) => +v.toFixed(1)), source: d.source,
        ai_confidence: d.conf === null ? null : +d.conf.toFixed(3),
      })),
      removed_ai_detections: rec!.review ? ai.damages.filter((a) => !eff.damages.some((d) => d.id === a.id)).map((d) => d.id) : [],
    };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }));
    a.download = `${rec!.id}_labels.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const c = margins ? centering(margins) : null;
  const mm = rules && ai ? rules.card_width_mm / ai.rectW : 0;
  const aiIds = new Set(ai?.damages.map((d) => d.id));
  const removedCount = rec.review && ai ? ai.damages.filter((a) => !rec.review!.damages.some((d) => d.id === a.id)).length : 0;
  const addedCount = damages.filter((d) => !aiIds.has(d.id)).length;

  return (
    <div className="inspect">
      <aside className="queue">
        <div className="queue-head"><a href="#/">← 목록</a><span>{all.length}건</span></div>
        {all.map((r) => (
          <a key={r.id} href={`#/case/${r.id}`} className={r.id === rec.id ? 'on' : ''}>
            <span className="id">{r.id}</span>
            <span className={`st status ${r.status}`}>{STATUS_LABEL[r.status]}</span>
          </a>
        ))}
      </aside>

      <div className="stage-wrap">
        <div className="stage-bar">
          <div className="seg">
            <button className={mode === 'rect' ? 'on' : ''} onClick={() => setMode('rect')}>보정 이미지</button>
            <button className={mode === 'scan' ? 'on' : ''} onClick={() => setMode('scan')}>원본 스캔</button>
          </div>
          <span className="sep" />
          <label className="check"><input type="checkbox" checked={layers.mask} onChange={(e) => setLayers({ ...layers, mask: e.target.checked })} />손상 영역</label>
          <label className="check"><input type="checkbox" checked={layers.boxes} onChange={(e) => setLayers({ ...layers, boxes: e.target.checked })} />박스</label>
          <label className="check"><input type="checkbox" checked={layers.margins} onChange={(e) => setLayers({ ...layers, margins: e.target.checked })} />여백선</label>
          <span className="sep" />
          <button className={`btn ${tool === 'add' ? 'primary' : ''}`} disabled={locked || mode !== 'rect' || !ai} onClick={() => setTool(tool === 'add' ? 'select' : 'add')}>
            영역 추가 (A)
          </button>
          <select value={addType} onChange={(e) => setAddType(e.target.value as DamageType)} disabled={locked}>
            {DAMAGE_TYPES.map((t) => <option key={t} value={t}>{DAMAGE_LABEL[t]}</option>)}
          </select>
          <span style={{ flex: 1 }} />
          <button className="btn" onClick={() => setFitSignal((n) => n + 1)}>화면 맞춤 (F)</button>
        </div>

        {ai || mode === 'scan' ? (
          <Viewer
            mode={mode}
            rectUrl={rectUrl}
            scanUrl={scanUrl}
            rectW={ai?.rectW ?? 0}
            rectH={ai?.rectH ?? 0}
            scanSize={scanSize}
            corners={ai?.corners ?? null}
            labels={ai?.labels ?? null}
            damages={damages}
            margins={margins}
            selected={selected}
            layers={layers}
            tool={locked ? 'select' : tool}
            onSelect={setSelected}
            onMargins={locked ? () => {} : onMargins}
            onAdd={onAdd}
            fitSignal={fitSignal}
          />
        ) : (
          <div className="stage"><div className="stage-msg">{busy ? 'AI 분석 중 — 처음에는 모델(약 8MB)을 내려받습니다' : error ?? ''}</div></div>
        )}

        <div className="stage-foot">
          <span className="legend">
            {DAMAGE_TYPES.map((t) => <span key={t}><i className="swatch" style={swatch(t)} />{DAMAGE_LABEL[t]}</span>)}
            <span><i className="swatch" style={{ background: '#51cf66' }} />여백선 (드래그로 수정)</span>
          </span>
          <span style={{ flex: 1 }} />
          <span>휠 확대 · 드래그 이동 · Delete 삭제 · Esc 해제</span>
        </div>
      </div>

      <aside className="side">
        <section>
          <div className="side-head">
            <span className="id">{rec.id}</span>
            <span className={`status ${rec.status}`}>{STATUS_LABEL[rec.status]}</span>
            <span className="spacer" />
            {locked ? (
              <button className="btn" onClick={reopen}>다시 열기</button>
            ) : (
              <button className="btn primary" disabled={!ai || busy} onClick={complete}>검사 완료</button>
            )}
          </div>
          <div className="hint" style={{ marginTop: 4 }}>{rec.fileName} · 등록 {fmtTime(rec.createdAt)}</div>
          {locked && nextOpen && (
            <div style={{ marginTop: 8 }}><a href={`#/case/${nextOpen.id}`}>다음 미완료 건 {nextOpen.id} →</a></div>
          )}
          {error && <p className="err">{error}</p>}
        </section>

        {ai && g && (
          <section>
            <h2>
              품질 등급
              <span className="faint" style={{ fontWeight: 400 }}>{rules?.name}</span>
            </h2>
            <div className="grade-row">
              <div>
                <div className="grade-big num">{finalG}</div>
                <div className="grade-meta">{rec.review?.gradeOverride != null ? '검사자 지정' : rec.review ? '수정 반영' : 'AI 판정'}</div>
              </div>
              <div className="grade-meta">
                AI 판정 <strong className="num">{ag?.grade}</strong>
                {g.grade !== ag?.grade && <> → 수정 반영 <strong className="num changed">{g.grade}</strong></>}
                <div style={{ marginTop: 6 }}>
                  확정 등급{' '}
                  <select
                    disabled={locked}
                    value={rec.review?.gradeOverride ?? ''}
                    onChange={(e) => {
                      const v = e.target.value === '' ? null : Number(e.target.value);
                      edit((r) => ({ ...r, gradeOverride: v }), v === null ? '확정 등급 자동 계산으로 변경' : `확정 등급 ${v}로 지정`);
                    }}
                  >
                    <option value="">자동 ({g.grade})</option>
                    {[10, 9, 8, 7, 6, 5, 4, 3, 2, 1].map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </div>
              </div>
            </div>
            <table className="kv" style={{ marginTop: 10 }}>
              <tbody>
                {g.items.length === 0 && <tr><td className="muted">감점 항목 없음</td></tr>}
                {g.items.map((it, i) => (
                  <tr key={i} onClick={() => it.damageId && setSelected(it.damageId)} style={{ cursor: it.damageId ? 'pointer' : undefined }}>
                    <th>{it.reason === 'centering' ? '센터링' : `${DAMAGE_LABEL[it.reason]} ${it.damageId}`}</th>
                    <td className="r num muted">{it.reason === 'centering' ? `최대 ${it.value.toFixed(1)}%` : `${it.value.toFixed(2)} ${it.unit}`}</td>
                    <td className="r num" style={{ width: 44 }}>−{it.deduct}</td>
                  </tr>
                ))}
                {g.cap < 10 && g.capReason && (
                  <tr><th colSpan={2}>{DAMAGE_LABEL[g.capReason]} 있음 → 최고 {g.cap}등급</th><td /></tr>
                )}
              </tbody>
            </table>
          </section>
        )}

        {ai && margins && c && (
          <section>
            <h2>
              인쇄 여백
              {rec.review && SIDES.some((s) => rec.review!.margins[s.key] !== ai.margins[s.key]) && (
                <button className="btn ghost small" disabled={locked} onClick={() => edit((r) => ({ ...r, margins: { ...ai.margins } }), '여백을 AI 값으로 되돌림')}>AI 값으로</button>
              )}
            </h2>
            <table className="kv">
              <tbody>
                <tr><th>좌 / 우</th>
                  <td className="r num">{margins.left} / {margins.right} px</td>
                  <td className="r num">{(margins.left * mm).toFixed(2)} / {(margins.right * mm).toFixed(2)} mm</td></tr>
                <tr><th>상 / 하</th>
                  <td className="r num">{margins.top} / {margins.bottom} px</td>
                  <td className="r num">{(margins.top * mm).toFixed(2)} / {(margins.bottom * mm).toFixed(2)} mm</td></tr>
                <tr><th>좌우 비율</th><td className="r num" colSpan={2}>{c.lr.toFixed(1)} : {(100 - c.lr).toFixed(1)}</td></tr>
                <tr><th>상하 비율</th><td className="r num" colSpan={2}>{c.tb.toFixed(1)} : {(100 - c.tb).toFixed(1)}</td></tr>
              </tbody>
            </table>
            <p className="hint">mm 환산은 대상물 폭 {rules?.card_width_mm}mm 기준입니다.</p>
          </section>
        )}

        {ai && (
          <section>
            <h2>
              손상 {damages.length}건
              <span className="faint" style={{ fontWeight: 400 }}>
                {(removedCount > 0 || addedCount > 0) && `AI ${ai.damages.length} · 삭제 ${removedCount} · 추가 ${addedCount}`}
              </span>
            </h2>
            {damages.length === 0 && <p className="hint">검출된 손상이 없습니다. 놓친 손상은 ‘영역 추가’로 표시하세요.</p>}
            <ul className="dmg-list">
              {damages.map((d) => {
                const meas = rules ? measureOf(d, rules, ai.rectW) : null;
                return (
                  <li key={d.id} className={d.id === selected ? 'sel' : ''} onClick={() => setSelected(d.id)}>
                    <i className="swatch" style={swatch(d.type)} />
                    <select value={d.type} disabled={locked} onClick={(e) => e.stopPropagation()} onChange={(e) => changeType(d, e.target.value as DamageType)}>
                      {DAMAGE_TYPES.map((t) => <option key={t} value={t}>{DAMAGE_LABEL[t]}</option>)}
                    </select>
                    <span className={`tag ${d.source === 'inspector' ? 'insp' : ''}`}>{d.source === 'ai' ? 'AI' : '검사자'}</span>
                    <button className="btn ghost small" disabled={locked} onClick={(e) => { e.stopPropagation(); removeDamage(d.id); }}>삭제</button>
                    <span className="meta num">
                      <span>{d.id}</span>
                      <span>{regionOf(d, ai.rectW, ai.rectH, margins!)}</span>
                      {meas && <span>{meas.value.toFixed(2)} {meas.unit}</span>}
                      {d.conf !== null && <span>신뢰도 {(d.conf * 100).toFixed(0)}%</span>}
                    </span>
                  </li>
                );
              })}
            </ul>
          </section>
        )}

        {ai && (
          <section>
            <h2>검사 메모</h2>
            <textarea
              disabled={locked}
              defaultValue={rec.review?.note ?? ''}
              placeholder="판정 근거, 재스캔 필요 여부 등"
              onBlur={(e) => {
                if (e.target.value !== (rec.review?.note ?? '')) edit((r) => ({ ...r, note: e.target.value }), '메모 수정');
              }}
            />
          </section>
        )}

        <section>
          <h2>
            처리 이력
            <span>
              {ai && <button className="btn ghost small" onClick={exportLabels}>라벨 JSON</button>}
              {rec.review && !locked && <button className="btn ghost small" onClick={revert}>수정 취소</button>}
              {!locked && <button className="btn ghost small" disabled={busy} onClick={reanalyze}>재분석</button>}
            </span>
          </h2>
          <ul className="log">
            {[...rec.log].reverse().map((l, i) => (
              <li key={i}><time className="num">{fmtTime(l.at)}</time><span>{l.text}</span></li>
            ))}
          </ul>
        </section>

        {ai && (
          <section className="hint">
            {ai.modelVersion} · {ai.provider} · 보정 {Math.round(ai.elapsedMs.detect)}ms · 추론 {Math.round(ai.elapsedMs.infer)}ms
          </section>
        )}
      </aside>
    </div>
  );
}
