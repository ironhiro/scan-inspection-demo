import { useEffect, useMemo, useRef, useState } from 'react';
import { loadRules, type Rules } from '../analysis/grade';
import { aiGrade, fmtTime, runAnalysis, finalGrade } from '../data/caseOps';
import { repo, STATUS_LABEL, type CaseRecord, type Status } from '../data/repo';
import { useObjectUrl } from '../util';

type Filter = 'all' | Status;
const FILTERS: Filter[] = ['all', 'new', 'analyzed', 'reviewing', 'done'];

function Thumb({ blob }: { blob: Blob }) {
  const url = useObjectUrl(blob);
  return url ? <img className="thumb" src={url} alt="" /> : <div className="thumb" />;
}

export default function CaseList() {
  const [cases, setCases] = useState<CaseRecord[] | null>(null);
  const [rules, setRules] = useState<Rules | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const reload = () => repo.list().then(setCases);
  useEffect(() => {
    reload();
    loadRules().then(setRules);
  }, []);

  const counts = useMemo(() => {
    const c: Record<Filter, number> = { all: 0, new: 0, analyzed: 0, reviewing: 0, done: 0 };
    for (const r of cases ?? []) { c.all++; c[r.status]++; }
    return c;
  }, [cases]);

  const shown = (cases ?? []).filter((r) => filter === 'all' || r.status === filter);

  async function analyzeAll() {
    const todo = (cases ?? []).filter((r) => r.status === 'new');
    setError(null);
    for (let i = 0; i < todo.length; i++) {
      setBusy(`분석 중 ${i + 1} / ${todo.length}`);
      try {
        await runAnalysis(todo[i]);
      } catch (e) {
        setError(`${todo[i].id}: ${(e as Error).message}`);
      }
      await reload();
    }
    setBusy(null);
  }

  async function addFiles(files: FileList | File[]) {
    const imgs = [...files].filter((f) => f.type.startsWith('image/'));
    if (!imgs.length) return;
    setBusy('등록 중');
    for (const f of imgs) await repo.create(f, f.name, 'upload');
    await reload();
    setBusy(null);
  }

  async function reset() {
    setBusy('초기화 중');
    await repo.reset();
    await reload();
    setBusy(null);
  }

  return (
    <div
      className={`page ${dragOver ? 'dropzone-on' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files); }}
    >
      <div className="page-head">
        <h1>검사 작업 목록</h1>
        <span className="sub">스캔 이미지별 AI 분석 결과를 확인하고 검사를 완료합니다.</span>
      </div>

      <div className="toolbar">
        <div className="seg">
          {FILTERS.map((f) => (
            <button key={f} className={filter === f ? 'on' : ''} onClick={() => setFilter(f)}>
              {f === 'all' ? '전체' : STATUS_LABEL[f]} <span className="num">{counts[f]}</span>
            </button>
          ))}
        </div>
        <span className="spacer" />
        {busy && <span className="muted">{busy}</span>}
        <button className="btn" disabled={!!busy || !counts.new} onClick={analyzeAll}>
          분석 전 {counts.new}건 일괄 분석
        </button>
        <button className="btn" disabled={!!busy} onClick={() => fileRef.current?.click()}>이미지 등록</button>
        <input
          ref={fileRef} type="file" accept="image/*" multiple hidden
          onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ''; }}
        />
      </div>
      {error && <p className="err">분석 실패 — {error}</p>}

      {cases === null ? (
        <div className="empty">불러오는 중</div>
      ) : shown.length === 0 ? (
        <div className="empty">해당 상태의 검사 건이 없습니다.</div>
      ) : (
        <table className="grid">
          <thead>
            <tr>
              <th style={{ width: 46 }} />
              <th>건 번호</th>
              <th className="hide-sm">파일</th>
              <th className="hide-sm">등록</th>
              <th>상태</th>
              <th className="r">AI 손상</th>
              <th className="r">AI 등급</th>
              <th className="r">확정 등급</th>
              <th className="hide-sm">검사 완료</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const ag = rules ? aiGrade(r, rules) : null;
              const fg = rules && r.status === 'done' ? finalGrade(r, rules) : null;
              return (
                <tr key={r.id} className="row" onClick={() => (location.hash = `#/case/${r.id}`)}>
                  <td><Thumb blob={r.image} /></td>
                  <td><strong>{r.id}</strong></td>
                  <td className="hide-sm muted">{r.fileName}</td>
                  <td className="hide-sm num muted">{fmtTime(r.createdAt)}</td>
                  <td><span className={`status ${r.status}`}>{STATUS_LABEL[r.status]}</span></td>
                  <td className="r num">{r.ai ? r.ai.damages.length : '–'}</td>
                  <td className="r num"><span className="grade-pill">{ag ? ag.grade : '–'}</span></td>
                  <td className="r num">
                    <span className="grade-pill">{fg ?? '–'}</span>
                    {fg !== null && ag && fg !== ag.grade && <span className="changed"> 수정</span>}
                  </td>
                  <td className="hide-sm num muted">{r.completedAt ? fmtTime(r.completedAt) : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      <p className="hint" style={{ marginTop: 12 }}>
        이미지를 이 화면에 끌어다 놓아도 등록됩니다. 등록·수정 내용은 이 브라우저에만 저장됩니다.{' '}
        <button className="btn ghost small" disabled={!!busy} onClick={reset}>데모 데이터 초기화</button>
      </p>
    </div>
  );
}
