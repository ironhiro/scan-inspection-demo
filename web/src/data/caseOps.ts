import { analyze } from '../analysis/client';
import { grade, type Rules } from '../analysis/grade';
import type { Damage, Margins } from '../analysis/types';
import { now, repo, type CaseRecord } from './repo';

export function effective(rec: CaseRecord): { damages: Damage[]; margins: Margins } | null {
  if (rec.review) return { damages: rec.review.damages, margins: rec.review.margins };
  if (rec.ai) return { damages: rec.ai.damages, margins: rec.ai.margins };
  return null;
}

export function aiGrade(rec: CaseRecord, rules: Rules) {
  return rec.ai ? grade(rec.ai.damages, rec.ai.margins, rec.ai.rectW, rules) : null;
}

export function finalGrade(rec: CaseRecord, rules: Rules) {
  const e = effective(rec);
  if (!e || !rec.ai) return null;
  const g = grade(e.damages, e.margins, rec.ai.rectW, rules);
  return rec.review?.gradeOverride ?? g.grade;
}

const inflight = new Map<string, Promise<CaseRecord>>();

/** analyse a case; concurrent calls for the same case share one run */
export function runAnalysis(rec: CaseRecord): Promise<CaseRecord> {
  let p = inflight.get(rec.id);
  if (!p) {
    p = analyzeCase(rec).finally(() => inflight.delete(rec.id));
    inflight.set(rec.id, p);
  }
  return p;
}

async function analyzeCase(rec: CaseRecord): Promise<CaseRecord> {
  const ai = await analyze(rec.image);
  const next: CaseRecord = {
    ...rec,
    ai,
    review: null,
    status: 'analyzed',
    completedAt: null,
    log: [...rec.log, { at: now(), text: `AI 분석 완료 · 손상 ${ai.damages.length}건 · ${Math.round(ai.elapsedMs.total)}ms (${ai.provider})` }],
  };
  await repo.put(next);
  return next;
}

export function regionOf(d: Damage, w: number, h: number, m: Margins): string {
  const cx = d.bbox[0] + d.bbox[2] / 2;
  const cy = d.bbox[1] + d.bbox[3] / 2;
  const nearL = cx < w * 0.12, nearR = cx > w * 0.88, nearT = cy < h * 0.09, nearB = cy > h * 0.91;
  if ((nearL || nearR) && (nearT || nearB)) return `${nearT ? '상' : '하'}${nearL ? '좌' : '우'} 모서리`;
  const [x0, y0, bw, bh] = d.bbox;
  if (y0 < m.top) return '상단 테두리';
  if (y0 + bh > h - m.bottom) return '하단 테두리';
  if (x0 < m.left) return '좌측 테두리';
  if (x0 + bw > w - m.right) return '우측 테두리';
  return '표면';
}

export function fmtTime(iso: string) {
  const d = new Date(iso);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getMonth() + 1)}.${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
