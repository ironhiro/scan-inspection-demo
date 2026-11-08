// Mirrors pipeline.grade(); rules come from public/grading_rules.json.
import type { Damage, DamageType, Margins } from './types';

interface Step { max: number; deduct: number }
export interface Rules {
  name: string;
  note: string;
  card_width_mm: number;
  centering: Step[];
  damage: Record<DamageType, { measure: 'area_mm2' | 'length_mm'; aggregate?: 'sum'; steps: Step[]; cap?: number }>;
}

export interface GradeItem {
  reason: 'centering' | DamageType;
  /** set for per-instance items; summed items have count > 1 instead */
  damageId?: string;
  count?: number;
  value: number;
  unit: string;
  deduct: number;
}

export interface GradeResult {
  grade: number;
  items: GradeItem[];
  cap: number;
  capReason?: DamageType;
}

let cached: Promise<Rules> | null = null;
export function loadRules(): Promise<Rules> {
  cached ??= fetch(`${import.meta.env.BASE_URL}grading_rules.json`).then((r) => r.json());
  return cached;
}

export function centering(m: Margins) {
  const lr = (100 * m.left) / Math.max(1, m.left + m.right);
  const tb = (100 * m.top) / Math.max(1, m.top + m.bottom);
  return { lr, tb, worst: Math.max(lr, 100 - lr, tb, 100 - tb) };
}

export function measureOf(d: Damage, rules: Rules, cardWpx: number) {
  const mm = rules.card_width_mm / cardWpx;
  const r = rules.damage[d.type];
  return r.measure === 'area_mm2'
    ? { value: d.area * mm * mm, unit: 'mm²' }
    : { value: Math.hypot(d.bbox[2], d.bbox[3]) * mm, unit: 'mm' };
}

export function grade(damages: Damage[], margins: Margins, cardWpx: number, rules: Rules): GradeResult {
  const items: GradeItem[] = [];
  const { worst } = centering(margins);
  for (const s of rules.centering) {
    if (worst <= s.max) {
      if (s.deduct) items.push({ reason: 'centering', value: worst, unit: '%', deduct: s.deduct });
      break;
    }
  }
  let cap = 10;
  let capReason: DamageType | undefined;
  const measured: { type: DamageType; damageId?: string; value: number; unit: string; count: number }[] = [];
  // aggregate: 'sum' types get one deduction per card from the summed measure, so the
  // grade does not depend on how one stretch of wear happens to split into pieces
  const sums = new Map<DamageType, { value: number; unit: string; count: number; id: string }>();
  for (const d of damages) {
    const r = rules.damage[d.type];
    const { value, unit } = measureOf(d, rules, cardWpx);
    if (r.aggregate === 'sum') {
      const s = sums.get(d.type) ?? { value: 0, unit, count: 0, id: d.id };
      s.value += value;
      s.count += 1;
      sums.set(d.type, s);
    } else {
      measured.push({ type: d.type, damageId: d.id, value, unit, count: 1 });
    }
    if (r.cap !== undefined && r.cap < cap) { cap = r.cap; capReason = d.type; }
  }
  for (const [type, s] of sums) {
    measured.push({ type, damageId: s.count === 1 ? s.id : undefined, value: s.value, unit: s.unit, count: s.count });
  }
  for (const m of measured) {
    for (const s of rules.damage[m.type].steps) {
      if (m.value <= s.max) {
        items.push({ reason: m.type, damageId: m.damageId, count: m.count, value: m.value, unit: m.unit, deduct: s.deduct });
        break;
      }
    }
  }
  const total = items.reduce((a, i) => a + i.deduct, 0);
  return { grade: Math.max(1, Math.min(cap, Math.floor(10 - total))), items, cap, capReason };
}
