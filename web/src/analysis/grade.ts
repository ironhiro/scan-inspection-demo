// Mirrors pipeline.grade(); rules come from public/grading_rules.json.
import type { Damage, DamageType, Margins } from './types';

interface Step { max: number; deduct: number }
export interface Rules {
  name: string;
  note: string;
  card_width_mm: number;
  centering: Step[];
  damage: Record<DamageType, { measure: 'area_mm2' | 'length_mm'; steps: Step[]; cap?: number }>;
}

export interface GradeItem {
  reason: 'centering' | DamageType;
  damageId?: string;
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
  for (const d of damages) {
    const r = rules.damage[d.type];
    const { value, unit } = measureOf(d, rules, cardWpx);
    for (const s of r.steps) {
      if (value <= s.max) {
        items.push({ reason: d.type, damageId: d.id, value, unit, deduct: s.deduct });
        break;
      }
    }
    if (r.cap !== undefined && r.cap < cap) { cap = r.cap; capReason = d.type; }
  }
  const total = items.reduce((a, i) => a + i.deduct, 0);
  return { grade: Math.max(1, Math.min(cap, Math.floor(10 - total))), items, cap, capReason };
}
