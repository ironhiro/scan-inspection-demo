// Port of ml/pipeline.py (detect_card, rectify, measure_margins, components).
// Keep the two implementations in step.
import type { Margins, Pt } from './types';

export interface Img {
  data: Uint8ClampedArray; // RGBA
  width: number;
  height: number;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function medianColor(img: Img, rects: [number, number, number, number][]): [number, number, number] {
  const r: number[] = [], g: number[] = [], b: number[] = [];
  for (const [x0, y0, x1, y1] of rects) {
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = (y * img.width + x) * 4;
        r.push(img.data[i]); g.push(img.data[i + 1]); b.push(img.data[i + 2]);
      }
    }
  }
  return [median(r), median(g), median(b)];
}

/** least squares v = a*t + b with outlier rejection (mirrors _robust_line) */
function robustLine(t: number[], v: number[], iters = 3, tol = 2.5): [number, number] {
  let keep = t.map(() => true);
  let a = 0, b = median(v);
  for (let it = 0; it < iters; it++) {
    const idx = keep.flatMap((k, i) => (k ? [i] : []));
    if (idx.length < 10) break;
    let st = 0, sv = 0, stt = 0, stv = 0;
    for (const i of idx) { st += t[i]; sv += v[i]; stt += t[i] * t[i]; stv += t[i] * v[i]; }
    const n = idx.length;
    const den = n * stt - st * st;
    a = den === 0 ? 0 : (n * stv - st * sv) / den;
    b = (sv - a * st) / n;
    const res = t.map((ti, i) => Math.abs(v[i] - (a * ti + b)));
    const sorted = [...res].sort((x, y) => x - y);
    const p70 = sorted[Math.floor(0.7 * (sorted.length - 1))];
    const lim = Math.max(tol, p70);
    keep = res.map((r) => r < lim);
  }
  return [a, b];
}

/** 8-connected component labelling. Returns labels and per-label stats. */
export function connected(mask: Uint8Array, w: number, h: number) {
  const labels = new Int32Array(w * h);
  const stats: { x0: number; y0: number; x1: number; y1: number; area: number; sum: number }[] = [{ x0: 0, y0: 0, x1: 0, y1: 0, area: 0, sum: 0 }];
  const stack: number[] = [];
  let next = 1;
  for (let p = 0; p < w * h; p++) {
    if (!mask[p] || labels[p]) continue;
    const s = { x0: w, y0: h, x1: -1, y1: -1, area: 0, sum: 0 };
    labels[p] = next;
    stack.push(p);
    while (stack.length) {
      const q = stack.pop()!;
      const x = q % w, y = (q / w) | 0;
      s.area++;
      if (x < s.x0) s.x0 = x; if (x > s.x1) s.x1 = x;
      if (y < s.y0) s.y0 = y; if (y > s.y1) s.y1 = y;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const r = yy * w + xx;
          if (mask[r] && !labels[r]) { labels[r] = next; stack.push(r); }
        }
      }
    }
    stats.push(s);
    next++;
  }
  return { labels, stats, count: next - 1 };
}

export function detectCard(img: Img): Pt[] {
  const { width: w, height: h, data } = img;
  const bg = medianColor(img, [[0, 0, w, 6], [0, h - 6, w, h], [0, 0, 6, h], [w - 6, 0, w, h]]);
  const fg0 = new Uint8Array(w * h);
  for (let p = 0, i = 0; p < w * h; p++, i += 4) {
    const dr = data[i] - bg[0], dg = data[i + 1] - bg[1], db = data[i + 2] - bg[2];
    fg0[p] = dr * dr + dg * dg + db * db > 900 ? 1 : 0;
  }
  // 3x3 opening
  const er = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let ok = 1;
    for (let dy = -1; dy <= 1 && ok; dy++) for (let dx = -1; dx <= 1; dx++) if (!fg0[(y + dy) * w + x + dx]) { ok = 0; break; }
    er[y * w + x] = ok;
  }
  const fg = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let any = 0;
    for (let dy = -1; dy <= 1 && !any; dy++) for (let dx = -1; dx <= 1; dx++) if (er[(y + dy) * w + x + dx]) { any = 1; break; }
    fg[y * w + x] = any;
  }
  const { labels, stats, count } = connected(fg, w, h);
  if (!count) throw new Error('스캔 이미지에서 대상물을 찾지 못했습니다');
  // largest bounding box, not most pixels: a dark print frame can split the
  // border ring from the (larger) print area
  const boxArea = (i: number) => (stats[i].x1 - stats[i].x0 + 1) * (stats[i].y1 - stats[i].y0 + 1);
  let k = 1;
  for (let i = 2; i <= count; i++) if (boxArea(i) > boxArea(k)) k = i;
  const st = stats[k];
  const bw = st.x1 - st.x0 + 1, bh = st.y1 - st.y0 + 1;

  const ys: number[] = [], lx: number[] = [], rx: number[] = [];
  for (let y = Math.floor(st.y0 + 0.2 * bh); y < Math.floor(st.y0 + 0.8 * bh); y++) {
    let l = -1, r = -1;
    for (let x = 0; x < w; x++) if (labels[y * w + x] === k) { l = x; break; }
    if (l < 0) continue;
    for (let x = w - 1; x >= 0; x--) if (labels[y * w + x] === k) { r = x; break; }
    ys.push(y); lx.push(l); rx.push(r + 1);
  }
  const xs: number[] = [], ty: number[] = [], by: number[] = [];
  for (let x = Math.floor(st.x0 + 0.2 * bw); x < Math.floor(st.x0 + 0.8 * bw); x++) {
    let t = -1, b = -1;
    for (let y = 0; y < h; y++) if (labels[y * w + x] === k) { t = y; break; }
    if (t < 0) continue;
    for (let y = h - 1; y >= 0; y--) if (labels[y * w + x] === k) { b = y; break; }
    xs.push(x); ty.push(t); by.push(b + 1);
  }
  const L = robustLine(ys, lx), R = robustLine(ys, rx);
  const T = robustLine(xs, ty), B = robustLine(xs, by);
  const inter = (v: [number, number], hl: [number, number]): Pt => {
    const y = (hl[0] * v[1] + hl[1]) / (1 - v[0] * hl[0]);
    return [v[0] * y + v[1], y];
  };
  return [inter(L, T), inter(R, T), inter(R, B), inter(L, B)];
}

export function rectSize(c: Pt[]): [number, number] {
  const d = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  return [Math.round((d(c[0], c[1]) + d(c[3], c[2])) / 2), Math.round((d(c[0], c[3]) + d(c[1], c[2])) / 2)];
}

/** homography mapping dst rectangle (0,0)-(w,h) back to the source quad */
function inverseHomography(c: Pt[], w: number, h: number): number[] {
  // solve H: dst -> src for the 4 correspondences (8x8 linear system)
  const dst: Pt[] = [[0, 0], [w, 0], [w, h], [0, h]];
  const A: number[][] = [];
  const bv: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = dst[i], [u, v] = c[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); bv.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); bv.push(v);
  }
  // gaussian elimination
  for (let i = 0; i < 8; i++) {
    let piv = i;
    for (let r = i + 1; r < 8; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    [A[i], A[piv]] = [A[piv], A[i]]; [bv[i], bv[piv]] = [bv[piv], bv[i]];
    for (let r = 0; r < 8; r++) {
      if (r === i) continue;
      const f = A[r][i] / A[i][i];
      for (let k = i; k < 8; k++) A[r][k] -= f * A[i][k];
      bv[r] -= f * bv[i];
    }
  }
  return [...bv.map((v, i) => v / A[i][i]), 1];
}

/** perspective warp with bilinear sampling and replicated borders (cv2.warpPerspective equivalent) */
export function rectify(img: Img, c: Pt[], w: number, h: number): Img {
  const H = inverseHomography(c, w, h);
  const out = new Uint8ClampedArray(w * h * 4);
  const sw = img.width, sh = img.height, src = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const z = H[6] * x + H[7] * y + H[8];
      let u = (H[0] * x + H[1] * y + H[2]) / z;
      let v = (H[3] * x + H[4] * y + H[5]) / z;
      u = Math.min(Math.max(u, 0), sw - 1.001);
      v = Math.min(Math.max(v, 0), sh - 1.001);
      const x0 = u | 0, y0 = v | 0, fx = u - x0, fy = v - y0;
      const i00 = (y0 * sw + x0) * 4, i10 = i00 + 4, i01 = i00 + sw * 4, i11 = i01 + 4;
      const o = (y * w + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const a = src[i00 + ch] * (1 - fx) + src[i10 + ch] * fx;
        const b = src[i01 + ch] * (1 - fx) + src[i11 + ch] * fx;
        out[o + ch] = a * (1 - fy) + b * fy;
      }
      out[o + 3] = 255;
    }
  }
  return { data: out, width: w, height: h };
}

export function measureMargins(img: Img, thr = 40, start = 4): Margins {
  const { width: w, height: h, data } = img;
  const bc = medianColor(img, [
    [Math.floor(w * 0.3), 5, Math.floor(w * 0.7), 10],
    [Math.floor(w * 0.3), h - 10, Math.floor(w * 0.7), h - 5],
    [5, Math.floor(h * 0.3), 10, Math.floor(h * 0.7)],
    [w - 10, Math.floor(h * 0.3), w - 5, Math.floor(h * 0.7)],
  ]);
  const t2 = thr * thr;
  const off = (x: number, y: number) => {
    const i = (y * w + x) * 4;
    const dr = data[i] - bc[0], dg = data[i + 1] - bc[1], db = data[i + 2] - bc[2];
    return dr * dr + dg * dg + db * db > t2 ? 1 : 0;
  };
  const xa = Math.floor(w * 0.2), xb = Math.floor(w * 0.8);
  const ya = Math.floor(h * 0.2), yb = Math.floor(h * 0.8);
  const scanRows = (rowAt: (d: number) => number) => {
    for (let d = start; d < Math.floor(h / 3); d++) {
      const y = rowAt(d);
      let s = 0;
      for (let x = xa; x < xb; x++) s += off(x, y);
      if (s / (xb - xa) > 0.5) return d;
    }
    return -1;
  };
  const scanCols = (colAt: (d: number) => number) => {
    for (let d = start; d < Math.floor(w / 3); d++) {
      const x = colAt(d);
      let s = 0;
      for (let y = ya; y < yb; y++) s += off(x, y);
      if (s / (yb - ya) > 0.5) return d;
    }
    return -1;
  };
  return {
    top: scanRows((d) => d),
    bottom: scanRows((d) => h - 1 - d),
    left: scanCols((d) => d),
    right: scanCols((d) => w - 1 - d),
  };
}
