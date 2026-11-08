/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import { connected, detectCard, measureMargins, rectify, rectSize, type Img } from './cv';
import { DAMAGE_TYPES, MODEL_H, MODEL_W, type Analysis, type Damage } from './types';

const MODEL_VERSION = 'damage-unet / synth-v1';
const MIN_AREA = [0, 10, 8, 10, 25]; // model px, same as pipeline.MIN_AREA
const MAX_SIDE = 2400;

let session: Promise<{ s: ort.InferenceSession; provider: string }> | null = null;

function getSession() {
  session ??= (async () => {
    const url = `${self.location.origin}${import.meta.env.BASE_URL}models/damage_unet.onnx`;
    const buf = await (await fetch(url)).arrayBuffer();
    // each attempt gets its own copy: a failed create can leave the buffer detached
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (gpu && (await gpu.requestAdapter().catch(() => null))) {
      try {
        const s = await ort.InferenceSession.create(new Uint8Array(buf.slice(0)), { executionProviders: ['webgpu'] });
        // a session can be created yet fail on first run; verify once
        await s.run({ image: new ort.Tensor('float32', new Float32Array(3 * MODEL_W * MODEL_H), [1, 3, MODEL_H, MODEL_W]) });
        return { s, provider: 'WebGPU' };
      } catch (e) {
        console.warn('WebGPU unavailable, using WASM', e);
      }
    }
    const s = await ort.InferenceSession.create(new Uint8Array(buf.slice(0)), { executionProviders: ['wasm'] });
    return { s, provider: 'WASM (CPU)' };
  })();
  return session;
}

async function decode(blob: Blob): Promise<Img> {
  const bmp = await createImageBitmap(blob);
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * k), h = Math.round(bmp.height * k);
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h);
  return { data: d.data, width: w, height: h };
}

async function analyze(blob: Blob): Promise<Analysis> {
  const t0 = performance.now();
  const img = await decode(blob);
  const corners = detectCard(img);
  const [rw, rh] = rectSize(corners);
  const rect = rectify(img, corners, rw, rh);
  const margins = measureMargins(rect);
  const small = rectify(img, corners, MODEL_W, MODEL_H);
  const t1 = performance.now();

  const n = MODEL_W * MODEL_H;
  const input = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) {
    input[p] = small.data[p * 4] / 255 - 0.5;
    input[n + p] = small.data[p * 4 + 1] / 255 - 0.5;
    input[2 * n + p] = small.data[p * 4 + 2] / 255 - 0.5;
  }
  const { s, provider } = await getSession();
  const out = await s.run({ image: new ort.Tensor('float32', input, [1, 3, MODEL_H, MODEL_W]) });
  const cls = out.cls.data as Int32Array;
  const conf = out.conf.data as Float32Array;
  const t2 = performance.now();

  const sx = rw / MODEL_W, sy = rh / MODEL_H;
  const labels = new Uint16Array(n);
  const damages: Damage[] = [];
  let comp = 0;
  DAMAGE_TYPES.forEach((type, ti) => {
    const c = ti + 1;
    const m = new Uint8Array(n);
    for (let p = 0; p < n; p++) m[p] = cls[p] === c ? 1 : 0;
    const cc = connected(m, MODEL_W, MODEL_H);
    const sums = new Float64Array(cc.count + 1);
    for (let p = 0; p < n; p++) if (cc.labels[p]) sums[cc.labels[p]] += conf[p];
    const remap = new Uint16Array(cc.count + 1);
    for (let k = 1; k <= cc.count; k++) {
      const st = cc.stats[k];
      if (st.area < MIN_AREA[c]) continue;
      remap[k] = ++comp;
      damages.push({
        id: `ai-${comp}`,
        type,
        bbox: [st.x0 * sx, st.y0 * sy, (st.x1 - st.x0 + 1) * sx, (st.y1 - st.y0 + 1) * sy],
        area: st.area * sx * sy,
        conf: sums[k] / st.area,
        source: 'ai',
        comp,
      });
    }
    for (let p = 0; p < n; p++) if (cc.labels[p] && remap[cc.labels[p]]) labels[p] = remap[cc.labels[p]];
  });

  const canvas = new OffscreenCanvas(rw, rh);
  canvas.getContext('2d')!.putImageData(new ImageData(rect.data, rw, rh), 0, 0);
  const rectified = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 });

  return {
    modelVersion: MODEL_VERSION,
    provider,
    elapsedMs: { detect: t1 - t0, infer: t2 - t1, total: performance.now() - t0 },
    corners,
    rectW: rw,
    rectH: rh,
    margins,
    damages,
    labels,
    rectified,
  };
}

// an InferenceSession cannot run concurrently: process requests one at a time
let queue: Promise<void> = Promise.resolve();
self.onmessage = (e: MessageEvent) => {
  queue = queue.then(() => handle(e));
};

async function handle(e: MessageEvent) {
  const { id, type, blob } = e.data;
  try {
    if (type === 'warmup') {
      const { provider } = await getSession();
      self.postMessage({ id, ok: true, result: provider });
    } else {
      const result = await analyze(blob);
      self.postMessage({ id, ok: true, result }, [result.labels.buffer]);
    }
  } catch (err) {
    console.error(err);
    self.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err), stack: err instanceof Error ? err.stack : undefined });
  }
}
