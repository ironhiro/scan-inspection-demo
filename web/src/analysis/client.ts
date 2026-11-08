import type { Analysis } from './types';

let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

function get() {
  if (!worker) {
    worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const p = pending.get(e.data.id);
      if (!p) return;
      pending.delete(e.data.id);
      if (e.data.ok) p.resolve(e.data.result);
      else {
        if (e.data.stack) console.error('[analysis]', e.data.stack);
        p.reject(new Error(e.data.error));
      }
    };
  }
  return worker;
}

function call<T>(msg: object): Promise<T> {
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    get().postMessage({ id, ...msg });
  });
}

/** loads the model; resolves with the execution provider name */
export const warmup = () => call<string>({ type: 'warmup' });
export const analyze = (blob: Blob) => call<Analysis>({ type: 'analyze', blob });
