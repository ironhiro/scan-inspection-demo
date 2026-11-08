// Case storage. The demo keeps everything in the browser (IndexedDB) so it can
// run on static hosting; server/ has the same contract as a FastAPI service.
import type { Analysis, Damage, Margins } from '../analysis/types';

export type Status = 'new' | 'analyzed' | 'reviewing' | 'done';

export const STATUS_LABEL: Record<Status, string> = {
  new: '분석 전',
  analyzed: '검토 대기',
  reviewing: '검토 중',
  done: '검사 완료',
};

export interface Review {
  damages: Damage[];
  margins: Margins;
  gradeOverride: number | null;
  note: string;
}

export interface LogEntry {
  at: string;
  text: string;
}

export interface CaseRecord {
  id: string;
  seq: number;
  createdAt: string;
  source: 'sample' | 'upload';
  fileName: string;
  image: Blob;
  status: Status;
  ai: Analysis | null;
  review: Review | null;
  completedAt: string | null;
  log: LogEntry[];
}

export interface CaseRepository {
  list(): Promise<CaseRecord[]>;
  get(id: string): Promise<CaseRecord | undefined>;
  put(rec: CaseRecord): Promise<void>;
  create(image: Blob, fileName: string, source: CaseRecord['source']): Promise<CaseRecord>;
  reset(): Promise<void>;
}

const DB = 'scan-inspection-demo';
const STORE = 'cases';
const SEED_KEY = 'seed-v1';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => {
      r.result.createObjectStore(STORE, { keyPath: 'id' });
      r.result.createObjectStore('meta');
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode);
        const req = fn(t.objectStore(store));
        t.oncomplete = () => resolve(req ? req.result : (undefined as T));
        t.onerror = () => reject(t.error);
      }),
  );
}

export const now = () => new Date().toISOString();

class IdbRepository implements CaseRepository {
  private seeding: Promise<void> | null = null;

  private seed() {
    this.seeding ??= (async () => {
      const done = await tx<string | undefined>('meta', 'readonly', (s) => s.get(SEED_KEY));
      if (done) return;
      const base = import.meta.env.BASE_URL;
      const manifest: { file: string; label: string }[] = await fetch(`${base}samples/manifest.json`).then((r) => r.json());
      const day = Date.now() - 3 * 86400_000;
      for (let i = 0; i < manifest.length; i++) {
        const blob = await fetch(`${base}samples/${manifest[i].file}`).then((r) => r.blob());
        const at = new Date(day + i * 47 * 60_000).toISOString();
        await this.insert(blob, manifest[i].file, 'sample', i + 1, at);
      }
      await tx('meta', 'readwrite', (s) => s.put('1', SEED_KEY));
    })();
    return this.seeding;
  }

  private async insert(image: Blob, fileName: string, source: CaseRecord['source'], seq: number, at = now()) {
    const rec: CaseRecord = {
      id: `SC-${String(seq).padStart(4, '0')}`,
      seq,
      createdAt: at,
      source,
      fileName,
      image,
      status: 'new',
      ai: null,
      review: null,
      completedAt: null,
      log: [{ at, text: source === 'sample' ? '기존 DB에서 스캔 이미지 불러옴' : '이미지 등록' }],
    };
    await tx(STORE, 'readwrite', (s) => s.put(rec));
    return rec;
  }

  async list() {
    await this.seed();
    const all = await tx<CaseRecord[]>(STORE, 'readonly', (s) => s.getAll());
    return all.sort((a, b) => b.seq - a.seq);
  }

  async get(id: string) {
    await this.seed();
    return tx<CaseRecord | undefined>(STORE, 'readonly', (s) => s.get(id));
  }

  put(rec: CaseRecord) {
    return tx<void>(STORE, 'readwrite', (s) => {
      s.put(rec);
    });
  }

  async create(image: Blob, fileName: string, source: CaseRecord['source']) {
    const all = await this.list();
    const seq = all.reduce((m, r) => Math.max(m, r.seq), 0) + 1;
    return this.insert(image, fileName, source, seq);
  }

  async reset() {
    await tx(STORE, 'readwrite', (s) => {
      s.clear();
    });
    await tx('meta', 'readwrite', (s) => {
      s.delete(SEED_KEY);
    });
    this.seeding = null;
  }
}

export const repo: CaseRepository = new IdbRepository();
