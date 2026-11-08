import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { MODEL_H, MODEL_W, TYPE_COLOR, type Box, type Damage, type DamageType, type Margins, type Pt } from '../analysis/types';

const rgb = (t: DamageType) => `rgb(${TYPE_COLOR[t].join(',')})`;
const MARGIN_COLOR = '#51cf66';

export interface Layers { mask: boolean; boxes: boolean; margins: boolean }

interface Props {
  mode: 'rect' | 'scan';
  rectUrl: string | null;
  scanUrl: string | null;
  rectW: number;
  rectH: number;
  scanSize: [number, number] | null;
  corners: Pt[] | null;
  labels: Uint16Array | null;
  damages: Damage[];
  margins: Margins | null;
  selected: string | null;
  layers: Layers;
  tool: 'select' | 'add';
  onSelect: (id: string | null) => void;
  onMargins: (m: Margins, commit: boolean) => void;
  onAdd: (box: Box) => void;
  fitSignal: number;
}

type Drag =
  | { kind: 'pan'; x: number; y: number; tx: number; ty: number }
  | { kind: 'margin'; side: keyof Margins }
  | { kind: 'draw'; x0: number; y0: number; x1: number; y1: number };

export default function Viewer(p: Props) {
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState({ s: 1, tx: 0, ty: 0 });
  const [drag, setDrag] = useState<Drag | null>(null);
  const viewRef = useRef(view);
  viewRef.current = view;

  const [W, H] = p.mode === 'rect' ? [p.rectW, p.rectH] : p.scanSize ?? [p.rectW, p.rectH];

  const fit = useCallback(() => {
    const el = stageRef.current;
    if (!el || !W || !H) return;
    const s = Math.min((el.clientWidth - 32) / W, (el.clientHeight - 32) / H);
    setView({ s, tx: (el.clientWidth - W * s) / 2, ty: (el.clientHeight - H * s) / 2 });
  }, [W, H]);
  useLayoutEffect(fit, [fit, p.mode, p.fitSignal]);
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => fit());
    ro.observe(el);
    return () => ro.disconnect();
  }, [fit]);

  // wheel zoom around the cursor
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const v = viewRef.current;
      const s = Math.min(12, Math.max(0.1, v.s * Math.exp(-e.deltaY * 0.0015)));
      setView({ s, tx: mx - ((mx - v.tx) * s) / v.s, ty: my - ((my - v.ty) * s) / v.s });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // damage mask overlay
  const compMap = useMemo(() => {
    const m = new Map<number, Damage>();
    for (const d of p.damages) if (d.comp) m.set(d.comp, d);
    return m;
  }, [p.damages]);
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !p.labels) return;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(MODEL_W, MODEL_H);
    const L = p.labels;
    for (let i = 0; i < L.length; i++) {
      const d = L[i] ? compMap.get(L[i]) : undefined;
      if (!d) continue;
      const col = TYPE_COLOR[d.type];
      const o = i * 4;
      img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2];
      img.data[o + 3] = d.id === p.selected ? 230 : 150;
    }
    ctx.putImageData(img, 0, 0);
  }, [p.labels, compMap, p.selected]);

  const toImage = (e: { clientX: number; clientY: number }) => {
    const r = stageRef.current!.getBoundingClientRect();
    return [(e.clientX - r.left - view.tx) / view.s, (e.clientY - r.top - view.ty) / view.s];
  };

  function onPointerDown(e: React.PointerEvent) {
    if (e.button !== 0 && e.button !== 1) return;
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); } catch { /* ignore */ }
    if (p.tool === 'add' && p.mode === 'rect' && e.button === 0) {
      const [x, y] = toImage(e);
      setDrag({ kind: 'draw', x0: x, y0: y, x1: x, y1: y });
    } else {
      setDrag({ kind: 'pan', x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty });
    }
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!drag) return;
    if (drag.kind === 'pan') {
      setView((v) => ({ ...v, tx: drag.tx + e.clientX - drag.x, ty: drag.ty + e.clientY - drag.y }));
    } else if (drag.kind === 'draw') {
      const [x, y] = toImage(e);
      setDrag({ ...drag, x1: x, y1: y });
    } else if (drag.kind === 'margin' && p.margins) {
      const [x, y] = toImage(e);
      const v = {
        left: x, right: p.rectW - x, top: y, bottom: p.rectH - y,
      }[drag.side];
      const lim = (drag.side === 'left' || drag.side === 'right' ? p.rectW : p.rectH) / 3;
      p.onMargins({ ...p.margins, [drag.side]: Math.round(Math.min(lim, Math.max(0, v))) }, false);
    }
  }

  function onPointerUp(e: React.PointerEvent) {
    if (drag?.kind === 'pan' && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 3) p.onSelect(null);
    if (drag?.kind === 'draw') {
      const x = Math.max(0, Math.min(drag.x0, drag.x1)), y = Math.max(0, Math.min(drag.y0, drag.y1));
      const w = Math.min(p.rectW, Math.max(drag.x0, drag.x1)) - x, h = Math.min(p.rectH, Math.max(drag.y0, drag.y1)) - y;
      if (w > 3 && h > 3) p.onAdd([x, y, w, h]);
    }
    if (drag?.kind === 'margin' && p.margins) p.onMargins(p.margins, true);
    setDrag(null);
  }

  const startMargin = (side: keyof Margins) => (e: React.PointerEvent) => {
    e.stopPropagation();
    setDrag({ kind: 'margin', side });
    try { stageRef.current!.setPointerCapture(e.pointerId); } catch { /* synthetic or released pointer */ }
  };

  const fs = 11 / view.s; // constant on-screen label size
  const url = p.mode === 'rect' ? p.rectUrl : p.scanUrl;
  const m = p.margins;

  return (
    <div
      ref={stageRef}
      className={`stage ${drag?.kind === 'pan' ? 'panning' : ''} ${p.tool === 'add' && p.mode === 'rect' ? 'drawing' : ''}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {url && (
        <div className="layer" style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.s})`, width: W, height: H }}>
          <img src={url} width={W} height={H} alt="" draggable={false} />
          {p.mode === 'rect' && (
            <canvas
              ref={canvasRef}
              width={MODEL_W}
              height={MODEL_H}
              style={{ width: W, height: H, display: p.layers.mask ? 'block' : 'none' }}
            />
          )}
          <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
            {p.mode === 'scan' && p.corners && (
              <polygon
                points={p.corners.map((c) => c.join(',')).join(' ')}
                fill="none" stroke={MARGIN_COLOR} strokeWidth={1.5} vectorEffect="non-scaling-stroke"
              />
            )}
            {p.mode === 'rect' && p.layers.margins && m && (
              <g>
                {(['left', 'right', 'top', 'bottom'] as const).map((side) => {
                  const vert = side === 'left' || side === 'right';
                  const pos = side === 'left' ? m.left : side === 'right' ? p.rectW - m.right : side === 'top' ? m.top : p.rectH - m.bottom;
                  const line = vert ? { x1: pos, x2: pos, y1: 0, y2: p.rectH } : { x1: 0, x2: p.rectW, y1: pos, y2: pos };
                  return (
                    <g key={side} style={{ cursor: vert ? 'ew-resize' : 'ns-resize' }} onPointerDown={startMargin(side)}>
                      <line {...line} stroke="transparent" strokeWidth={10} vectorEffect="non-scaling-stroke" />
                      <line {...line} stroke={MARGIN_COLOR} strokeWidth={1} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
                    </g>
                  );
                })}
              </g>
            )}
            {p.mode === 'rect' && p.layers.boxes &&
              p.damages.map((d) => {
                const [x, y, w, h] = d.bbox;
                const sel = d.id === p.selected;
                const pad = 3 / view.s;
                return (
                  <g key={d.id} style={{ cursor: 'pointer' }}
                     onPointerDown={(e) => { e.stopPropagation(); p.onSelect(d.id); }}>
                    <rect
                      x={x - pad} y={y - pad} width={w + 2 * pad} height={h + 2 * pad}
                      fill="transparent" stroke={rgb(d.type)} strokeWidth={sel ? 2.5 : 1.25}
                      strokeDasharray={d.source === 'inspector' ? '5 3' : undefined}
                      vectorEffect="non-scaling-stroke"
                    />
                    {sel && (
                      <text x={x - pad} y={y - pad - 3 / view.s} fontSize={fs} fill="#fff" stroke="#000" strokeWidth={3 / view.s} paintOrder="stroke">
                        {d.id}
                      </text>
                    )}
                  </g>
                );
              })}
            {drag?.kind === 'draw' && (
              <rect
                x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)}
                width={Math.abs(drag.x1 - drag.x0)} height={Math.abs(drag.y1 - drag.y0)}
                fill="rgba(255,255,255,0.08)" stroke="#fff" strokeWidth={1} strokeDasharray="4 3" vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>
        </div>
      )}
    </div>
  );
}
