export const MODEL_W = 448;
export const MODEL_H = 624;

export const DAMAGE_TYPES = ['corner_wear', 'edge_wear', 'scratch', 'crease'] as const;
export type DamageType = (typeof DAMAGE_TYPES)[number];

export const DAMAGE_LABEL: Record<DamageType, string> = {
  corner_wear: '모서리 마모',
  edge_wear: '테두리 마모',
  scratch: '표면 긁힘',
  crease: '구김·눌림',
};

export const TYPE_COLOR: Record<DamageType, [number, number, number]> = {
  corner_wear: [247, 103, 7],
  edge_wear: [214, 51, 108],
  scratch: [21, 170, 191],
  crease: [250, 176, 5],
};

/** class index in the model output (0 = none) */
export const CLASS_INDEX: Record<DamageType, number> = {
  corner_wear: 1,
  edge_wear: 2,
  scratch: 3,
  crease: 4,
};

export type Pt = [number, number];
export type Box = [number, number, number, number]; // x, y, w, h in rectified px

export interface Margins {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface Damage {
  id: string;
  type: DamageType;
  bbox: Box;
  /** pixel area in rectified px (mask area for AI items, box area for manual ones) */
  area: number;
  conf: number | null;
  source: 'ai' | 'inspector';
  /** component label in the stored label map, AI items only */
  comp?: number;
}

export interface Analysis {
  modelVersion: string;
  provider: string;
  elapsedMs: { detect: number; infer: number; total: number };
  corners: Pt[];
  rectW: number;
  rectH: number;
  margins: Margins;
  damages: Damage[];
  /** component label map at MODEL_W x MODEL_H (0 = none) */
  labels: Uint16Array;
  /** rectified image, JPEG */
  rectified: Blob;
}
