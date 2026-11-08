"""Classical-CV part of the pipeline: card detection, rectification, margin
measurement, and post-processing of the segmentation output.

The browser implementation (web/src/analysis/*.ts) mirrors this file; keep
the two in sync. Parity is checked by ml/eval.py against the same test scans.
"""
import json
import os

import cv2
import numpy as np

MODEL_W, MODEL_H = 448, 624
CLASS_NAMES = ["none", "corner_wear", "edge_wear", "scratch", "crease"]
RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "public", "grading_rules.json")


# ----------------------------------------------------------------------------
# detection / rectification
# ----------------------------------------------------------------------------

def _robust_line(t, v, iters=3, tol=2.5):
    """Fit v = a*t + b, dropping outliers (corner rounding, wear, dust)."""
    keep = np.ones(len(t), bool)
    a, b = 0.0, float(np.median(v))
    for _ in range(iters):
        if keep.sum() < 10:
            break
        a, b = np.polyfit(t[keep], v[keep], 1)
        res = np.abs(v - (a * t + b))
        keep = res < max(tol, np.percentile(res, 70))
    return a, b


def detect_card(scan):
    """Return 4 corners (TL, TR, BR, BL) of the card's bounding rectangle."""
    img = scan.astype(np.float32)
    h, w = img.shape[:2]
    edge = np.concatenate([img[:6].reshape(-1, 3), img[-6:].reshape(-1, 3),
                           img[:, :6].reshape(-1, 3), img[:, -6:].reshape(-1, 3)])
    bg = np.median(edge, 0)
    diff = np.linalg.norm(img - bg, axis=2)
    fg = (diff > 30).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n < 2:
        raise ValueError("card not found")
    # pick the component with the largest bounding box, not the most pixels:
    # a dark print frame can split the border ring from the (larger) print area
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]))
    x0, y0, bw, bh = stats[k, :4]
    m = lab == k

    ys = np.arange(int(y0 + 0.2 * bh), int(y0 + 0.8 * bh))
    rows = m[ys]
    has = rows.any(1)
    ys = ys[has]
    lx = rows[has].argmax(1).astype(np.float32)
    rx = (w - 1 - rows[has][:, ::-1].argmax(1)).astype(np.float32) + 1

    xs = np.arange(int(x0 + 0.2 * bw), int(x0 + 0.8 * bw))
    cols = m[:, xs]
    hasc = cols.any(0)
    xs = xs[hasc]
    ty = cols[:, hasc].argmax(0).astype(np.float32)
    by = (h - 1 - cols[::-1, hasc].argmax(0)).astype(np.float32) + 1

    L = _robust_line(ys.astype(np.float32), lx)   # x = a*y + b
    R = _robust_line(ys.astype(np.float32), rx)
    T = _robust_line(xs.astype(np.float32), ty)   # y = a*x + b
    B = _robust_line(xs.astype(np.float32), by)

    def inter(vline, hline):
        a1, b1 = vline   # x = a1*y + b1
        a2, b2 = hline   # y = a2*x + b2
        y = (a2 * b1 + b2) / (1 - a1 * a2)
        return [a1 * y + b1, y]

    return np.array([inter(L, T), inter(R, T), inter(R, B), inter(L, B)], np.float32)


def rect_size(corners):
    wt = np.linalg.norm(corners[1] - corners[0])
    wb = np.linalg.norm(corners[2] - corners[3])
    hl = np.linalg.norm(corners[3] - corners[0])
    hr = np.linalg.norm(corners[2] - corners[1])
    return int(round((wt + wb) / 2)), int(round((hl + hr) / 2))


def rectify(scan, corners, size=None, interp=cv2.INTER_LINEAR):
    w, h = size or rect_size(corners)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    Hm = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(scan, Hm, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------
# margins
# ----------------------------------------------------------------------------

def measure_margins(card, thr=40.0, start=4):
    """Distance from each outer edge to the first row/col that is mostly not
    border colour. card: rectified BGR image of the whole card."""
    img = card.astype(np.float32)
    h, w = img.shape[:2]
    strips = [img[5:10, int(w * .3):int(w * .7)], img[h - 10:h - 5, int(w * .3):int(w * .7)],
              img[int(h * .3):int(h * .7), 5:10], img[int(h * .3):int(h * .7), w - 10:w - 5]]
    bc = np.median(np.concatenate([s.reshape(-1, 3) for s in strips]), 0)
    off = np.linalg.norm(img - bc, axis=2) > thr

    def first(profile):
        idx = np.nonzero(profile[start:] > 0.5)[0]
        return int(idx[0] + start) if len(idx) else -1

    xs = slice(int(w * .2), int(w * .8))
    ys = slice(int(h * .2), int(h * .8))
    top = first(off[: h // 3, xs].mean(1))
    bottom = first(off[::-1][: h // 3, xs].mean(1))
    left = first(off[ys, : w // 3].mean(0))
    right = first(off[ys, ::-1][:, : w // 3].mean(0))
    return {"left": left, "right": right, "top": top, "bottom": bottom,
            "border_color": [float(v) for v in bc]}


def centering(m):
    lr = 100 * m["left"] / max(1, m["left"] + m["right"])
    tb = 100 * m["top"] / max(1, m["top"] + m["bottom"])
    return lr, tb


# ----------------------------------------------------------------------------
# segmentation post-processing + grading
# ----------------------------------------------------------------------------

MIN_AREA = {1: 10, 2: 8, 3: 10, 4: 25}   # in model-resolution px


def components(classmap, prob=None):
    """classmap: (MODEL_H, MODEL_W) int; returns damage instances in model px."""
    out = []
    for c in range(1, len(CLASS_NAMES)):
        m = (classmap == c).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if area < MIN_AREA[c]:
                continue
            conf = float(prob[lab == k].mean()) if prob is not None else 1.0
            out.append({"cls": c, "type": CLASS_NAMES[c], "bbox": [int(x), int(y), int(bw), int(bh)],
                        "area": int(area), "conf": round(conf, 3)})
    return out


def load_rules():
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


def grade(instances, margins, card_w_px, rules=None):
    """instances use the same pixel space as card_w_px."""
    rules = rules or load_rules()
    mm = rules["card_width_mm"] / card_w_px
    items = []
    lr, tb = centering(margins)
    worst = max(lr, 100 - lr, tb, 100 - tb)
    for step in rules["centering"]:
        if worst <= step["max"]:
            if step["deduct"]:
                items.append({"reason": "centering", "value": round(worst, 1), "deduct": step["deduct"]})
            break
    cap = 10
    for d in instances:
        r = rules["damage"][d["type"]]
        x, y, bw, bh = d["bbox"]
        if r["measure"] == "area_mm2":
            v = d["area"] * mm * mm
        else:
            v = float(np.hypot(bw, bh)) * mm
        for step in r["steps"]:
            if v <= step["max"]:
                items.append({"reason": d["type"], "value": round(v, 2), "deduct": step["deduct"]})
                break
        cap = min(cap, r.get("cap", 10))
    total = sum(i["deduct"] for i in items)
    g = int(max(1, min(cap, np.floor(10 - total))))
    return {"grade": g, "items": items, "cap": cap}
