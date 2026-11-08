"""Synthetic card-scan generator.

Renders a card-shaped printed item with a border and an inner print area,
adds four damage types with pixel-exact masks, then simulates a flatbed scan
(background, small rotation, blur, noise, JPEG).

Classes: 0 background/none, 1 corner_wear, 2 edge_wear, 3 scratch, 4 crease
"""
import math

import cv2
import numpy as np

CARD_W, CARD_H = 630, 880          # ideal card size in px (~254 dpi for 63x88 mm)
CLASSES = ["none", "corner_wear", "edge_wear", "scratch", "crease"]

BORDER_PALETTE = [  # BGR
    (232, 234, 236), (240, 240, 238), (38, 38, 42), (52, 48, 46),
    (40, 175, 228), (196, 198, 202), (150, 70, 35), (45, 45, 165),
    (70, 125, 45), (120, 40, 110),
]
PAPER_CORE = np.array([238, 240, 241], np.float32)  # whitened paper fibre colour


def _color_dist(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def _rand_color(rng, far_from=None, min_dist=0.0):
    for _ in range(100):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        if far_from is None or _color_dist(c, far_from) >= min_dist:
            return c
    return tuple(255 - v for v in far_from)


def smooth_noise(rng, h, w, grid):
    """Low-frequency noise in [0,1] at (h,w)."""
    gh, gw = max(2, int(h / grid)), max(2, int(w / grid))
    n = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def rounded_rect_mask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    r = int(r)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    return m


# ----------------------------------------------------------------------------
# card artwork
# ----------------------------------------------------------------------------

def _draw_art(rng, img, x0, y0, x1, y1, border_color):
    h, w = y1 - y0, x1 - x0
    c1 = _rand_color(rng, border_color, 70)
    c2 = _rand_color(rng)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    area = (np.array(c1, np.float32) * (1 - t) + np.array(c2, np.float32) * t)
    area = np.repeat(area, w, axis=1)

    layer = area.copy()
    for _ in range(int(rng.integers(5, 15))):
        col = _rand_color(rng)
        ov = layer.copy()
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(ov, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(rng.integers(10, w // 3)), col, -1, cv2.LINE_AA)
        elif kind == 1:
            pts = np.stack([rng.integers(0, w, 5), rng.integers(0, h, 5)], 1).astype(np.int32)
            cv2.fillPoly(ov, [pts], col, cv2.LINE_AA)
        else:
            cv2.ellipse(ov, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                        (int(rng.integers(10, w // 2)), int(rng.integers(5, h // 6))),
                        float(rng.uniform(0, 180)), 0, 360, col, -1, cv2.LINE_AA)
        a = float(rng.uniform(0.3, 0.8))
        layer = layer * (1 - a) + ov * a

    # art window with texture
    wx0, wy0 = int(w * 0.07), int(h * rng.uniform(0.09, 0.13))
    wx1, wy1 = w - wx0, int(h * rng.uniform(0.52, 0.6))
    sc = rng.uniform(15, 60)
    lum = smooth_noise(rng, wy1 - wy0, wx1 - wx0, sc)[..., None]
    tint = np.stack([smooth_noise(rng, wy1 - wy0, wx1 - wx0, sc * 3) for _ in range(3)], -1)
    base = np.array(_rand_color(rng), np.float32)
    layer[wy0:wy1, wx0:wx1] = (base * 0.45 + lum * 180 * 0.35 + tint * 255 * 0.2)
    cv2.rectangle(layer, (wx0, wy0), (wx1, wy1), _rand_color(rng), int(rng.integers(2, 5)))

    # name bar
    nb_col = _rand_color(rng)
    cv2.rectangle(layer, (wx0, int(h * 0.025)), (wx1, int(h * 0.075)), nb_col, -1)
    ink = (20, 20, 20) if sum(nb_col) > 380 else (235, 235, 235)
    x = wx0 + 8
    while x < wx0 + (wx1 - wx0) * rng.uniform(0.4, 0.75):
        gw = int(rng.integers(5, 12))
        cv2.rectangle(layer, (x, int(h * 0.038)), (x + gw, int(h * 0.062)), ink, -1)
        x += gw + int(rng.integers(2, 8))

    # text lines
    ty = wy1 + int(h * 0.05)
    ink = (25, 25, 25) if layer[min(ty, h - 1), w // 2].sum() > 380 else (230, 230, 230)
    while ty < h * 0.93:
        x, xe = wx0, wx0 + int((wx1 - wx0) * rng.uniform(0.5, 1.0))
        while x < xe:
            gw = int(rng.integers(3, 30))
            cv2.rectangle(layer, (x, ty), (min(x + gw, xe), ty + 7), ink, -1)
            x += gw + int(rng.integers(3, 7))
        ty += int(rng.integers(14, 26))

    img[y0:y1, x0:x1] = layer.clip(0, 255)

    # print-area frame line (always contrasts with the border colour)
    frame = _rand_color(rng, border_color, 110)
    th = int(rng.integers(2, 5))
    o = th // 2  # keep the stroke inside the print area so its outer edge == x0/y0
    cv2.rectangle(img, (x0 + o, y0 + o), (x1 - 1 - o, y1 - 1 - o), frame, th)


def render_card(rng):
    W, H = CARD_W, CARD_H
    border = np.array(BORDER_PALETTE[int(rng.integers(len(BORDER_PALETTE)))], np.float32)
    border = (border + rng.normal(0, 6, 3)).clip(0, 255)
    img = np.empty((H, W, 3), np.float32)
    img[:] = border

    m = rng.uniform(0.045, 0.068) * W
    mv = m * rng.uniform(0.9, 1.2)
    ox = float(np.clip(rng.normal(0, 0.22), -0.6, 0.6))
    oy = float(np.clip(rng.normal(0, 0.22), -0.6, 0.6))
    left, right = int(round(m * (1 + ox))), int(round(m * (1 - ox)))
    top, bottom = int(round(mv * (1 + oy))), int(round(mv * (1 - oy)))
    _draw_art(rng, img, left, top, W - right, H - bottom, tuple(border))

    img += rng.normal(0, 2.0, img.shape)  # paper/print grain
    radius = rng.uniform(20, 30)
    alpha = rounded_rect_mask(W, H, radius)
    mask = np.zeros((H, W), np.uint8)

    # pad with zeros so the image border counts as "outside" the card
    inside = np.pad((alpha > 127).astype(np.uint8), 1)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)[1:-1, 1:-1]
    border_bright = border.mean() > 150

    # order matters: later damage overwrites earlier labels
    if rng.random() < 0.16:
        _crease(rng, img, mask, alpha)
    for _ in range(int(rng.poisson(1.2))):
        _scratch(rng, img, mask, alpha, dist)
    for _ in range(int(rng.poisson(1.5))):
        _edge_wear(rng, img, mask, alpha, dist, border_bright, radius)
    for corner in range(4):
        if rng.random() < 0.38:
            _corner_wear(rng, img, mask, alpha, dist, corner, radius, border_bright)

    return {
        "img": img.clip(0, 255).astype(np.uint8),
        "alpha": alpha,
        "mask": mask,
        "margins": {"left": left, "right": right, "top": top, "bottom": bottom},
        "radius": radius,
    }


# ----------------------------------------------------------------------------
# damage
# ----------------------------------------------------------------------------

def _whiten(img, region, strength, rng, bright_base):
    """Paper-fibre whitening on dark borders, grey grime/fray on bright ones."""
    s = (strength * region)[..., None]
    if bright_base:
        target = img * 0.55 + rng.normal(0, 8, img.shape)
    else:
        target = PAPER_CORE + rng.normal(0, 10, img.shape)
    img[:] = img * (1 - s) + target * s


def _corner_wear(rng, img, mask, alpha, dist, corner, radius, bright):
    H, W = mask.shape
    size = int(radius + rng.uniform(8, 40))
    depth = rng.uniform(3, 13)
    yy, xx = np.mgrid[0:H, 0:W]
    cx = 0 if corner in (0, 3) else W - 1
    cy = 0 if corner in (0, 1) else H - 1
    box = (np.abs(xx - cx) < size) & (np.abs(yy - cy) < size)
    n = smooth_noise(rng, H, W, 18) * 0.6 + smooth_noise(rng, H, W, 4) * 0.4
    thr = depth * (0.35 + 0.9 * n)
    region = box & (alpha > 127) & (dist < thr)
    if region.sum() < 15:
        return
    _whiten(img, region.astype(np.float32), rng.uniform(0.7, 1.0), rng, bright)
    loss = region & (dist < thr * rng.uniform(0.0, 0.35))
    alpha[loss] = 0
    mask[region] = 1


def _edge_wear(rng, img, mask, alpha, dist, bright, radius):
    H, W = mask.shape
    side = int(rng.integers(4))
    length = rng.uniform(15, 160)
    depth = rng.uniform(2, 8)
    span = (W if side in (0, 2) else H) - 2 * (radius + 25)
    start = radius + 25 + rng.uniform(0, max(1, span - length))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    along = xx if side in (0, 2) else yy
    u = (along - start) / length
    inside = (u > 0) & (u < 1)
    taper = np.sqrt(np.clip(np.sin(np.clip(u, 0, 1) * math.pi), 0, 1))
    n = smooth_noise(rng, H, W, 6)
    thr = depth * taper * (0.4 + 0.9 * n)
    near_side = {0: yy < 20, 1: xx > W - 21, 2: yy > H - 21, 3: xx < 20}[side]
    region = inside & near_side & (alpha > 127) & (dist < thr)
    if region.sum() < 12:
        return
    _whiten(img, region.astype(np.float32), rng.uniform(0.6, 1.0), rng, bright)
    loss = region & (dist < thr * rng.uniform(0.0, 0.3))
    alpha[loss] = 0
    mask[region] = 2


def _scratch(rng, img, mask, alpha, dist):
    H, W = mask.shape
    p0 = np.array([rng.uniform(20, W - 20), rng.uniform(20, H - 20)])
    ang = rng.uniform(0, math.pi)
    L = rng.uniform(40, 300)
    p2 = p0 + L * np.array([math.cos(ang), math.sin(ang)])
    mid = (p0 + p2) / 2 + rng.normal(0, L * 0.12, 2)
    t = np.linspace(0, 1, 60)[:, None]
    pts = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * mid + t ** 2 * p2).astype(np.int32)
    line = np.zeros((H, W), np.uint8)
    cv2.polylines(line, [pts], False, 255, int(rng.integers(1, 3)), cv2.LINE_AA)
    line = cv2.GaussianBlur(line, (3, 3), 0)
    lf = (line.astype(np.float32) / 255) * (alpha > 127)
    if rng.random() < 0.8:
        img[:] = img + lf[..., None] * rng.uniform(45, 95)          # bright scuff
    else:
        img[:] = img - lf[..., None] * rng.uniform(35, 70)          # dark scratch
    label = cv2.dilate((line > 40).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    label &= (alpha > 127) & (dist > 2)
    if label.sum() < 20:
        return
    mask[label] = 3


def _crease(rng, img, mask, alpha):
    H, W = mask.shape
    p = np.array([rng.uniform(0, W), rng.uniform(0, H)])
    ang = rng.uniform(0, math.pi)
    L = rng.uniform(80, 420)
    d = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rel_x, rel_y = xx - p[0], yy - p[1]
    s = rel_x * d[0] + rel_y * d[1]
    q = rel_x * nrm[0] + rel_y * nrm[1]
    w = rng.uniform(3, 7)
    along = np.clip(1 - np.abs(s) / (L / 2), 0, 1) ** 0.4
    prof = np.where(np.abs(q) < w * 2, np.tanh(q / w) * np.exp(-(q / (w * 1.6)) ** 2), 0)
    shade = prof * along * rng.uniform(40, 75)
    ridge = np.exp(-(q / 1.2) ** 2) * along * rng.uniform(30, 60)
    img[:] = img + (shade + ridge)[..., None] * (alpha[..., None] > 127)
    region = (np.abs(q) < w * 1.4) & (along > 0.05) & (alpha > 127)
    if region.sum() < 30:
        return
    mask[region] = 4


# ----------------------------------------------------------------------------
# scan simulation
# ----------------------------------------------------------------------------

def simulate_scan(rng, card):
    W, H = CARD_W, CARD_H
    pad_x, pad_y = int(rng.integers(45, 95)), int(rng.integers(45, 95))
    Ws, Hs = W + 2 * pad_x, H + 2 * pad_y

    # scanning protocol assumption: bright-bordered items go on a dark lid
    lum = float(card["img"][4:10, CARD_W // 3:2 * CARD_W // 3].mean())
    p_dark = 0.92 if lum > 150 else (0.05 if lum < 90 else 0.45)
    dark = rng.random() < p_dark
    lvl = rng.uniform(14, 40) if dark else rng.uniform(222, 246)
    bg = np.full((Hs, Ws, 3), lvl, np.float32)
    bg += (smooth_noise(rng, Hs, Ws, 200)[..., None] - 0.5) * 18
    bg += rng.normal(0, 2.5, bg.shape)

    ang = rng.uniform(-2.5, 2.5)
    scale = rng.uniform(0.94, 1.04)
    cx, cy = Ws / 2 + rng.uniform(-15, 15), Hs / 2 + rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, scale)
    M[0, 2] += cx - W / 2
    M[1, 2] += cy - H / 2

    warped = cv2.warpAffine(card["img"], M, (Ws, Hs), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    a = cv2.warpAffine(card["alpha"], M, (Ws, Hs), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    if not dark:  # soft contact shadow on light lids
        sh = cv2.GaussianBlur(a, (0, 0), 3) * rng.uniform(0.1, 0.25)
        bg *= (1 - sh[..., None])
    out = bg * (1 - a[..., None]) + warped * a[..., None]

    # photometric
    out = out * rng.uniform(0.92, 1.06) + rng.uniform(-8, 8)
    sigma = rng.uniform(0, 0.8)
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out += rng.normal(0, rng.uniform(1, 3.5), out.shape)
    out = out.clip(0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(82, 96))])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    corners = (np.c_[corners, np.ones(4)] @ M.T).astype(np.float32)
    mask_scan = cv2.warpAffine(card["mask"], M, (Ws, Hs), flags=cv2.INTER_NEAREST)
    return {"scan": out, "corners": corners, "M": M, "mask_scan": mask_scan, "dark_bg": dark}


def sample(seed):
    rng = np.random.default_rng(seed)
    card = render_card(rng)
    scan = simulate_scan(rng, card)
    return card, scan
