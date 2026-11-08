"""End-to-end evaluation on held-out test scans, using the exported ONNX model
and the same detection/rectification path the web app uses.

Writes web/public/metrics.json and qualitative figures to web/public/figs/.
"""
import glob, json, os, sys, time
import cv2, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import (CLASS_NAMES, MODEL_H, MODEL_W, centering, components, detect_card, grade,
                      load_rules, measure_margins, rectify)
from synth import CARD_W

WEB = os.path.join(HERE, "..", "web", "public")
FAIL_PX = 10.0  # corner error above this counts as a detection failure
COLORS = {1: (7, 103, 247), 2: (108, 51, 214), 3: (191, 170, 21), 4: (5, 176, 250)}  # BGR


def run_model(sess, img_bgr):
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - 0.5
    x = x.transpose(2, 0, 1)[None]
    cls, conf = sess.run(None, {"image": x})
    return cls[0], conf[0]


def match(pred_map, gt_map, c, dil=3, frac=0.3):
    """instance matching for class c on component label maps"""
    pm = (pred_map == c).astype(np.uint8)
    gm = (gt_map == c).astype(np.uint8)
    npd, pl, ps, _ = cv2.connectedComponentsWithStats(pm, 8)
    ngt, gl, gs, _ = cv2.connectedComponentsWithStats(gm, 8)
    from pipeline import MIN_AREA
    preds = [k for k in range(1, npd) if ps[k, 4] >= MIN_AREA[c]]
    gts = [k for k in range(1, ngt) if gs[k, 4] >= MIN_AREA[c]]
    hit_gt, hit_pred = set(), set()
    k3 = np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)
    for g in gts:
        gd = cv2.dilate((gl == g).astype(np.uint8), k3) > 0
        for p in preds:
            pp = pl == p
            if (pp & gd).sum() >= frac * pp.sum():
                hit_gt.add(g)
                hit_pred.add(p)
    return len(gts), len(preds), len(hit_gt), len(hit_pred)


def overlay(img, cls, alpha=0.55):
    out = img.copy()
    for c, col in COLORS.items():
        m = cls == c
        out[m] = (out[m] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
    return out


def main():
    rules = load_rules()
    sess = ort.InferenceSession(os.path.join(HERE, "out", "damage_unet.onnx"), providers=["CPUExecutionProvider"])
    files = sorted(glob.glob(os.path.join(HERE, "data", "test", "*.jpg")))
    stats = {c: [0, 0, 0, 0] for c in range(1, len(CLASS_NAMES))}
    any_stats = [0, 0, 0, 0]
    inter = np.zeros(len(CLASS_NAMES))
    union = np.zeros(len(CLASS_NAMES))
    corner_err, margin_err, ratio_err, grade_diff, times = [], [], [], [], []
    clean_fp, n_clean = 0, 0
    low_contrast = []
    examples = []
    for f in files:
        meta = json.load(open(f[:-4] + ".json"))
        scan = cv2.imread(f)
        gt_full = cv2.imread(f[:-4] + "_mask.png", 0)
        gt = cv2.resize(gt_full, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)

        t0 = time.perf_counter()
        c = detect_card(scan)
        rect = rectify(scan, c)
        m = measure_margins(rect)
        small = rectify(scan, c, (MODEL_W, MODEL_H))
        cls, conf = run_model(sess, small)
        times.append((time.perf_counter() - t0) * 1000)

        corner_err.append(float(np.abs(c - np.array(meta["corners"])).max()))
        # bright border scanned on a bright lid: the case the scanning protocol rules out
        low_contrast.append(not meta["dark_bg"] and float(scan[:6].mean()) > 200)
        sc = rect.shape[1] / CARD_W
        gm = meta["margins"]
        margin_err.append(max(abs(m[k] / sc - gm[k]) for k in gm))
        lr, tb = centering(m)
        glr, gtb = centering(gm)
        ratio_err.append(max(abs(lr - glr), abs(tb - gtb)))

        inst = components(cls, conf)
        for k in range(1, len(CLASS_NAMES)):
            s = match(cls, gt, k)
            for i in range(4):
                stats[k][i] += s[i]
            inter[k] += ((cls == k) & (gt == k)).sum()
            union[k] += ((cls == k) | (gt == k)).sum()
        a = match((cls > 0).astype(np.uint8), (gt > 0).astype(np.uint8), 1)
        for i in range(4):
            any_stats[i] += a[i]

        gt_inst = components(gt)
        if not gt_inst:
            n_clean += 1
            clean_fp += int(len(inst) > 0)
        g_pred = grade(inst, m, MODEL_W, rules)["grade"]
        g_gt = grade(gt_inst, gm, MODEL_W, rules)["grade"]
        grade_diff.append(g_pred - g_gt)
        examples.append((f, len(gt_inst), small, cls, gt))

    ce, me, re_ = np.array(corner_err), np.array(margin_err), np.array(ratio_err)
    fail = ce > FAIL_PX
    ok = ~fail
    low_contrast = np.array(low_contrast)
    per_class = {}
    for k in range(1, len(CLASS_NAMES)):
        ng, npd, hg, hp = stats[k]
        per_class[CLASS_NAMES[k]] = {
            "gt": ng, "pred": npd,
            "recall": hg / ng if ng else None,
            "precision": hp / npd if npd else None,
            "pixel_iou": float(inter[k] / union[k]) if union[k] else None,
        }
    gd = np.abs(np.array(grade_diff))
    log = json.load(open(os.path.join(HERE, "out", "train_log.json")))
    metrics = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "data": {"train": len(glob.glob(os.path.join(HERE, "data", "train", "*.jpg"))),
                 "val": len(glob.glob(os.path.join(HERE, "data", "val", "*.jpg"))),
                 "test": len(files)},
        "model": {"input": [MODEL_W, MODEL_H], "params": None, "onnx_mb": round(os.path.getsize(os.path.join(HERE, "out", "damage_unet.onnx")) / 1e6, 2)},
        "per_class": per_class,
        "any_damage": {"recall": any_stats[2] / any_stats[0], "precision": any_stats[3] / any_stats[1],
                        "gt": any_stats[0], "pred": any_stats[1]},
        "clean_items": {"n": n_clean, "false_alarm_rate": clean_fp / n_clean if n_clean else None},
        "detection": {"fail_threshold_px": FAIL_PX, "failures": int(fail.sum()), "n": len(files),
                      "failures_low_contrast": int((fail & low_contrast).sum())},
        "corners_px": {"median": float(np.median(ce[ok])), "p95": float(np.percentile(ce[ok], 95))},
        "margins": {"err_px_median": float(np.median(me[ok])), "err_px_p95": float(np.percentile(me[ok], 95)),
                    "ratio_err_pp_median": float(np.median(re_[ok])), "ratio_err_pp_p95": float(np.percentile(re_[ok], 95)),
                    "px_per_mm": round(CARD_W / 63, 2)},
        "grade": {"exact": float((gd == 0).mean()), "within1": float((gd <= 1).mean()),
                  "mean_signed": float(np.mean(grade_diff))},
        "cpu_ms_per_image": float(np.median(times)),
        "train_log": [{"epoch": r["epoch"], "loss": round(r["loss"], 4), "val_miou_fg": round(r["val_miou_fg"], 4)} for r in log],
    }
    with open(os.path.join(WEB, "metrics.json"), "w", encoding="utf-8") as fo:
        json.dump(metrics, fo, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "train_log"}, ensure_ascii=False, indent=1))

    # qualitative figure: prediction vs ground truth for damaged examples
    os.makedirs(os.path.join(WEB, "figs"), exist_ok=True)
    ex = [e for e in examples if e[1] >= 3][:4]
    for i, (f, _, small, cls, gt) in enumerate(ex):
        fig = np.hstack([small, overlay(small, gt), overlay(small, cls)])
        cv2.imwrite(os.path.join(WEB, "figs", f"compare_{i}.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
