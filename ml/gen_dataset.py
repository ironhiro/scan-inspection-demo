"""Generate the synthetic dataset.

train/val: rectified card images at model resolution + class masks.
           Corners are the ground-truth corners plus N(0, 1.2px) noise to
           mimic detection error, so the model learns to tolerate it.
test:      raw scans + ground truth JSON; evaluated end-to-end (eval.py).
"""
import json, os, sys
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from synth import sample, CARD_W, CARD_H
from pipeline import rectify, MODEL_W, MODEL_H, components

ROOT = os.path.join(os.path.dirname(__file__), "data")
SPLITS = {"train": (0, 5000), "val": (100000, 300), "test": (200000, 500)}


def work(args):
    split, seed = args
    card, scan = sample(seed)
    d = os.path.join(ROOT, split)
    name = f"{seed:06d}"
    if split == "test":
        cv2.imwrite(os.path.join(d, name + ".jpg"), scan["scan"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        rng = np.random.default_rng(seed + 7)
        gt_inst = components(cv2.resize(card["mask"], (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST))
        meta = {"corners": scan["corners"].tolist(), "margins": card["margins"],
                "instances": gt_inst, "dark_bg": bool(scan["dark_bg"])}
        cv2.imwrite(os.path.join(d, name + "_mask.png"), card["mask"])
        with open(os.path.join(d, name + ".json"), "w") as f:
            json.dump(meta, f)
        return
    rng = np.random.default_rng(seed + 7)
    c = scan["corners"] + rng.normal(0, 1.2, scan["corners"].shape).astype(np.float32)
    img = rectify(scan["scan"], c, (MODEL_W, MODEL_H))
    m = rectify(scan["mask_scan"], c, (MODEL_W, MODEL_H), interp=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(d, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(d, name + "_mask.png"), m)


if __name__ == "__main__":
    jobs = []
    for split, (start, n) in SPLITS.items():
        os.makedirs(os.path.join(ROOT, split), exist_ok=True)
        jobs += [(split, start + i) for i in range(n)]
    with Pool(max(1, os.cpu_count() - 2)) as p:
        for i, _ in enumerate(p.imap_unordered(work, jobs, chunksize=8)):
            if i % 500 == 0:
                print(i, "/", len(jobs), flush=True)
    print("done")
