"""Pick demo scans for the web app (seeds outside train/val/test ranges)."""
import json, os, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth import sample

OUT = os.path.join(HERE, "..", "web", "public", "samples")


def kinds(mask):
    return {int(c) for c in np.unique(mask) if c}


def main(n=12):
    os.makedirs(OUT, exist_ok=True)
    want = [set(), set(), {4}, {4}, {1, 2}, {1, 2, 3}, {3}, {1}, {2, 3}, {1, 3}, {1, 2, 3}, {2}]
    picked, seed = [], 300000
    while len(picked) < n and seed < 302000:
        card, scan = sample(seed)
        k = kinds(card["mask"])
        target = want[len(picked)]
        ok = (k == set()) if not target else target <= k
        if ok:
            picked.append((seed, scan["scan"], card["margins"]))
        seed += 1
    manifest = []
    for i, (s, img, margins) in enumerate(picked):
        name = f"scan_{i + 1:04d}.jpg"
        cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append({"file": name, "seed": s, "margins_gt": margins})
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print([m["seed"] for m in manifest])


if __name__ == "__main__":
    main()
