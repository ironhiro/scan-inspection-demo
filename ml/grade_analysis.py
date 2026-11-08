"""Why do predicted grades disagree with ground-truth grades?

Runs the same end-to-end path as eval.py on the test scans, then for every
mismatch swaps one deduction category at a time from prediction to ground
truth and checks whether the grade then matches. A category that fixes many
mismatches on its own is where the error comes from.

Writes ml/out/grade_analysis.json (does not touch web/public/metrics.json).
"""
import glob, json, os, sys
from collections import Counter, defaultdict
import cv2, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import CLASS_NAMES, MODEL_H, MODEL_W, components, detect_card, grade, load_rules, measure_margins, rectify
from eval import FAIL_PX, run_model

REASONS = ["centering"] + CLASS_NAMES[1:]


def grade_from(items, cap):
    total = sum(i["deduct"] for i in items)
    return int(max(1, min(cap, np.floor(10 - total))))


def by_reason(items):
    d = defaultdict(float)
    for i in items:
        d[i["reason"]] += i["deduct"]
    return d


def main():
    rules = load_rules()
    sess = ort.InferenceSession(os.path.join(HERE, "out", "damage_unet.onnx"), providers=["CPUExecutionProvider"])
    files = sorted(glob.glob(os.path.join(HERE, "data", "test", "*.jpg")))
    rows = []
    for f in files:
        meta = json.load(open(f[:-4] + ".json"))
        scan = cv2.imread(f)
        gt = cv2.resize(cv2.imread(f[:-4] + "_mask.png", 0), (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)
        c = detect_card(scan)
        m = measure_margins(rectify(scan, c))
        cls, conf = run_model(sess, rectify(scan, c, (MODEL_W, MODEL_H)))
        gp = grade(components(cls, conf), m, MODEL_W, rules)
        gg = grade(components(gt), meta["margins"], MODEL_W, rules)
        fixes = {}
        for r in REASONS:
            items = [i for i in gp["items"] if i["reason"] != r] + [i for i in gg["items"] if i["reason"] == r]
            cap = gg["cap"] if r == "crease" else gp["cap"]
            fixes[r] = grade_from(items, cap) == gg["grade"]
        dp, dg = by_reason(gp["items"]), by_reason(gg["items"])
        rows.append({
            "file": os.path.basename(f),
            "pred": gp["grade"], "gt": gg["grade"],
            "det_fail": bool(np.abs(c - np.array(meta["corners"])).max() > FAIL_PX),
            "deduct_diff": {r: round(dp[r] - dg[r], 2) for r in REASONS if dp[r] != dg[r]},
            "total_pred": sum(dp.values()), "total_gt": sum(dg.values()),
            "fixed_by": [r for r in REASONS if fixes[r]],
        })

    mis = [r for r in rows if r["pred"] != r["gt"]]
    n = len(rows)
    fixed_by = Counter(x for r in mis for x in r["fixed_by"])
    sole = Counter(r["fixed_by"][0] for r in mis if len(r["fixed_by"]) == 1)
    involved = Counter(k for r in mis for k in r["deduct_diff"])
    # deduction total off by < 1 point yet grade differs -> floor() boundary effect
    small = [r for r in mis if abs(r["total_pred"] - r["total_gt"]) < 1.0]
    summary = {
        "n": n, "mismatch": len(mis), "exact": 1 - len(mis) / n,
        "mismatch_with_det_fail": sum(r["det_fail"] for r in mis),
        "direction": dict(Counter("pred_lower" if r["pred"] < r["gt"] else "pred_higher" for r in mis)),
        "category_involved": dict(involved.most_common()),
        "fixed_if_category_correct": dict(fixed_by.most_common()),
        "fixed_only_by_category": dict(sole.most_common()),
        "not_fixed_by_any_single": sum(not r["fixed_by"] for r in mis),
        "deduct_total_diff_lt_1pt": len(small),
        "deduct_total_diff_hist": dict(sorted(Counter(round(r["total_pred"] - r["total_gt"], 1) for r in mis).items())),
    }
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "grade_analysis.json"), "w", encoding="utf-8") as fo:
        json.dump({"summary": summary, "mismatches": mis}, fo, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
