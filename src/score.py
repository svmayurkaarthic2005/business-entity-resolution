"""Competition metric (macro F0.5 per Source-1 entity) and blocking recall.

Run `python src/score.py` to execute the self-tests.
"""
import numpy as np


def f05(pred, truth):
    """F0.5 for one S1 entity. Empty/empty = 1; any empty side otherwise = 0."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    return 1.25 * tp / (0.25 * len(truth) + len(pred)) if pred else 0.0


def score(pred, truth, country=None):
    """Macro F0.5 over every S1 in `truth` (missing preds count as empty).

    pred, truth: dict s1 -> set of matched ids. country: optional dict s1 -> label.
    Returns dict with f05, precision, recall, singleton_acc and per_country f05.
    """
    f, p, r, single, groups = [], [], [], [], {}
    for s, t in truth.items():
        q = pred.get(s, set())
        v = f05(q, t)
        f.append(v)
        tp = len(q & t)
        p.append(tp / len(q) if q else float(not t))
        r.append(tp / len(t) if t else float(not q))
        if not t:
            single.append(v)
        if country is not None:
            groups.setdefault(country[s], []).append(v)
    return {"f05": float(np.mean(f)), "precision": float(np.mean(p)),
            "recall": float(np.mean(r)),
            "singleton_acc": float(np.mean(single)) if single else None,
            "per_country": {c: float(np.mean(v)) for c, v in groups.items()}}


def blocking_recall(cands, truth):
    """Share of true pairs present in candidates, % S1 fully covered, avg/p95 candidates."""
    hit = total = full = 0
    sizes = []
    for s, t in truth.items():
        c = cands.get(s, set())
        sizes.append(len(c))
        h = len(c & t)
        hit, total, full = hit + h, total + len(t), full + (h == len(t))
    sizes = np.array(sizes)
    return {"pair_recall": hit / total if total else 1.0,
            "s1_fully_covered": full / len(truth),
            "avg_cands": float(sizes.mean()), "p95_cands": float(np.percentile(sizes, 95))}


def _tests():
    # problem-statement example: P=2/3, R=1 -> 0.714
    ex = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(ex - 0.714) < 1e-3, ex
    assert f05(set(), set()) == 1.0            # singleton + empty
    assert f05({"x"}, set()) == 0.0            # singleton + any
    assert f05(set(), {"x"}) == 0.0            # non-singleton + empty
    assert f05({"a", "b"}, {"a", "b"}) == 1.0
    s = score({1: {"a"}, 2: set()}, {1: {"a", "b"}, 2: set(), 3: {"c"}},
              country={1: "US", 2: "US", 3: "India"})
    assert abs(s["f05"] - (1.25 / 1.5 + 1 + 0) / 3) < 1e-9, s
    assert s["per_country"] == {"US": (1.25 / 1.5 + 1) / 2, "India": 0.0}
    b = blocking_recall({1: {"a", "z"}, 3: {"c"}}, {1: {"a", "b"}, 3: {"c"}})
    assert b["pair_recall"] == 2 / 3 and b["s1_fully_covered"] == 0.5 and b["avg_cands"] == 1.5
    print(f"all score tests passed (example F0.5 = {ex:.4f})")


if __name__ == "__main__":
    _tests()
