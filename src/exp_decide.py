"""Experiment only: wider decision-rule search on the EXISTING VAL predictions (no retrain/blocking/features).

Usage:  python src/exp_decide.py
Reads cache/pred_TRAIN.parquet, splits, queried_TRAIN, gt_*, decide.json. Writes nothing to cache/ or output/.
Checks: (1) same VAL S1 set as decide.tune and the current config reproduces decide.json's val_f05;
(2) paired per-S1 difference best-vs-current with std error; (3) split-half: tune on one half, score on the other.
"""
import itertools
import json

import numpy as np
import polars as pl

from common import CACHE
from decide import apply, choose, macro_f05, owner_adjust
from score import score

GRID = dict(alpha=(0.3, 0.5, 0.7), miss=(0.4, 0.6, 0.8, 1.0),
            gamma=(1.5, 2.0, 2.5, 3.0, 4.0, 5.0), floor=(0.1, 0.15, 0.2))


def per_s1(taken, truth_n):
    """Per-S1 f, precision, recall, k (same definitions as score.score), in truth_n row order."""
    agg = taken.group_by("s1_id").agg(k=pl.len(), tp=pl.col("is_match").sum())
    t = truth_n.join(agg, on="s1_id", how="left").fill_null(0)
    n, k, tp = (t[c].to_numpy().astype(float) for c in ("n", "k", "tp"))
    both0 = (n == 0) & (k == 0)
    f = np.where(both0, 1.0, np.where((n == 0) | (k == 0), 0.0, 1.25 * tp / np.maximum(0.25 * n + k, 1e-9)))
    p = np.where(k > 0, tp / np.maximum(k, 1), (n == 0).astype(float))
    r = np.where(n > 0, tp / np.maximum(n, 1), (k == 0).astype(float))
    return f, p, r, k


def main():
    pred = pl.read_parquet(CACHE / "pred_TRAIN.parquet")
    # identical VAL definition to decide.tune (pred_TRAIN branch)
    val_ids = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
    val_ids = val_ids.join(pl.read_parquet(CACHE / "queried_TRAIN.parquet"), on="s1_id")
    truth_n = val_ids.join(pl.read_parquet(CACHE / "gt_s1.parquet").rename({"n_matches": "n"}), on="s1_id").sort("s1_id")
    saved = json.load(open(CACHE / "decide.json"))
    cur = saved["cfg"]
    print(f"VAL S1 {len(truth_n):,}, true pairs {truth_n['n'].sum():,}; current cfg {cur} saved F0.5 {saved['val_f05']:.5f}")

    pa = {a: owner_adjust(pred, a).filter(pl.col("role") == "val") for a in sorted(set(GRID["alpha"]) | {cur["alpha"]})}
    res = {}
    keys = list(itertools.product(*GRID.values()))
    key_cur = (cur["alpha"], cur["miss"], cur["gamma"], cur["floor"])
    for key in [key_cur] + [k for k in keys if k != key_cur]:
        a, miss, gamma, floor = key
        res[key] = per_s1(choose(pa[a], miss, gamma, floor).filter("take"), truth_n)

    # (1) evaluation unchanged: current cfg reproduces the saved score with both scorers
    f_cur = res[key_cur][0].mean()
    ref_cur = macro_f05(apply(pred, cur).filter(pl.col("role") == "val"), truth_n)
    assert abs(f_cur - saved["val_f05"]) < 1e-9 and abs(ref_cur - saved["val_f05"]) < 1e-9, (f_cur, ref_cur, saved["val_f05"])
    print(f"reproduced current F0.5 {f_cur:.5f} (matches decide.json)")

    ranked = sorted(res, key=lambda k: -res[k][0].mean())
    print("\ntop 10 configs (alpha, miss, gamma, floor):")
    for k in ranked[:10]:
        print(f"  {res[k][0].mean():.5f}  {k}")
    for g in GRID["gamma"]:  # best per gamma, to see whether the optimum sits on an edge
        kb = max((k for k in res if k[2] == g), key=lambda k: res[k][0].mean())
        print(f"  best at gamma={g}: {res[kb][0].mean():.5f} {kb}")

    n_s1 = len(truth_n)
    print("\n             macroF0.5  precision  recall  pred/S1  alpha miss gamma floor")
    for name, k in (("current", key_cur), ("best", ranked[0])):
        f, p, r, kk = res[k]
        print(f"  {name:8s}  {f.mean():.5f}   {p.mean():.4f}    {r.mean():.4f}  {kk.sum() / n_s1:.3f}   {k[0]}  {k[1]}  {k[2]}  {k[3]}")
    print(f"  true matches/S1 {truth_n['n'].mean():.3f}")

    # reference scorer on the best cfg (same cross-check decide.tune does)
    kb = ranked[0]
    kept = choose(pa[kb[0]], kb[1], kb[2], kb[3]).filter("take")
    pd_ = {s: set(v) for s, v in kept.group_by("s1_id").agg(pl.col("cand_id") * 10 + pl.col("source")).iter_rows()}
    td = {s: set() for s in truth_n["s1_id"].to_list()}
    for s, c, src in pl.read_parquet(CACHE / "gt_pairs.parquet").join(val_ids, on="s1_id").iter_rows():
        td[s].add(c * 10 + src)
    ref = score(pd_, td)
    assert abs(ref["f05"] - res[kb][0].mean()) < 1e-9
    print(f"reference scorer on best: {ref['f05']:.5f} precision {ref['precision']:.4f} recall {ref['recall']:.4f}")

    # (2) paired per-S1 difference
    d = res[kb][0] - res[key_cur][0]
    print(f"\npaired diff best-current: {d.mean():+.5f} ± {d.std(ddof=1) / np.sqrt(n_s1):.5f} (1 s.e.); "
          f"S1 changed {np.count_nonzero(d)} (better {np.sum(d > 0)}, worse {np.sum(d < 0)})")

    # (3) split-half: choose the config on half A, measure its gain over current on unseen half B
    gains = []
    for seed in range(10):
        m = np.random.default_rng(seed).random(n_s1) < 0.5
        for A, B in ((m, ~m), (~m, m)):
            kA = max(res, key=lambda k: res[k][0][A].mean())
            gains.append(res[kA][0][B].mean() - res[key_cur][0][B].mean())
    g = np.array(gains)
    print(f"split-half held-out gain over current: mean {g.mean():+.5f}, min {g.min():+.5f}, max {g.max():+.5f} ({len(g)} runs)")


if __name__ == "__main__":
    main()
