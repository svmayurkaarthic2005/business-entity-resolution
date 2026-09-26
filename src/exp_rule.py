"""Decision-rule / calibration experiment on the E-equivalent predictions (exp/rank2/pred_CTRL.parquet; no retraining), same 20 pinned split-halves.  Usage: python src/exp_rule.py
Part 1  diagnostic: true pairs rejected by the production rule -- why (owner-adjust demoted / below floor / cut by expected-F prefix / S1 predicted empty) and their p.
Part 2  held-out comparison (rule re-chosen on one half, scored on the other half, exactly exp_stack.evaluate's protocol):
   PROD    the production grid (exp_stack.GRID)                                                    -> reproduces 0.98633 held-out
   EXT     extended grid: alpha, miss, gamma, floor
   EXT+CAL EXT plus a calibration of p in logit space  p' = sigmoid(tau*logit(p) + b)
   +CTRY   any of the above with a SEPARATE config per country (US / India), each chosen on the selection half only
Reports full-val F0.5 of the config chosen on ALL validation S1 (in-sample, for reference only), held-out mean/worst/best, paired gain vs PROD (mean/min/max, #positive/20), US/India, P, R.
"""
import itertools

import numpy as np
import polars as pl

import exp_stack as S
from common import CACHE
from decide import choose, owner_adjust
from exp_budget import val_truth

EX = CACHE.parent / "exp"
KEY = S.KEY
EXT = dict(alpha=(0.2, 0.35, 0.5, 0.7, 0.85), miss=(0.3, 0.6, 0.9), gamma=(1.5, 2.0, 3.0, 4.0), floor=(0.08, 0.15, 0.22))
CAL = dict(tau=(0.85, 1.0, 1.2), b=(-0.3, 0.0, 0.3))


def calib(p, tau, b):
    lg = (p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6))).log()
    return 1 / (1 + (-(tau * lg + b)).exp())


def grid_scores(pred, truth_n, grid, cal=False):
    res = {}
    for tau, b in (itertools.product(*CAL.values()) if cal else [(1.0, 0.0)]):
        q = pred.with_columns(p=calib(pl.col("p"), tau, b)) if (tau, b) != (1.0, 0.0) else pred
        pa = {a: owner_adjust(q, a) for a in grid["alpha"]}
        for cfg in itertools.product(*grid.values()):
            res[cfg + (tau, b)] = S.per_s1(choose(pa[cfg[0]], *cfg[1:]).filter("take"), truth_n)
    return res


def heldout(res, india, mask_fn=None, per_country=False):
    """20 split-halves: choose on one half, score on the other.  per_country: choose separately for US and India."""
    out = []
    keys = list(res)
    F = np.stack([res[c][0] for c in keys])                        # cfgs x S1
    P = np.stack([res[c][1] for c in keys])
    R = np.stack([res[c][2] for c in keys])
    for seed in range(10):
        m = np.random.default_rng(seed).random(F.shape[1]) < 0.5
        for a, b in ((m, ~m), (~m, m)):
            if per_country:
                f = np.zeros(F.shape[1])
                for grp in (india, ~india):
                    ia = np.flatnonzero(a & grp)
                    pick = int(np.argmax(F[:, ia].mean(1)))
                    ib = b & grp
                    f[ib] = F[pick, ib]
                out.append((f[b].mean(), None, None, None, None))
            else:
                pick = int(np.argmax(F[:, a].mean(1)))
                out.append((F[pick, b].mean(), F[pick, b & india].mean(), F[pick, b & ~india].mean(), P[pick, b].mean(), R[pick, b].mean()))
    return out


def main():
    _, truth_n, gt = val_truth()
    ctry = pl.read_parquet(CACHE / "norm" / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    india = truth_n.join(ctry, on="s1_id")["country"].to_numpy() == "India"
    pred = pl.read_parquet(EX / "rank2" / "pred_CTRL.parquet").filter(pl.col("role") == "val")
    # ---- part 1: why are true pairs rejected?
    cfg = (0.5, 0.6, 2.0, 0.15)
    pa = owner_adjust(pred, cfg[0])
    ch = choose(pa, *cfg[1:])
    kept = ch.filter("take").select(*KEY).with_columns(kept=pl.lit(True))
    t = pa.filter(pl.col("is_match")).join(kept, on=KEY, how="left").filter(pl.col("kept").is_null())
    own = pred.with_columns(top=pl.col("p") >= pl.col("p").max().over("cand_id", "source")).select(*KEY, "top")
    t = t.join(own, on=KEY, how="left").join(ch.select(*KEY, "k", "kbest"), on=KEY, how="left")
    why = (pl.when(~pl.col("top")).then(pl.lit("owner-demoted (another S1 scores the record higher)"))
           .when(pl.col("pa") < cfg[3]).then(pl.lit("below floor"))
           .when(pl.col("kbest") == 0).then(pl.lit("S1 predicted empty (gamma * P(no match) won)"))
           .otherwise(pl.lit("cut by expected-F prefix")))
    t = t.with_columns(why=why)
    print(f"rejected true pairs among retrieved: {t.height:,}")
    for w, n, med, hi in t.group_by("why").agg(n=pl.len(), med=pl.col("p").median(), hi=(pl.col("p") >= 0.5).sum()).sort("n", descending=True).iter_rows():
        print(f"  {w:52s} {n:6,}  median p {med:.3f}  with p>=0.5: {hi:,}")
    # ---- part 2
    print("\ncomputing PROD grid ...")
    prod = grid_scores(pred, truth_n, S.GRID)
    print("computing EXT grid ...")
    ext = grid_scores(pred, truth_n, EXT)
    print("computing EXT+CAL grid ...")
    extc = grid_scores(pred, truth_n, EXT, cal=True)
    base = heldout(prod, india)
    bm = np.array([x[0] for x in base])
    print(f"\n{'variant':16s} {'inVAL F0.5':>10s} | held-out mean  worst   best |   P      R     India     US   | vs PROD: mean   min    max   #pos/20 | vs 0.98108 | vs 0.98647")
    for name, res, pc in (("PROD", prod, False), ("EXT", ext, False), ("EXT+CAL", extc, False), ("PROD+CTRY", prod, True), ("EXT+CTRY", ext, True), ("EXT+CAL+CTRY", extc, True)):
        h = heldout(res, india, per_country=pc)
        v = np.array([x[0] for x in h])
        g = v - bm
        best = max(res, key=lambda c: res[c][0].mean())
        pi = "" if pc else f"{np.mean([x[3] for x in h]):.4f} {np.mean([x[4] for x in h]):.4f} {np.mean([x[1] for x in h]):.5f} {np.mean([x[2] for x in h]):.5f}"
        print(f"{name:16s} {res[best][0].mean():10.5f} | {v.mean():.5f} {v.min():.5f} {v.max():.5f} | {pi:34s} | {g.mean():+.5f} {g.min():+.5f} {g.max():+.5f} {int((g > 0).sum()):3d}/20 | {v.mean() - 0.98108:+.5f} | {v.mean() - 0.98647:+.5f}   best cfg {best if not pc else '(per country)'}")
    print("DONE")


if __name__ == "__main__":
    main()
