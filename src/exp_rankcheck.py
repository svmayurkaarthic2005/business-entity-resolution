"""Does the 0-based vs 1-based `rank` mismatch change predictions?  Read-only, no training.  Usage: python src/exp_rankcheck.py
block.py stores rank 0-based (best = 0; the production stage-1 models were trained on that).  exp_union.py / exp_fwd.py recompute rank 1-based over the union for ALL rows
(exp_final.py-style `rank(...)` without -1), so the production stage-1 models see rank+1 on every union row (fit, val, comp) of the E / FWD4 models.
Here: apply the SAME production stage-1 models to the VAL rows of exp/union/allf_union.parquet twice -- rank as stored (1-based) and rank-1 (0-based, as in training) --
and compare p1 (calibrated, fold-mean) and the production-rule F0.5 of p1 alone.  Also reports which feature columns carry 'rank'.
"""
import pickle

import numpy as np
import polars as pl

import exp_stack as S
from common import CACHE
from decide import choose, owner_adjust
from exp_budget import val_truth

E = CACHE.parent / "exp"


def main():
    cal = pickle.load(open(CACHE / "calib.pkl", "rb"))
    cols = cal["cols"]
    print("stage-1 feature columns containing 'rank':", [c for c in cols if "rank" in c], "| 'rank' itself in cols:", "rank" in cols)
    lf = pl.scan_parquet(E / "union" / "allf_union.parquet")
    keep = list(dict.fromkeys(["s1_id", "cand_id", "source", "role", "is_match", *cols]))
    d = lf.select(keep).filter(pl.col("role") == "val").collect()
    print(f"VAL rows {d.height:,}; rank min/median/max {d['rank'].min()}/{d['rank'].median()}/{d['rank'].max()} (1-based if min == 1)")
    models = S.load_models(cal)
    _, truth_n, _ = val_truth()

    def p1(dd):
        x = dd.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
        raw = np.stack([m(x) for m in models]).mean(0)
        return cal["iso"].predict(raw).astype(np.float32)

    pa = p1(d)
    fixed = d.with_columns((pl.col("rank") - 1).cast(d.schema["rank"]).alias("rank")) if "rank" in cols else d
    pb = p1(fixed)
    diff = np.abs(pa - pb)
    print(f"p1 as used by E (rank 1-based) vs p1 with rank-1 (0-based, as trained): mean|diff| {diff.mean():.5f}  p99 {np.quantile(diff, .99):.4f}  max {diff.max():.4f}  rows with |diff|>0.02: {(diff > .02).mean():.4%}  corr {np.corrcoef(pa, pb)[0, 1]:.6f}")
    cfg = (0.5, 0.6, 2.0, 0.15)
    for name, p in (("rank as used by E (1-based)", pa), ("rank-1 (0-based, as trained)", pb)):
        q = d.select("s1_id", "cand_id", "source", "role", "is_match").with_columns(p=pl.Series(p))
        kept = choose(owner_adjust(q, cfg[0]).filter(pl.col("role") == "val"), *cfg[1:]).filter("take")
        f = S.per_s1(kept, truth_n)[0]
        print(f"  stage-1 p1 alone, production rule: F0.5 {f.mean():.5f}   [{name}]")
    print("DONE")


if __name__ == "__main__":
    main()
