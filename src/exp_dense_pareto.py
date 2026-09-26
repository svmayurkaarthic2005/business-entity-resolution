"""Candidate-union Pareto sweep on VAL (no model, no retraining).  Usage: python src/exp_dense_pareto.py   (needs exp/dense/{fwd,rev}_train.parquet)
Configurations (all = current TF-IDF set + ...): rev k1 | rev k1 + fwd k4 | rev k1 + fwd k8 | rev k2 | rev k2 + fwd k4.  fwd k = top-k per source.
Extra rows: overlap of the ADDED dense candidates with the current set.  Ceilings are candidate-generation ceilings (perfect matcher), not model scores.
"""
import polars as pl

from common import CACHE
from exp_block2 import ceiling
from exp_budget import val_truth
from exp_dense import KEY, OUTD as D


def main():
    val_ids, truth_n, gt = val_truth()
    n_s1, n_true = truth_n.height, gt.height
    n_pos = truth_n.filter(pl.col("n") > 0).height
    cv = pl.read_parquet(CACHE / "candidates_VAL.parquet").join(val_ids, on="s1_id", how="semi")
    ex = cv.filter((pl.col("rank") < 10) & (pl.col("score") >= 0.003)).select(KEY).unique()
    missed = gt.join(ex, on=KEY, how="anti")
    fwd = pl.read_parquet(D / "fwd_train.parquet").join(val_ids, on="s1_id", how="semi")
    rev = pl.read_parquet(D / "rev_train.parquet").join(val_ids, on="s1_id", how="semi")
    R = lambda k: rev.filter(pl.col("rrank") < k).select(KEY)
    F = lambda k: fwd.filter(pl.col("drank") < k).select(KEY)
    c0 = ceiling(ex, truth_n, gt)
    print(f"current: {ex.height / n_s1:.2f} cands/S1, recall {gt.join(ex, on=KEY, how='semi').height / n_true:.4f}, ceiling {c0:.5f}; true pairs {n_true:,}; missed by current {missed.height:,}\n")
    cfgs = [("1 cur + rev k1", [R(1)]), ("2 cur + rev k1 + fwd k4", [R(1), F(4)]), ("3 cur + rev k1 + fwd k8", [R(1), F(8)]),
            ("4 cur + rev k2", [R(2)]), ("5 cur + rev k2 + fwd k4", [R(2), F(4)])]
    print(f"{'configuration':26s} {'recall':>7s} {'newTrue':>8s} {'missedRec':>9s} {'cand/S1':>8s} {'+cand/S1':>8s} {'S1full':>7s} {'ceiling':>8s} {'dCeil':>8s} {'new/1k':>7s} | {'added':>10s} {'inCur%':>6s} {'newCand':>9s}")
    for name, parts in cfgs:
        add = pl.concat(parts).unique()
        f = pl.concat([ex, add]).unique()
        hit = gt.join(f, on=KEY, how="semi")
        new_true = hit.join(ex, on=KEY, how="anti").height
        extra = f.height - ex.height
        rec_m = missed.join(f, on=KEY, how="semi").height
        full = (truth_n.filter(pl.col("n") > 0).join(hit.group_by("s1_id").len("h"), on="s1_id", how="left").fill_null(0)
                .select((pl.col("h") == pl.col("n")).mean()).item())
        c = ceiling(f, truth_n, gt)
        inter = add.join(ex, on=KEY, how="semi").height
        print(f"{name:26s} {hit.height / n_true:7.4f} {new_true:8,} {rec_m / missed.height:9.1%} {f.height / n_s1:8.2f} {extra / n_s1:8.2f} {full:7.4f} {c:8.5f} {c - c0:+8.5f} "
              f"{1000 * new_true / extra:7.1f} | {add.height:10,} {inter / add.height:6.1%} {add.height - inter:9,}")
    print("DONE")


if __name__ == "__main__":
    main()
