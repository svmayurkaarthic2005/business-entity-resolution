"""Controlled rank-convention experiment on the FWD4 configuration (current + reverse-dense k1 + forward-dense k4).  Usage: python src/exp_rankfix.py
Reference  : exp/union_fwd4/pred_TRAIN_union_fwd4.parquet  (rank as built by exp_union/exp_fwd: 1-based over the union; production stage-1 models were TRAINED on block.py's 0-based rank)
This run   : identical allf_union.parquet, identical candidate set / features / model / production folds / decision rule, with ONLY `rank` := rank - 1 (0-based, as trained) before stage 1.
Stage 1 = the production stage-1 models (not retrained); stage 2 (NUM/QUAL/NEW3/NEW4 + dense) is retrained on the resulting p1, exactly as for the reference.
Prints the exp_rank2.report table (F0.5 / P / R / India / US / held-out mean-worst-best / paired gain vs the reference with #positive of 20) for A, FWD4 (reference), RANK0.
"""
import pickle

import numpy as np
import polars as pl

import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
import exp_stack4 as X4
from common import CACHE, log
from exp_final import assemble
from exp_rank2 import report
from exp_union import DENSE

EX = CACHE.parent / "exp"
FWD = EX / "union_fwd4"
OUT = EX / "rankfix"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUT
    X.OUT3 = OUT
    allf = pl.read_parquet(FWD / "allf_union.parquet")
    r0 = allf["rank"]
    log(f"rank before: min {r0.min()} median {r0.median()} max {r0.max()} dtype {r0.dtype}")
    allf = allf.with_columns((pl.col("rank") - 1).cast(r0.dtype).alias("rank"))
    log(f"rank after : min {allf['rank'].min()} (0-based)")
    fmap = dict(pl.read_parquet(FWD / "fold_map.parquet").iter_rows())

    class ProductionFolds:
        def __init__(self, n):
            self.n = n

        def split(self, X_, y, groups):
            f = np.array([fmap[g] for g in groups])
            for i in range(self.n):
                yield np.flatnonzero(f != i), np.flatnonzero(f == i)

    S.GroupKFold = ProductionFolds
    du, folds, fit_ids = assemble(allf, "fwd4_rank0")
    del allf
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    ref = base_cols + S.NEW + E.NEW2 + ["is_india"] + X.GA + X.GC + X.GD + X4.NEW4 + DENSE
    log(f"training stage 2 on rank-0-based p1: {len(ref)} features")
    pv, _ = X.train_oof(du, ref, folds, fit_ids, "r_rank0")
    pv.write_parquet(OUT / "pred_TRAIN_rank0.parquet")
    report({"A saved 0.98108": EX / "final" / "pred_TRAIN__NEW3_NEW4.parquet", "FWD4 ref (rank 1-based)": FWD / "pred_TRAIN_union_fwd4.parquet", "RANK0 (0-based)": OUT / "pred_TRAIN_rank0.parquet"})
    print("DONE")


if __name__ == "__main__":
    main()
