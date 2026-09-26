"""Pre-flight for 'current + reverse-dense-k1 + forward-dense-k4' (no model, no retraining).  Usage: python src/exp_fwd_pre.py
Existing union = the VAL rows of the E model's prediction file (exp/union/pred_TRAIN_union_rev_k1.parquet: current TF-IDF candidates + reverse dense k1).
Added = forward dense top-4 per source (exp/dense/fwd_train.parquet, drank < 4) for the VAL S1.
Reports exact counts, candidates/S1, blocking recall, genuinely new pairs, overlap with the existing union, and how many of the 1,602 never-retrieved true pairs of E
(exp/union/errbudget_fn.parquet, category 1) become retrievable.  Candidate ceilings are NOT model scores.
"""
import polars as pl

from common import CACHE
from exp_budget import val_truth
from exp_dense import KEY, OUTD as D

E = CACHE.parent / "exp"


def main():
    val_ids, truth_n, gt = val_truth()
    n_s1, n_true = truth_n.height, gt.height
    ex = pl.read_parquet(E / "union" / "pred_TRAIN_union_rev_k1.parquet", columns=[*KEY, "role"]).filter(pl.col("role") == "val").select(KEY)
    fwd = pl.read_parquet(D / "fwd_train.parquet").join(val_ids, on="s1_id", how="semi")
    dt = {c: ex.schema[c] for c in KEY}
    add = fwd.filter(pl.col("drank") < 4).select([pl.col(c).cast(dt[c]) for c in KEY]).unique()
    new = add.join(ex, on=KEY, how="anti")
    inter = add.height - new.height
    f = pl.concat([ex, new])
    assert f.unique().height == f.height
    hit0, hit1 = gt.join(ex, on=KEY, how="semi").height, gt.join(f, on=KEY, how="semi").height
    nt = gt.join(new, on=KEY, how="semi").height
    nr = pl.read_parquet(E / "union" / "errbudget_fn.parquet").filter(pl.col("cat").str.starts_with("1")).select(KEY)
    rec = nr.join(new.select([pl.col(c).cast(nr.schema[c]) for c in KEY]), on=KEY, how="semi").height
    print(f"VAL S1 {n_s1:,}; true pairs {n_true:,}")
    print(f"existing union (current + rev k1): {ex.height:,} candidates = {ex.height / n_s1:.2f}/S1, blocking recall {hit0 / n_true:.4f}")
    print(f"forward k4 rows: {add.height:,} ({add.height / n_s1:.2f}/S1); already in the union {inter:,} ({inter / add.height:.1%}); genuinely NEW {new.height:,} ({new.height / n_s1:.2f}/S1)")
    print(f"new union: {f.height:,} candidates = {f.height / n_s1:.2f}/S1, blocking recall {hit1 / n_true:.4f} (+{hit1 - hit0:,} true pairs = {nt:,} among the new rows)")
    print(f"never-retrieved true pairs of E: {nr.height:,}; retrievable with fwd k4: {rec:,} ({rec / nr.height:.1%}); precision of new rows (true/new) {nt / max(new.height, 1):.4f}")
    # sizing of the FULL build (all TRAIN S1, as exp_union.py does): rows the union build would have to featurise
    old = pl.read_parquet(CACHE / "candidates_TRAIN.parquet", columns=KEY)
    dt2 = {c: old.schema[c] for c in KEY}
    allf4 = pl.read_parquet(D / "fwd_train.parquet").filter(pl.col("drank") < 4).select([pl.col(c).cast(dt2[c]) for c in KEY]).unique()
    known = pl.concat([old, pl.read_parquet(E / "union" / "new_candidates.parquet").select([pl.col(c).cast(dt2[c]) for c in KEY])])
    extra = allf4.join(known, on=KEY, how="anti")
    print(f"FULL build sizing: fwd k4 rows over all TRAIN S1 {allf4.height:,}; not already in current+rev-k1 build (rows to featurise anew) {extra.height:,}; existing new rows {known.height - old.height:,}")
    print("DONE")


if __name__ == "__main__":
    main()
