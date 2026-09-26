"""Experiment A: candidate-aware GROUPED RANKING (LambdaMART-style) on the cached NEW3 feature table.  VAL held-out.  exp/rank/
Usage: python src/exp_rank.py   (needs exp/stack3/feats_TRAIN.parquet)

Group = one S1 entity (all its candidates, both sources).  XGBoost `rank:pairwise` (GPU), trained STRICTLY out-of-fold with the same
3 S1-grouped folds as every other model (fit rows get their out-of-fold ranker score; VAL / competitor rows get the fold average).
Ranker features = everything in the +all NEW3 model plus the source indicator.
Ranking-derived features fed to the final pointwise model:
   r_score, r_rank (within S1), r_gap (to the best OTHER candidate of the S1), r_share (softmax within S1),
   or_rank / or_gap (target side: rank of this S1's score among S1s claiming the same record, gap to the best other S1).
Variants: ref (+all NEW3, pointwise) ; +ranking features ; +ranking features only for S1 side ; the ranker score alone is NOT used for the
decision (it is not a probability) -- it only enters the pointwise model, which is calibrated and decided exactly like before.
"""
import pickle

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
from common import CACHE, log

OUTR = CACHE.parent / "exp" / "rank"
KEY = S.KEY
RANKF = ["r_score", "r_rank", "r_gap", "r_share", "or_rank", "or_gap"]


def ranker_oof(d, cols, folds, fit_pos, tag):
    """Grouped ranker; returns a np.float32 score per row of d (OOF for fit rows)."""
    fit = d[fit_pos]
    y = fit["is_match"].to_numpy().astype(np.float32)
    s = fit["s1_id"].to_numpy()
    x = fit.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
    xa = d.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
    params = dict(objective="rank:pairwise", eval_metric="ndcg@5", device="cuda", tree_method="hist", max_depth=8, learning_rate=0.08,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5, max_bin=256, seed=42)
    out_oof = np.zeros(len(y), np.float32)
    avg = np.zeros(len(xa), np.float64)
    for i, (tr, va) in enumerate(folds):
        def dm(idx, feats):
            o = np.argsort(s[idx], kind="stable")
            return xgb.DMatrix(feats[idx][o], label=y[idx][o], qid=s[idx][o], feature_names=cols), o
        dtr, _ = dm(tr, x)
        dva, ova = dm(va, x)
        m = xgb.train(params, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=60, verbose_eval=250)
        pv = m.predict(xgb.DMatrix(x[va], feature_names=cols), iteration_range=(0, m.best_iteration + 1))
        out_oof[va] = pv
        avg += m.predict(xgb.DMatrix(xa, feature_names=cols), iteration_range=(0, m.best_iteration + 1))
        m.save_model(str(OUTR / f"{tag}_fold{i}.json"))
        log(f"ranker fold {i}: best iter {m.best_iteration}")
    r = (avg / len(folds)).astype(np.float32)
    r[fit_pos] = out_oof
    return r


def rank_features(d, r):
    d = d.with_columns(r_score=pl.Series(r))
    g = ["s1_id"]
    top2 = pl.col("r_score").top_k(2).min().over(g)
    d = d.with_columns(r_rank=pl.col("r_score").rank("ordinal", descending=True).over(g).cast(pl.Int16),
                       r_top1=pl.col("r_score").max().over(g), r_top2=top2, r_n=pl.len().over(g))
    d = d.with_columns(r_gap=pl.when(pl.col("r_n") == 1).then(pl.col("r_score") + 5.0).when(pl.col("r_rank") == 1)
                       .then(pl.col("r_score") - pl.col("r_top2")).otherwise(pl.col("r_score") - pl.col("r_top1")))
    d = d.with_columns(r_e=(pl.col("r_score") - pl.col("r_top1")).exp())
    d = d.with_columns(r_share=pl.col("r_e") / pl.col("r_e").sum().over(g))
    t = ["cand_id", "source"]
    d = d.with_columns(or_rank=pl.col("r_score").rank("ordinal", descending=True).over(t).cast(pl.Int16),
                       or_top1=pl.col("r_score").max().over(t), or_top2=pl.col("r_score").top_k(2).min().over(t), or_n=pl.len().over(t))
    d = d.with_columns(or_gap=pl.when(pl.col("or_n") == 1).then(pl.col("r_score") + 5.0).when(pl.col("or_rank") == 1)
                       .then(pl.col("r_score") - pl.col("or_top2")).otherwise(pl.col("r_score") - pl.col("or_top1")))
    return d.drop("r_top1", "r_top2", "r_n", "r_e", "or_top1", "or_top2", "or_n")


def main():
    OUTR.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUTR
    X.OUT3 = OUTR
    d = pl.read_parquet(CACHE.parent / "exp" / "stack3" / "feats_TRAIN.parquet")
    d = d.with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32), src=pl.col("source").cast(pl.Float32))
    fit = d.filter(pl.col("role") == "fit")
    fit_pos = np.flatnonzero((d["role"] == "fit").to_numpy())
    folds = list(GroupKFold(3).split(np.zeros(fit.height), fit["is_match"].to_numpy(), fit["s1_id"].to_numpy()))
    fit_ids = fit.select("s1_id")
    del fit
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    ref_cols = base_cols + S.NEW + E.NEW2 + X.GA + X.GC + X.GD
    log(f"table {d.shape}; ranker features {len(ref_cols) + 1}")
    preds = {"base (production model)": d.select(*KEY, "role", "is_match", p="p1")}
    log("training: ref (+all NEW3, pointwise)")
    preds["ref (+all NEW3)"], _ = X.train_oof(d, ref_cols, folds, fit_ids, "r_ref")
    log("training: grouped ranker (out-of-fold)")
    r = ranker_oof(d, ref_cols + ["src"], folds, fit_pos, "ranker")
    d = rank_features(d, r)
    fit_r = d.filter(pl.col("role") == "fit")
    from sklearn.metrics import roc_auc_score
    log(f"ranker OOF AUC (fit rows): {roc_auc_score(fit_r['is_match'].to_numpy(), fit_r['r_score'].to_numpy()):.5f}")
    log("training: +ranking features (S1 and target side)")
    preds["+ranking features"], _ = X.train_oof(d, ref_cols + RANKF, folds, fit_ids, "r_rankf")
    log("training: +ranking features (S1 side only)")
    preds["+ranking features (S1 side)"], _ = X.train_oof(d, ref_cols + RANKF[:4], folds, fit_ids, "r_rankf_s1")
    for k, v in preds.items():
        v.write_parquet(OUTR / f"pred_TRAIN_{''.join(c if c.isalnum() else '_' for c in k)}.parquet")
    S.evaluate(preds, ref="ref (+all NEW3)")


if __name__ == "__main__":
    main()
