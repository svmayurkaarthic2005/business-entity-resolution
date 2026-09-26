"""Experiment: TEST submission from the stacked stage-2 model (exp_stack.py) -> exp/stack_test/.

Usage: python src/exp_stack_test.py
p1 = cache/pred_TEST.parquet (production model, fold-averaged: same treatment as the VAL rows in exp_stack).
Stage-2 features with exp_stack.stage2_features on the test pools. Scoring: US/India rows with the
stage2+is_india model, France rows with plain stage2 (France is unseen; is_india would treat it as US).
Decision: production rule. output_full is never touched.
"""
import json
import pickle

import numpy as np
import polars as pl
import xgboost as xgb

from common import CACHE, OUT, log
from decide import apply
from exp_stack import KEY, NEW, stage2_features

N = CACHE / "norm"
STACK = CACHE.parent / "exp" / "stack"
DEST = CACHE.parent / "exp" / "stack_test"


def score(d, tag):
    cal = pickle.load(open(STACK / f"{tag}_calib.pkl", "rb"))
    x = d.select([pl.col(k).cast(pl.Float32) for k in cal["cols"]]).to_numpy()
    raw = np.mean([xgb.Booster(model_file=str(STACK / f"{tag}_fold{i}.json")).predict(xgb.DMatrix(x, feature_names=cal["cols"]))
                   for i in range(cal["folds"])], axis=0)
    return cal["iso"].predict(raw).astype(np.float32)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    f = pl.read_parquet(CACHE / "features_TEST.parquet")
    p = pl.read_parquet(CACHE / "pred_TEST.parquet", columns=[*KEY, "p"]).rename({"p": "p1"})
    d = f.join(p, on=KEY, how="left").with_columns(role=pl.lit("test"))
    assert d["p1"].null_count() == 0
    log(f"TEST rows {d.height:,}")
    d = stage2_features(d, pool="test")
    d = d.with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32))
    p_plain, p_india = score(d, "stage2"), score(d, "stage2_india")
    fr = (d["country"] == "France").to_numpy()
    pred = d.select(KEY).with_columns(p=pl.Series(np.where(fr, p_plain, p_india)))
    pred.write_parquet(DEST / "pred_TEST_stack.parquet")
    d.select(*KEY, *NEW, "p1").write_parquet(DEST / "stage2_features_TEST.parquet")

    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    ctry = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    for tag, pr in (("OLD (submitted)", pl.read_parquet(CACHE / "pred_TEST.parquet", columns=[*KEY, "p"])), ("NEW stacked", pred)):
        k = apply(pr, cfg).join(ctry, on="s1_id")
        for c in ("France", "India", "US"):
            n = ctry.filter(pl.col("country") == c).height
            kc = k.filter(pl.col("country") == c)
            print(f"  {tag:16s} {c:7s} pred/S1 {kc.height / n:.3f}  empty {1 - kc['s1_id'].n_unique() / n:.2%}")

    kept = apply(pred, cfg)
    ids = {s: pl.read_parquet(CACHE / f"test_source{s}.parquet", columns=["id", "entity_id"]) for s in (1, 2, 3)}
    ent = pl.concat([ids[s].with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    s1 = ids[1].rename({"id": "s1_id", "entity_id": "source1_entity_id"})
    for rows, col, name in ((kept, "matched_entity_ids", "matching_results.tsv"), (pred, "candidate_entity_ids", "candidate_pairs.tsv")):
        lists = (rows.select(KEY).unique().join(ent, on=["cand_id", "source"])
                 .sort("entity_id").group_by("s1_id").agg(pl.col("entity_id").str.join(",").alias(col)))
        out = s1.join(lists, on="s1_id", how="left").with_columns(pl.col(col).fill_null("")).sort("s1_id").select("source1_entity_id", col)
        out.write_csv(DEST / name, separator="\t", quote_style="never")
        print(f"wrote {DEST / name}: {len(out):,} rows, non-empty {(out[col] != '').mean():.3f}")
    full = OUT.parent / "output_full"
    print("candidate_pairs identical to output_full:", (full / "candidate_pairs.tsv").read_bytes() == (DEST / "candidate_pairs.tsv").read_bytes())


if __name__ == "__main__":
    main()
