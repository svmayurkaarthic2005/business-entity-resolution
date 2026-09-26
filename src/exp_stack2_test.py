"""TEST submission from the stage-2 model WITH number-set + name-qualifier features (exp/stack2) -> exp/stack2_test/.

Usage: python src/exp_stack2_test.py
p1 = cache/pred_TEST.parquet (production model). Stage-2 features via exp_stack.stage2_features(pool="test"); number-set and
qualifier features via exp_stack2.tok_num + the token log-odds tables learned on the FIT rows (exp/stack2/vocab_{X,M}.parquet;
unseen tokens -> 0, which is also how unseen tokens behave in training). Scoring: US/India with s2new_india, France with s2new
(France is unseen; the India flag would treat it as US). Decision: production rule. output_full is never touched.
"""
import json
import pickle

import numpy as np
import polars as pl
import xgboost as xgb

import exp_stack as S
import exp_stack2 as E
from common import CACHE, OUT, log
from decide import apply

N = CACHE / "norm"
S2 = CACHE.parent / "exp" / "stack2"
DEST = CACHE.parent / "exp" / "stack2_test"
KEY = S.KEY
CH = 4_000_000


def qual_test(t, vocab):
    out = t.select(KEY)
    for col, pre in (("X", "x"), ("M", "m")):
        e = (t.select(*KEY, tok=col).explode("tok").drop_nulls("tok").join(vocab[col], on="tok", how="left")
              .with_columns(pl.col("lo").fill_null(0.0)))
        agg = e.group_by(KEY).agg(**{f"{pre}_lo_min": pl.col("lo").min(), f"{pre}_lo_sum": pl.col("lo").sum()})
        out = out.join(agg, on=KEY, how="left")
    return out.with_columns([pl.col(c).fill_null(0.0) for c in ("x_lo_min", "x_lo_sum", "m_lo_min", "m_lo_sum")])


def score(d, tag):
    cal = pickle.load(open(S2 / f"{tag}_calib.pkl", "rb"))
    x = d.select([pl.col(k).cast(pl.Float32) for k in cal["cols"]]).to_numpy()
    raw = np.mean([xgb.Booster(model_file=str(S2 / f"{tag}_fold{i}.json")).predict(xgb.DMatrix(x, feature_names=cal["cols"]))
                   for i in range(cal["folds"])], axis=0)
    return cal["iso"].predict(raw).astype(np.float32)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    vocab = {k: pl.read_parquet(S2 / f"vocab_{k}.parquet") for k in ("X", "M")}
    f = pl.read_parquet(CACHE / "features_TEST.parquet")
    p = pl.read_parquet(CACHE / "pred_TEST.parquet", columns=[*KEY, "p"]).rename({"p": "p1"})
    d = f.join(p, on=KEY, how="left").with_columns(role=pl.lit("test"))
    del f, p
    assert d["p1"].null_count() == 0
    log(f"TEST rows {d.height:,}")
    d = S.stage2_features(d, pool="test")
    parts = []
    for i in range(0, d.height, CH):
        t = E.tok_num(d.slice(i, CH).select(KEY), "test")
        parts.append(t.drop("X", "M").join(qual_test(t.select(*KEY, "X", "M"), vocab), on=KEY, how="left"))
        log(f"  number/qualifier features {min(i + CH, d.height):,}/{d.height:,}")
    d = d.join(pl.concat(parts), on=KEY, how="left")
    d = d.with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32))
    p_plain, p_india = score(d, "s2new"), score(d, "s2new_india")
    fr = (d["country"] == "France").to_numpy()
    pred = d.select(KEY).with_columns(p=pl.Series(np.where(fr, p_plain, p_india)))
    pred.write_parquet(DEST / "pred_TEST_stack2.parquet")
    log("scored")

    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    ctry = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    old = pl.read_parquet(CACHE / "pred_TEST.parquet", columns=[*KEY, "p"])
    prev = pl.read_parquet(CACHE.parent / "exp" / "stack_test" / "pred_TEST_stack.parquet")
    for tag, pr in (("submitted", old), ("stack v1", prev), ("stack2 (new)", pred)):
        k = apply(pr, cfg).join(ctry, on="s1_id")
        for c in ("France", "India", "US"):
            n = ctry.filter(pl.col("country") == c).height
            kc = k.filter(pl.col("country") == c)
            print(f"  {tag:14s} {c:7s} pred/S1 {kc.height / n:.3f}  empty {1 - kc['s1_id'].n_unique() / n:.2%}")
    for c in ("France", "India", "US"):
        q = pred.join(ctry, on="s1_id").filter(pl.col("country") == c)["p"]
        print(f"  stack2 {c:7s} score quantiles p10/25/50/75/90: " + "/".join(f"{q.quantile(x):.3f}" for x in (.1, .25, .5, .75, .9)))

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
