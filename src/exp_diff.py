"""Density-shift audit (label-free on TEST, labelled on VAL).  Usage: python src/exp_diff.py
Test has ~2x the distractors per S1 of train.  A new model that looks better on VAL can over-accept near-copy distractors on TEST
(another team: VAL +0.003, leaderboard -0.008; the added test pairs had 60-90% house-number mismatch).
For NEW vs BASE (production) kept pairs: count added/removed per country and their house-number mismatch / exact-core-name rates.
VAL: same table plus the true precision of added/removed pairs -> tells what mismatch rate a GOOD addition has.
"""
import json

import polars as pl

from common import CACHE
from decide import apply

N = CACHE / "norm"
E = CACHE.parent / "exp"
KEY = ["s1_id", "cand_id", "source"]


def meta(pool):
    c = pl.concat([pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id", "house_no", "name_core"])
                   .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id", "house_no": "h2", "name_core": "n2"})
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "country", "house_no", "name_core"]).rename(
        {"id": "s1_id", "house_no": "h1", "name_core": "n1"})
    return c, s1


def table(tag, base, new, pool, lab=None):
    c, s1 = meta(pool)
    b, n = base.select(KEY).with_columns(inb=pl.lit(True)), new.select(KEY).with_columns(inn=pl.lit(True))
    u = b.join(n, on=KEY, how="full", coalesce=True).with_columns(pl.col("inb").fill_null(False), pl.col("inn").fill_null(False))
    u = u.join(c, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
    if lab is not None:
        u = u.join(lab, on=KEY, how="left").with_columns(pl.col("is_match").fill_null(False))
    u = u.with_columns(grp=pl.when(pl.col("inb") & pl.col("inn")).then(pl.lit("kept")).when(pl.col("inn")).then(pl.lit("ADDED")).otherwise(pl.lit("removed")),
                       hn_mis=(pl.col("h1").fill_null("") != "") & (pl.col("h2").fill_null("") != "") & (pl.col("h1") != pl.col("h2")),
                       core_eq=pl.col("n1") == pl.col("n2"))
    agg = dict(n=pl.len(), hn_mismatch=pl.col("hn_mis").mean().round(3), exact_core=pl.col("core_eq").mean().round(3))
    if lab is not None:
        agg["precision"] = pl.col("is_match").mean().round(3)
    t = u.group_by("country", "grp").agg(**agg).sort("country", "grp")
    print(f"\n=== {tag}")
    print(t)


def main():
    pl.Config.set_tbl_rows(40)
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    # VAL: production base p1 vs stage2+NUM+QUAL (the model behind exp/stack2_test)
    v = pl.read_parquet(E / "stack2" / "pred_TRAIN_stage2_new_is_india.parquet")
    f = pl.read_parquet(E / "stack3" / "feats_TRAIN.parquet", columns=[*KEY, "p1"])
    vb = apply(v.drop("p").join(f, on=KEY).rename({"p1": "p"}), cfg).filter(pl.col("role") == "val")
    vn = apply(v, cfg).filter(pl.col("role") == "val")
    table("VAL: stack2 vs production (labelled)", vb, vn, "train", v.select(*KEY, "is_match"))
    # TEST
    base = apply(pl.read_parquet(CACHE / "pred_TEST.parquet", columns=[*KEY, "p"]), cfg)
    for tag, fn in (("stack2", E / "stack2_test" / "pred_TEST_stack2.parquet"), ("stack2fr", E / "stack2fr_test" / "pred_TEST_stack2.parquet")):
        if fn.exists():
            table(f"TEST: {tag} vs production (submitted, LB 0.9515)", base, apply(pl.read_parquet(fn), cfg), "test")


if __name__ == "__main__":
    main()
