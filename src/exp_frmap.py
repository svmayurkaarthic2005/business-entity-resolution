"""France qualifier-token map.  Usage: python src/exp_frmap.py     (writes exp/frmap/ and exp/stack2fr_test/; output_full untouched)

The biggest feature family (QUAL: OOF log-odds of name words present on one side only) is learned on US/India words.  The generator
adds the SAME kind of sibling-distractor words in France, but in French (groupe, participations, fils, associes, city names ...),
which are not in the vocab -> log-odds 0 -> France siblings look like matches and all French scores drop.
Part 1 (labelled estimate, VAL US only): zero the English equivalents in the vocab (= what France sees today) and measure the F0.5 loss.
Part 2 (TEST): map French tokens to the English equivalents for France rows only, rebuild the stack2 TEST file in exp/stack2fr_test/,
and compare France/US/India prediction statistics with the current stack2 file.
"""
import json
import sys

import numpy as np
import polars as pl

import exp_stack as S
import exp_stack2 as E
import exp_stack2_test as T
from common import CACHE, log
from decide import apply
from exp_budget import val_truth
from exp_errbudget import macro_f

N = CACHE / "norm"
KEY = S.KEY
CITY = ["bordeaux", "nantes", "lille", "tourcoing", "dunkerque", "roubaix", "calais", "nazaire", "pessac", "teste", "buch",
        "merignac", "lege", "ferret", "pornic", "baule", "escoublac", "herblain"]
FRMAP = {"groupe": "group", "groupement": "group", "participations": "holdings", "holding": "holdings", "associes": "associates",
         "fils": "sons", "freres": "sons", "developpement": "development", "france": "usa", "et": "and", "cie": "co",
         **{c: "downtown" for c in CITY}}
LOC = ["downtown", "midtown", "uptown", "northside", "southside", "eastgate", "westgate", "riverside", "lakeside", "central",
       "north", "south", "east", "west", "metro", "harbor", "valley", "coastal", "summit", "greater"]
ZERO = sorted(set(FRMAP.values()) - {"and"} | set(LOC))       # 'and' is already shared by both languages


def map_fr(t, country):
    t = t.join(country, on="s1_id", how="left")
    fr = pl.col("country") == "France"
    rep = lambda c: pl.when(fr).then(pl.col(c).list.eval(pl.element().replace(FRMAP)).list.unique()).otherwise(pl.col(c))
    return t.with_columns(X=rep("X"), M=rep("M")).drop("country")


def val_estimate(vocab):
    # ponytail: VAL rows only, so owner competition from non-VAL S1 is missing -> read the two rows as a RELATIVE difference
    d = (pl.read_parquet(CACHE.parent / "exp" / "stack3" / "feats_TRAIN.parquet").filter(pl.col("role") == "val")
           .with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32)))
    t = E.tok_num(d.select(KEY), "train")
    cty = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    _, truth_n, _ = val_truth()
    us = truth_n.join(cty, on="s1_id").filter(pl.col("country") == "US").select(truth_n.columns)
    zv = {k: v.with_columns(lo=pl.when(pl.col("tok").is_in(ZERO)).then(0.0).otherwise(pl.col("lo"))) for k, v in vocab.items()}
    for name, voc in (("full vocab (today's US)", vocab), ("English sibling words zeroed (today's France)", zv)):
        q = T.qual_test(t.select(*KEY, "X", "M"), voc)
        x = d.drop("x_lo_min", "x_lo_sum", "m_lo_min", "m_lo_sum").join(q, on=KEY, how="left")
        pred = x.select(*KEY, "is_match").with_columns(p=pl.Series(T.score(x, "s2new_india")))
        kept = apply(pred, cfg)
        tp, k = kept.filter(pl.col("is_match")).group_by("s1_id").len("tp"), kept.group_by("s1_id").len("k")
        print(f"  VAL {name:48s} all {macro_f(truth_n, tp, k):.5f}   US {macro_f(us, tp, k):.5f}   pred/S1 {kept.height / truth_n.height:.3f}")


def main():
    vocab = {k: pl.read_parquet(T.S2 / f"vocab_{k}.parquet") for k in ("X", "M")}
    print("in vocab:", vocab["X"].filter(pl.col("tok").is_in(ZERO)).sort("lo").with_columns(pl.col("lo").round(2)).rows())
    log("part 1: VAL estimate")
    val_estimate(vocab)
    if "--val-only" in sys.argv:
        return
    log("part 2: TEST with France token map")
    T.DEST = CACHE.parent / "exp" / "stack2fr_test"
    cty = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    orig = T.qual_test
    T.qual_test = lambda t, v: orig(map_fr(t, cty), v)
    T.main()
    # record assignment rate per country: share of S2/S3 records given to some S1 (same generator -> should be similar)
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    for tag, f in (("stack2", CACHE.parent / "exp" / "stack2_test" / "pred_TEST_stack2.parquet"),
                   ("stack2fr", T.DEST / "pred_TEST_stack2.parquet")):
        k = apply(pl.read_parquet(f), cfg).join(cty, on="s1_id")
        for c in ("France", "India", "US"):
            n = sum(pl.read_parquet(N / f"test_source{s}.parquet", columns=["country"]).filter(pl.col("country") == c).height for s in (2, 3))
            print(f"  {tag:9s} {c:7s} records assigned {k.filter(pl.col('country') == c).select('cand_id', 'source').n_unique() / n:.3f}")


if __name__ == "__main__":
    main()
