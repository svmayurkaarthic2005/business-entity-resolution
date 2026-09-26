"""Forensic audit of the submitted TEST artifacts (Colab side).  Usage: python src/exp_audit.py
Files: SUB = Drive ber_ckpt/SUBMIT_stack2fr (== exp/stack2fr_test) matching_results.tsv + candidate_pairs.tsv;  REF = output_full (first submission, LB 0.952).
Checks: md5/rows, matching subset of candidates per S1, candidate_pairs identical to output_full, pairs added/removed vs output_full by country, candidates per S1 by country,
predicted-pair score distribution by country from exp/stack2fr_test/pred_TEST_stack2.parquet (best p per S1, accepted p), and the same statistics for the VAL rows of the
stack2 model when its prediction file is found.  Read-only.
"""
import glob
import hashlib

import polars as pl

from common import CACHE

B = CACHE.parent
SUB = "/content/drive/MyDrive/ber_ckpt/SUBMIT_stack2fr"
REF = str(B / "output_full")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def pairs(path, col):
    d = pl.read_csv(path, separator="\t", infer_schema_length=0, quote_char=None).with_columns(pl.col(col).fill_null(""))
    return d, d.with_columns(l=pl.col(col).str.split(",")).explode("l").filter(pl.col("l") != "").select(s1=pl.col("source1_entity_id"), e="l")


def main():
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        print(f"{f}: SUB md5 {md5(f'{SUB}/{f}')}   REF(output_full) md5 {md5(f'{REF}/{f}')}")
    t1 = pl.read_csv("/content/data/test_source1.tsv", separator="\t", infer_schema_length=0, quote_char=None, columns=["entity_id", "country"]) if glob.glob("/content/data/test_source1.tsv") \
        else pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"])
    cty = t1.rename({"entity_id": "s1"})
    _, mp = pairs(f"{SUB}/matching_results.tsv", "matched_entity_ids")
    _, cp = pairs(f"{SUB}/candidate_pairs.tsv", "candidate_entity_ids")
    _, rp = pairs(f"{REF}/matching_results.tsv", "matched_entity_ids")
    print(f"\npairs: submitted matches {mp.height:,} | candidate pairs {cp.height:,} | output_full matches {rp.height:,}")
    miss = mp.join(cp, on=["s1", "e"], how="anti")
    print(f"matches NOT in candidate_pairs: {miss.height:,}  (must be 0)")
    ns = cty.height
    cnt = cp.group_by("s1").len("c")
    a = cty.join(cnt, on="s1", how="left").with_columns(pl.col("c").fill_null(0))
    print("\ncandidates per S1 by country (mean / p50 / p90 / max):")
    for r in a.group_by("country").agg(m=pl.col("c").mean(), p50=pl.col("c").median(), p90=pl.col("c").quantile(0.9), mx=pl.col("c").max(), empty=(pl.col("c") == 0).mean()).sort("country").iter_rows():
        print("  ", r)
    add = mp.join(rp, on=["s1", "e"], how="anti").join(cty, on="s1")
    rem = rp.join(mp, on=["s1", "e"], how="anti").join(cty, on="s1")
    print("\nvs output_full (first submission): pairs ADDED by country", dict(add.group_by("country").len().iter_rows()), " REMOVED", dict(rem.group_by("country").len().iter_rows()))
    # score distributions
    pt = f"{B}/exp/stack2fr_test/pred_TEST_stack2.parquet"
    p = pl.read_parquet(pt)
    print("\npred_TEST columns:", p.columns[:12], "rows", p.height)
    ids = pl.read_parquet(CACHE / "test_source1.parquet", columns=["id", "entity_id"]).rename({"id": "s1_id", "entity_id": "s1"}).join(cty, on="s1")
    p = p.join(ids.select("s1_id", "country"), on="s1_id", how="left")
    best = p.group_by("s1_id", "country").agg(b=pl.col("p").max(), sp=pl.col("p").sum(), n=pl.len())
    q = lambda c, x: best.filter(pl.col("country") == c)["b"].quantile(x)
    print("best-candidate p by country (p10/p25/p50/p75; share >=0.9; share <0.15; mean sum p):")
    for c in ("US", "India", "France"):
        s = best.filter(pl.col("country") == c)
        print(f"  {c:7s} {q(c, .1):.3f}/{q(c, .25):.3f}/{q(c, .5):.3f}/{q(c, .75):.3f}  >=.9 {(s['b'] >= .9).mean():.3f}  <.15 {(s['b'] < .15).mean():.3f}  sum p {s['sp'].mean():.2f}  cands {s['n'].mean():.2f}")
    v = sorted(glob.glob(f"{B}/exp/stack2/pred_TRAIN*.parquet"))
    print("\nVAL prediction files for stack2:", v[:4])
    if v:
        pv = pl.read_parquet(v[0]).filter(pl.col("role") == "val")
        bv = pv.group_by("s1_id").agg(b=pl.col("p").max(), sp=pl.col("p").sum(), n=pl.len())
        print(f"  VAL (US+India) best p p10/p25/p50/p75 {bv['b'].quantile(.1):.3f}/{bv['b'].quantile(.25):.3f}/{bv['b'].quantile(.5):.3f}/{bv['b'].quantile(.75):.3f}  >=.9 {(bv['b'] >= .9).mean():.3f}  <.15 {(bv['b'] < .15).mean():.3f}  sum p {bv['sp'].mean():.2f}  cands {bv['n'].mean():.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
