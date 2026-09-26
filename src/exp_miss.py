"""Experiment: why does blocking lose true pairs?  Uses the existing labelled raw union (candidates_FIT_raw.parquet)
-- no new blocking run. Splits true pairs into found / lost-by-K10+floor / never-retrieved, then shows which cheap
exact keys on the normalised fields would have caught the never-retrieved ones and how big those key blocks are.

Usage: python src/exp_miss.py
"""
import polars as pl

from common import CACHE, log
from block import RERANK  # noqa: F401  (import check only)

N = CACHE / "norm"
COLS = ["id", "country", "name_raw", "name_norm", "name_latin", "name_core", "addr_norm", "house_no",
        "postcode", "city_guess", "state_guess", "street_tokens", "addr_missing"]

raw = pl.read_parquet(CACHE / "candidates_FIT_raw.parquet").select("s1_id", "cand_id", "source")
q = pl.read_parquet(CACHE / "queried_FIT.parquet")
gt = pl.read_parquet(CACHE / "gt_pairs.parquet").rename({"match_id": "cand_id", "match_source": "source"}).join(q, on="s1_id")
gt = gt.join(raw.with_columns(in_union=pl.lit(True)), on=["s1_id", "cand_id", "source"], how="left").with_columns(
    pl.col("in_union").fill_null(False))
log(f"{q.height:,} S1, {gt.height:,} true pairs, union recall {gt['in_union'].mean():.4f}")

s1 = pl.read_parquet(N / "train_source1.parquet", columns=COLS).rename(lambda c: c + "_1").rename({"id_1": "s1_id"})
pool = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=COLS).with_columns(source=pl.lit(s, pl.Int8))
                  for s in (2, 3)]).rename({"id": "cand_id"})
m = gt.join(s1, on="s1_id").join(pool, on=["cand_id", "source"])

first = lambda c: pl.col(c).str.split(" ").list.first()
last = lambda c: pl.col(c).str.split(" ").list.last()
ne = lambda c: (pl.col(c).is_not_null()) & (pl.col(c) != "") & (pl.col(c + "_1").is_not_null()) & (pl.col(c + "_1") != "")
m = m.with_columns(
    indic=pl.col("name_raw").str.contains(r"[ऀ-෿]"),
    same_core=pl.col("name_core") == pl.col("name_core_1"),
    same_first=first("name_core") == first("name_core_1"),
    same_last=last("name_core") == last("name_core_1"),
    tok_overlap=pl.col("name_core").str.split(" ").list.set_intersection(pl.col("name_core_1").str.split(" ")).list.len() > 0,
    same_house=ne("house_no") & (pl.col("house_no") == pl.col("house_no_1")),
    same_city=ne("city_guess") & (pl.col("city_guess") == pl.col("city_guess_1")),
    same_state=ne("state_guess") & (pl.col("state_guess") == pl.col("state_guess_1")),
    same_street=ne("street_tokens") & (pl.col("street_tokens") == pl.col("street_tokens_1")),
    same_pc=ne("postcode") & (pl.col("postcode") == pl.col("postcode_1")),
    same_addr=ne("addr_norm") & (pl.col("addr_norm") == pl.col("addr_norm_1")))

lost = m.filter(~pl.col("in_union"))
print(f"\nnever retrieved: {lost.height:,} of {m.height:,} pairs ({lost.height / m.height:.2%})")
print("share of misses vs share of all true pairs, by group:")
for g in ("country", "source", "indic", "addr_missing"):
    a = lost.group_by(g).len().sort(g)
    b = m.group_by(g).agg(n=pl.len(), miss=(~pl.col("in_union")).mean()).sort(g)
    print(g, [(r[g], f"{r['miss']:.3%}", r["n"]) for r in b.iter_rows(named=True)])
flags = ["same_core", "same_first", "same_last", "tok_overlap", "same_house", "same_city", "same_state",
         "same_street", "same_pc", "same_addr"]
print("\nfraction of never-retrieved pairs with each property (vs all pairs):")
for f in flags:
    print(f"  {f:12s} lost {lost[f].mean():.3f}   all {m[f].mean():.3f}")
key_sets = {
    "core-equal": ["same_core"],
    "first-tok+city": ["same_first", "same_city"],
    "first-tok+state": ["same_first", "same_state"],
    "last-tok+city": ["same_last", "same_city"],
    "house+city": ["same_house", "same_city"],
    "house+street": ["same_house", "same_street"],
    "addr-equal": ["same_addr"],
    "tok-overlap+house": ["tok_overlap", "same_house"],
    "tok-overlap+city+street": ["tok_overlap", "same_city", "same_street"],
    "tok-overlap+addr": ["tok_overlap", "same_addr"],
}
print("\ncatchable share of the never-retrieved pairs by a candidate exact key:")
for k, fs in key_sets.items():
    e = pl.all_horizontal(*[pl.col(f) for f in fs])
    print(f"  {k:26s} {lost.filter(e).height / lost.height:.3f}")
any_key = pl.any_horizontal(*[pl.all_horizontal(*[pl.col(f) for f in fs]) for fs in key_sets.values()])
print(f"  ANY of the above          {lost.filter(any_key).height / lost.height:.3f}")
pl.Config.set_fmt_str_lengths(60)
print("\nsamples of never-retrieved pairs:")
for r in lost.sample(min(25, lost.height), seed=3).iter_rows(named=True):
    print(f"S1 {r['name_raw_1']} | {r['addr_norm_1']}\nS{r['source']} {r['name_raw']} | {r['addr_norm']}\n")
