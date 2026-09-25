"""Phase 4: pair features for every blocked candidate.

Usage:  python src/features.py --split TRAIN [--fit-frac 0.15]   # all VAL S1 + a sample of FIT S1
        python src/features.py --split TEST
Input : cache/candidates_<split>.parquet (+ cache/norm/*)
Output: cache/features_<split>.parquet  (one row per candidate pair; is_match + split column for TRAIN)

Features are country-agnostic string similarities (rapidfuzz, MIT) plus:
  * blocking features carried over (cosines, channel flags, per-S1 maxima/ranks, re-ranker score)
  * competition features computed over ALL queried S1 -- the "best owner" of each S2/S3 record
    (every record has exactly one owner in the ground truth, so a record that scores much higher
    with another S1 is probably not ours). TRAIN queries every train S1, like TEST does.
"""
import argparse

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from common import CACHE, log

N = CACHE / "norm"
POOL = {"TRAIN": "train", "TEST": "test"}
TXT = ["name_core", "name_latin", "name_alt", "name_suffix", "addr_norm", "house_no",
       "city_guess", "state_guess", "street_tokens", "addr_missing"]
CHUNK = 4_000_000


def competition(c):
    """Owner-competition features over the whole candidate table (all S1 queried)."""
    return c.with_columns(
        cand_best=pl.col("score").max().over("cand_id", "source"),
        cand_n_owners=pl.len().over("cand_id", "source").cast(pl.Int16),
        owner_rank=pl.col("score").rank("ordinal", descending=True).over("cand_id", "source").cast(pl.Int16),
        s1_best=pl.col("score").max().over("s1_id", "source"),
        s1_n_strong=(pl.col("score") > 0.5).sum().over("s1_id").cast(pl.Int16),
    ).with_columns(
        gap_owner=pl.col("score") - pl.col("cand_best"),
        gap_s1=pl.col("score") - pl.col("s1_best"))


def sim(a, b, scorer):
    """Pairwise similarity of two equal-length string columns (0..1)."""
    return cpdist(a.fill_null("").to_list(), b.fill_null("").to_list(), scorer=scorer, workers=-1).astype(np.float32) / (
        1.0 if scorer is JaroWinkler.normalized_similarity else 100.0)


def pair_features(d):
    """String features for a joined chunk (left cols *_1 = S1, right = candidate)."""
    f = {}
    a, b = d["name_core_1"], d["name_core"]
    f["n_ratio"] = sim(a, b, fuzz.ratio)
    f["n_tset"] = sim(a, b, fuzz.token_set_ratio)
    f["n_tsort"] = sim(a, b, fuzz.token_sort_ratio)
    f["n_partial"] = sim(a, b, fuzz.partial_ratio)
    f["n_jw"] = sim(a, b, JaroWinkler.normalized_similarity)
    f["n_full_ratio"] = sim(d["name_latin_1"], d["name_latin"], fuzz.token_set_ratio)
    f["n_alt_ratio"] = np.maximum(sim(a, d["name_alt"], fuzz.token_set_ratio),
                                  sim(d["name_alt_1"], b, fuzz.token_set_ratio))
    f["a_tset"] = sim(d["addr_norm_1"], d["addr_norm"], fuzz.token_set_ratio)
    f["a_ratio"] = sim(d["addr_norm_1"], d["addr_norm"], fuzz.ratio)
    f["street_tset"] = sim(d["street_tokens_1"], d["street_tokens"], fuzz.token_set_ratio)
    f["city_ratio"] = sim(d["city_guess_1"], d["city_guess"], fuzz.ratio)
    d = d.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    ta, tb = pl.col("name_core_1").str.split(" "), pl.col("name_core").str.split(" ")
    num = lambda c: pl.col(c).str.extract(r"^(\d+)", 1)
    both = lambda c: pl.col(c + "_1").is_not_null() & pl.col(c).is_not_null() & (pl.col(c) != "") & (pl.col(c + "_1") != "")
    loc1 = pl.concat_list("city_guess_1", "state_guess_1").list.drop_nulls()
    loc2 = pl.concat_list("city_guess", "state_guess").list.drop_nulls()
    return d.with_columns(
        n_exact=pl.col("name_core_1") == pl.col("name_core"),
        n_jacc=(ta.list.set_intersection(tb).list.len() / ta.list.set_union(tb).list.len()).cast(pl.Float32),
        n_first_eq=ta.list.first() == tb.list.first(),
        n_len_ratio=(pl.min_horizontal(pl.col("name_core_1").str.len_chars(), pl.col("name_core").str.len_chars())
                     / pl.max_horizontal(pl.col("name_core_1").str.len_chars(), pl.col("name_core").str.len_chars(), 1)).cast(pl.Float32),
        n_ntok=tb.list.len().cast(pl.Int8),
        suffix_state=pl.when(both("name_suffix")).then(
            pl.when(pl.col("name_suffix_1") == pl.col("name_suffix")).then(2).otherwise(1)).otherwise(0).cast(pl.Int8),
        cand_indic=pl.col("name_raw_c").str.contains(r"[ऀ-෿]"),
        house_state=pl.when(both("house_no")).then(
            pl.when(pl.col("house_no_1") == pl.col("house_no")).then(3)
              .when(num("house_no_1") == num("house_no")).then(2).otherwise(1)).otherwise(0).cast(pl.Int8),
        state_state=pl.when(both("state_guess")).then(
            (pl.col("state_guess_1") == pl.col("state_guess")).cast(pl.Int8) + 1).otherwise(0).cast(pl.Int8),
        loc_overlap=(loc1.list.set_intersection(loc2).list.len() > 0),
    )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="TRAIN", choices=list(POOL))
    ap.add_argument("--fit-frac", type=float, default=0.15, help="share of FIT S1 kept for model training")
    a = ap.parse_args()
    pool = POOL[a.split]
    c = competition(pl.read_parquet(CACHE / f"candidates_{a.split}.parquet"))
    log(f"candidates: {len(c):,}")
    if a.split == "TRAIN":
        sp = pl.read_parquet(CACHE / "splits.parquet").select("s1_id", "rand")
        fit = sp.filter(pl.col("rand") == "FIT").sample(fraction=a.fit_frac, seed=42)
        role = pl.concat([sp.filter(pl.col("rand") == "VAL").select("s1_id", role=pl.lit("val")),
                          fit.select("s1_id", role=pl.lit("fit"))])
        c = c.join(role, on="s1_id", how="left")
        # rows of other S1 competing for the same records as VAL rows: needed for owner exclusivity
        vcand = c.filter(pl.col("role") == "val").select("cand_id", "source").unique()
        comp = c.join(vcand, on=["cand_id", "source"], how="semi").select("s1_id", "cand_id", "source", is_comp=pl.lit(True))
        c = c.join(comp, on=["s1_id", "cand_id", "source"], how="left").with_columns(
            role=pl.coalesce("role", pl.when(pl.col("is_comp")).then(pl.lit("comp")))).drop("is_comp")
        c = c.filter(pl.col("role").is_not_null())
        gt = pl.read_parquet(CACHE / "gt_pairs.parquet").select(
            "s1_id", cand_id="match_id", source="match_source", is_match=pl.lit(True))
        c = c.join(gt, on=["s1_id", "cand_id", "source"], how="left").with_columns(pl.col("is_match").fill_null(False))
        log(f"kept {len(c):,} rows by role {dict(c['role'].value_counts().iter_rows())}; positives {c['is_match'].mean():.3%}")
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "country", *TXT]).rename(
        lambda x: x if x in ("id", "country") else x + "_1").rename({"id": "s1_id"})
    cand = pl.concat([pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id", "name_raw", *TXT])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename(
        {"id": "cand_id", "name_raw": "name_raw_c"})
    out = []
    for i in range(0, len(c), CHUNK):
        d = c.slice(i, CHUNK).join(s1, on="s1_id", how="left").join(cand, on=["cand_id", "source"], how="left")
        d = pair_features(d)
        out.append(d.drop([x for x in d.columns if x.endswith("_1") or x in TXT or x == "name_raw_c"]))
        log(f"  features {min(i + CHUNK, len(c)):,}/{len(c):,}")
    f = pl.concat(out)
    path = CACHE / f"features_{a.split}.parquet"
    f.write_parquet(path)
    log(f"wrote {path.name}: {f.shape}")
    if "is_match" in f.columns:
        num = [x for x in f.columns if x not in ("s1_id", "cand_id", "role", "country", "is_match")]
        print(f.group_by("is_match").agg([pl.col(x).cast(pl.Float32).mean().round(3) for x in num]).transpose(include_header=True))


if __name__ == "__main__":
    main()
