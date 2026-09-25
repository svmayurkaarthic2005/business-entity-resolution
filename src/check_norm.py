"""Phase 2 gate: agreement of normalised fields on VAL true pairs + before/after examples."""
import json

import polars as pl

from common import CACHE, log

N = CACHE / "norm"
FIELDS = ["name_norm", "name_latin", "name_core", "house_no", "postcode", "city_guess",
          "state_guess", "street_tokens"]


def main():
    val = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
    pairs = pl.read_parquet(CACHE / "gt_pairs.parquet").join(val, on="s1_id")
    s1 = pl.read_parquet(N / "train_source1.parquet").rename(lambda c: c + "_1").rename({"id_1": "s1_id"})
    m = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet")
                   .with_columns(match_source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "match_id"})
    j = pairs.join(s1, on="s1_id").join(m, on=["match_id", "match_source"])
    log(f"VAL true pairs: {len(j):,}")
    rep = {}
    for f in FIELDS:
        both = j.filter(pl.col(f).is_not_null() & (pl.col(f) != "") &
                        pl.col(f + "_1").is_not_null() & (pl.col(f + "_1") != ""))
        eq = (both[f] == both[f + "_1"]).mean() if len(both) else None
        rep[f] = {"both_present_pct": round(100 * len(both) / len(j), 1),
                  "equal_when_both_pct": round(100 * eq, 1) if eq is not None else None}
    # location agreement as a set (city/state order is not reliable, esp. France)
    loc = j.with_columns(
        a=pl.concat_list(pl.col("city_guess_1"), pl.col("state_guess_1")).list.drop_nulls(),
        b=pl.concat_list(pl.col("city_guess"), pl.col("state_guess")).list.drop_nulls())
    loc = loc.filter((pl.col("a").list.len() > 0) & (pl.col("b").list.len() > 0))
    rep["location_any_overlap_pct"] = round(100 * (loc["a"].list.set_intersection(loc["b"]).list.len() > 0).mean(), 1)
    rep["indic_match_name_pct"] = round(100 * j["name_raw"].str.contains(r"[ऀ-෿]").mean(), 1)
    ind = j.filter(pl.col("name_raw").str.contains(r"[ऀ-෿]"))
    rep["indic_pairs_core_equal_pct"] = round(100 * (ind["name_core"] == ind["name_core_1"]).mean(), 1)

    # test-time dictionary coverage of Indic tokens
    tdict = json.load(open(N / "norm_maps.json", encoding="utf-8"))["translit"]
    toks = (pl.concat([pl.read_parquet(N / f"test_source{s}.parquet", columns=["name_norm"]) for s in (2, 3)])
              .filter(pl.col("name_norm").str.contains(r"[ऀ-෿]"))["name_norm"]
              .str.split(" ").explode())
    toks = toks.filter(toks.str.contains(r"[ऀ-෿]"))
    rep["test_indic_token_dict_coverage_pct"] = round(100 * toks.is_in(list(tdict)).mean(), 1)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(N / "gate_report.json", "w"), indent=1)

    pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_rows(40)
    cols = ["name_raw", "name_core", "name_suffix", "addr_norm", "house_no", "city_guess", "state_guess"]
    ex = pl.concat([
        pl.read_parquet(N / "train_source2.parquet").filter(pl.col("name_raw").str.contains(r"[ऀ-෿]")).sample(10, seed=1),
        pl.read_parquet(N / "test_source3.parquet").filter(pl.col("country") == "France").sample(10, seed=1),
        pl.read_parquet(N / "train_source3.parquet").filter(pl.col("country") == "US").sample(5, seed=1),
        pl.read_parquet(N / "test_source1.parquet").filter(pl.col("country") == "India").sample(5, seed=1)])
    print(ex.select(cols))


if __name__ == "__main__":
    main()
