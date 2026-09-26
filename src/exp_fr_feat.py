"""Experiment (read-only): which pair features make French candidates score low?
Compares feature distributions of TEST France pairs with labelled VAL positives/negatives, and prints the
normalised fields of low-scored French top candidates.  Usage: python src/exp_fr_feat.py
"""
import polars as pl

from common import CACHE

N = CACHE / "norm"
CAT = ["state_state", "suffix_state", "house_state", "postcode_state", "n_exact", "loc_overlap", "n_first_eq", "acronym"]
NUM = ["n_tset", "n_ratio", "n_full_ratio", "n_idf_overlap", "a_tset", "street_tset", "city_ratio", "owner_rank", "cand_n_owners"]
FLD = ["name_core", "name_suffix", "addr_norm", "house_no", "street_tokens", "city_guess", "state_guess"]


def dist(df, label):
    print(f"\n--- {label}: {df.height:,} pairs")
    print("  means: " + "  ".join(f"{c}={df[c].cast(pl.Float32).mean():.3f}" for c in NUM))
    for c in CAT:
        vc = df[c].cast(pl.Int8).value_counts(normalize=True).sort(c)
        print(f"  {c:15s} " + "  ".join(f"{r[0]}:{r[1]:.3f}" for r in vc.iter_rows()))


def main():
    cols = ["s1_id", "cand_id", "source", *CAT, *NUM]
    ctry = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    ft = (pl.read_parquet(CACHE / "features_TEST.parquet", columns=cols)
          .join(pl.read_parquet(CACHE / "pred_TEST.parquet", columns=["s1_id", "cand_id", "source", "p"]),
                on=["s1_id", "cand_id", "source"]).join(ctry, on="s1_id"))
    top = ft.sort("p", descending=True).group_by("s1_id", maintain_order=True).first()
    fv = pl.read_parquet(CACHE / "features_TRAIN.parquet", columns=[*cols, "role", "is_match"]).filter(pl.col("role") == "val")

    dist(fv.filter(pl.col("is_match")), "VAL true pairs (US+India)")
    dist(fv.filter(~pl.col("is_match") & (pl.col("owner_rank") == 1)), "VAL false pairs, owner_rank 1")
    for c in ("US", "India", "France"):
        dist(ft.filter((pl.col("country") == c) & (pl.col("p") >= 0.5)), f"TEST {c} pairs p>=0.5")
    dist(top.filter((pl.col("country") == "France") & (pl.col("p") < 0.15)), "TEST France TOP candidate with p<0.15 (S1 predicted empty)")
    dist(top.filter((pl.col("country") != "France") & (pl.col("p") < 0.15)), "TEST US+India TOP candidate with p<0.15")

    # normalised fields of low-scored French top candidates
    s1 = pl.read_parquet(N / "test_source1.parquet", columns=["id", "name_raw", *FLD])
    src = {s: pl.read_parquet(N / f"test_source{s}.parquet", columns=["id", "name_raw", *FLD]) for s in (2, 3)}
    low = top.filter((pl.col("country") == "France") & (pl.col("p") < 0.15) & (pl.col("p") > 0.005)).sample(25, seed=7)
    print("\n\n##### low-scored French top candidates (S1 / candidate normalised fields)")
    for r in low.iter_rows(named=True):
        a, b = s1.row(r["s1_id"], named=True), src[r["source"]].row(r["cand_id"], named=True)
        print(f"\np={r['p']:.3f} state={r['state_state']} suffix={r['suffix_state']} house={r['house_state']} "
              f"loc={r['loc_overlap']} n_tset={r['n_tset']:.2f} a_tset={r['a_tset']:.2f} owner_rank={r['owner_rank']}")
        for tag, x in (("S1", a), (f"S{r['source']}", b)):
            print(f"  {tag}: " + " | ".join(f"{k}={x[k]}" for k in ["name_raw", *FLD]))

    # region/department labels used as state in France
    print("\n\n##### France state_guess / city_guess top values")
    for name, f in (("S1", "test_source1"), ("S2", "test_source2"), ("S3", "test_source3")):
        vc = pl.read_parquet(N / f"{f}.parquet", columns=["country", "state_guess", "city_guess"]).filter(pl.col("country") == "France")
        print(name, "state:", vc["state_guess"].value_counts(sort=True).head(12).rows())
        print(name, "city :", vc["city_guess"].value_counts(sort=True).head(12).rows())


if __name__ == "__main__":
    main()
