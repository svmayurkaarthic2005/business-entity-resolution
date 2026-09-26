"""Experiment (read-only): France candidate-cap check + VAL error budget + false-negative breakdown.

Usage: python src/exp_budget.py
A. cap check (TEST): per country/source share of (S1,source) groups that fill K=10, re-ranker score of the 10th
   candidate, and the expected number of true pairs just beyond rank 10, estimated from the labelled FIT raw union
   (held-out S1 of the re-ranker) as a function of the 10th score.
B. error budget (VAL, labelled): current F0.5 vs oracles -> losses from blocking (never retrieved / truncated),
   model ranking, decision rule.
C. false negatives on VAL by country and pair attributes.
"""
import json
import pickle

import numpy as np
import polars as pl

from block import FEATS
from common import CACHE, log
from decide import apply, owner_adjust
from exp_decide import per_s1

KEY = ["s1_id", "cand_id", "source"]
FLOOR = 0.003
BINS = [0.003, 0.01, 0.05, 0.2, 0.5]


def val_truth():
    val_ids = (pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
               .join(pl.read_parquet(CACHE / "queried_TRAIN.parquet"), on="s1_id"))
    truth_n = val_ids.join(pl.read_parquet(CACHE / "gt_s1.parquet").rename({"n_matches": "n"}), on="s1_id").sort("s1_id")
    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").rename({"match_id": "cand_id", "match_source": "source"}).join(val_ids, on="s1_id")
    return val_ids, truth_n, gt


def qs(s, q=(0.1, 0.25, 0.5, 0.75, 0.9)):
    s = s.drop_nulls()
    return "/".join(f"{s.quantile(x):.3f}" for x in q) if len(s) else "-"


def binned(col):
    return pl.col(col).cut(BINS, labels=["a <.003", "b .003-.01", "c .01-.05", "d .05-.2", "e .2-.5", "f >=.5"]).cast(pl.Utf8)


def cap_check():
    print("=" * 30, "A. candidate-cap check (TEST)")
    ctry = pl.read_parquet(CACHE / "norm" / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    ct = pl.read_parquet(CACHE / "candidates_TEST.parquet", columns=["s1_id", "source", "rank", "score"])
    g = ct.group_by("s1_id", "source").agg(n=pl.len(), s10=pl.col("score").filter(pl.col("rank") == 9).first()).join(ctry, on="s1_id")
    for c in ("France", "India", "US"):
        for s in (2, 3):
            x = g.filter((pl.col("country") == c) & (pl.col("source") == s))
            full = x.filter(pl.col("n") >= 10)
            print(f"{c:7s} S{s}: groups {x.height:>8,} | reach 10: {full.height / x.height:6.1%} | 10th score p10/25/50/75/90 {qs(full['s10'])} | "
                  f"10th>=0.05: {(x['s10'] >= 0.05).sum() / x.height:6.2%}  >=0.2: {(x['s10'] >= 0.2).sum() / x.height:6.2%}")
        s1 = g.filter(pl.col("country") == c).group_by("s1_id").agg(full=(pl.col("n") >= 10).any(), hot=(pl.col("s10") >= 0.05).any())
        print(f"{c:7s} S1 with a full source: {s1['full'].mean():.1%}; with a full source whose 10th score >= 0.05: {s1['hot'].mean():.2%}")

    # labelled estimate of what lies beyond rank 10, as a function of the 10th score (held-out FIT S1)
    raw = pl.read_parquet(CACHE / "candidates_FIT_raw.parquet")
    hold = raw["s1_id"].unique().sort().sample(fraction=0.2, seed=42)
    raw = raw.filter(pl.col("s1_id").is_in(hold))
    model = pickle.load(open(CACHE / "rerank.pkl", "rb"))
    raw = raw.with_columns(score=pl.Series(model.predict_proba(raw.select(FEATS).to_numpy().astype(np.float32))[:, 1]))
    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").rename({"match_id": "cand_id", "match_source": "source"}).join(
        raw.select("s1_id").unique(), on="s1_id")
    raw = raw.join(gt.with_columns(is_match=pl.lit(True)), on=KEY, how="left").with_columns(pl.col("is_match").fill_null(False))
    raw = raw.with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("s1_id", "source") - 1)
    ok = pl.col("is_match") & (pl.col("score") >= FLOOR)
    grp = raw.group_by("s1_id", "source").agg(
        s10=pl.col("score").filter(pl.col("rank") == 9).first(),
        full=((pl.col("rank") < 10) & (pl.col("score") >= FLOOR)).sum() >= 10,
        b15=(ok & pl.col("rank").is_between(10, 14)).sum(), b20=(ok & pl.col("rank").is_between(10, 19)).sum(),
        b30=(ok & pl.col("rank").is_between(10, 29)).sum())
    n_true, in_union = gt.height, raw["is_match"].sum()
    kept = raw.filter(ok & (pl.col("rank") < 10)).height
    trunc_k = raw.filter(pl.col("is_match") & (pl.col("rank") >= 10)).height
    trunc_f = raw.filter(pl.col("is_match") & (pl.col("rank") < 10) & (pl.col("score") < FLOOR)).height
    split = {"never_retrieved": 1 - in_union / n_true, "cut_by_K": trunc_k / n_true, "cut_by_floor": trunc_f / n_true,
             "kept": kept / n_true}
    print(f"\nFIT held-out ({hold.len():,} S1, {n_true:,} true pairs): " + ", ".join(f"{k} {v:.2%}" for k, v in split.items()))
    rate = grp.filter(pl.col("full")).with_columns(bin=binned("s10")).group_by("bin").agg(
        groups=pl.len(), b15=pl.col("b15").mean(), b20=pl.col("b20").mean(), b30=pl.col("b30").mean()).sort("bin")
    print("true pairs recovered per full group by raising K, by 10th-score bin (US/India labelled):")
    print(rate)
    s1_n = raw["s1_id"].n_unique()
    print(f"FIT held-out true pairs recoverable per S1: K15 {grp['b15'].sum() / s1_n:.4f}  K20 {grp['b20'].sum() / s1_n:.4f}  K30 {grp['b30'].sum() / s1_n:.4f}")

    full = g.filter(pl.col("n") >= 10).with_columns(bin=binned("s10")).join(rate.select("bin", "b15", "b20", "b30"), on="bin", how="left")
    print("\nexpected true pairs recovered per S1 on TEST (FIT rates applied to each country's 10th-score distribution)")
    for c in ("France", "India", "US"):
        x = full.filter(pl.col("country") == c)
        n = ctry.filter(pl.col("country") == c).height
        e = {k: x[k].fill_null(0).sum() / n for k in ("b15", "b20", "b30")}
        # a recovered pair lifts one S1 from F=1.25(n-1)/(1.25n-1) to 1 (n=3.47): +0.075 upper bound
        print(f"  {c:7s} K15 {e['b15']:.4f}  K20 {e['b20']:.4f}  K30 {e['b30']:.4f} pairs/S1 -> F0.5 upper bound "
              f"+{0.075 * e['b20']:.4f} (K20), +{0.075 * e['b30']:.4f} (K30); leaderboard share {n / ctry.height:.1%}")
    return split


def budget(split):
    print("\n" + "=" * 30, "B. error budget (VAL, labelled US+India)")
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    val_ids, truth_n, gt = val_truth()
    pfull = pl.read_parquet(CACHE / "pred_TRAIN.parquet")
    pa = owner_adjust(pfull, cfg["alpha"]).filter(pl.col("role") == "val")
    kept = apply(pfull, cfg).filter(pl.col("role") == "val")
    cur = per_s1(kept, truth_n)
    f_oc = per_s1(pa.filter(pl.col("is_match")), truth_n)[0].mean()
    d = (pa.sort(["s1_id", "pa"], descending=[False, True])
           .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1_id"), tp=pl.col("is_match").cast(pl.Int32).cum_sum().over("s1_id"))
           .join(truth_n, on="s1_id")
           .with_columns(f=pl.when(pl.col("n") == 0).then(0.0).otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n") + pl.col("k")))))
    t = truth_n.join(d.group_by("s1_id").agg(fm=pl.col("f").max()), on="s1_id", how="left").with_columns(
        fb=pl.max_horizontal(pl.col("fm").fill_null(0.0), (pl.col("n") == 0).cast(pl.Float64)))
    f_ok = t["fb"].mean()
    f_cur = cur[0].mean()
    lost = split["never_retrieved"] + split["cut_by_K"] + split["cut_by_floor"]
    blk = 1 - f_oc
    print(f"current macro F0.5 {f_cur:.5f} (P {cur[1].mean():.4f} R {cur[2].mean():.4f})")
    print(f"oracle: perfect choice among current candidates {f_oc:.5f}; best prefix of the model's ranking {f_ok:.5f}")
    print(f"LOSS (points of F0.5, total {1 - f_cur:.5f}):")
    print(f"  blocking            {blk:.5f}  (never retrieved ~{blk * split['never_retrieved'] / lost:.5f}, "
          f"top-K cut ~{blk * split['cut_by_K'] / lost:.5f}, floor cut ~{blk * split['cut_by_floor'] / lost:.5f})")
    print(f"  model ranking       {f_oc - f_ok:.5f}")
    print(f"  decision/threshold  {f_ok - f_cur:.5f}")
    ctry = pl.read_parquet(CACHE / "norm" / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    tc = truth_n.with_columns(f=pl.Series(cur[0])).join(ctry, on="s1_id")
    print("per-country VAL F0.5:", {c: round(v, 5) for c, v in tc.group_by("country").agg(pl.col("f").mean()).iter_rows()})
    return cfg, pfull, pa, kept, gt


def fn_breakdown(pa, kept, gt):
    print("\n" + "=" * 30, "C. false negatives on VAL")
    attrs = ["n_tset", "a_tset", "state_state", "city_ratio", "house_state", "suffix_state", "cand_indic", "am", "owner_rank", "country"]
    fv = pl.read_parquet(CACHE / "features_TRAIN.parquet", columns=[*KEY, "role", *attrs]).filter(pl.col("role") == "val").drop("role")
    t = (gt.join(pa.select(*KEY, "pa", in_c=pl.lit(True)), on=KEY, how="left")
           .join(kept.select(*KEY, kp=pl.lit(True)), on=KEY, how="left").join(fv, on=KEY, how="left")
           .with_columns(status=pl.when(pl.col("kp")).then(pl.lit("TP")).when(pl.col("in_c")).then(pl.lit("FN_model"))
                         .otherwise(pl.lit("FN_block"))))
    print("true pairs by status:", dict(t.group_by("status").len().sort("status").iter_rows()))
    ctry = pl.read_parquet(CACHE / "norm" / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id", "country": "c1"})
    t = t.join(ctry, on="s1_id")
    print("status share by country:", t.group_by("c1", "status").len().sort("c1", "status").rows())
    # blocking misses: candidate side attributes from the normalised files
    meta = pl.concat([pl.read_parquet(CACHE / "norm" / f"train_source{s}.parquet", columns=["id", "addr_missing", "name_raw"])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    b = t.join(meta, on=["cand_id", "source"]).with_columns(indic=pl.col("name_raw").str.contains(r"[ऀ-෿]"))
    for col in ("addr_missing", "indic"):
        r = b.group_by(col).agg(n=pl.len(), block_miss=(pl.col("status") == "FN_block").mean(), model_miss=(pl.col("status") == "FN_model").mean()).sort(col)
        print(f"  miss rates by candidate {col}: {r.rows()}")
    ic = t.filter(pl.col("in_c")).with_columns(
        n_bin=pl.col("n_tset").cut([0.5, 0.8, 0.95]), a_bin=pl.col("a_tset").cut([0.5, 0.8, 0.95]),
        city_bin=pl.col("city_ratio").cut([0.5, 0.9]), owner2=pl.col("owner_rank") > 1)
    print("\nretrieved true pairs: model-FN rate and share of all model-FNs, by attribute")
    nfn = ic.filter(pl.col("status") == "FN_model").height
    for col in ("n_bin", "a_bin", "state_state", "city_bin", "house_state", "suffix_state", "cand_indic", "am", "owner2"):
        r = ic.group_by(col).agg(n=pl.len(), fn_rate=(pl.col("status") == "FN_model").mean(),
                                 fn_share=(pl.col("status") == "FN_model").sum() / nfn).sort(col)
        print(f"  {col:12s} " + "  ".join(f"{str(a)}: n={n:,} fn={fr:.1%} share={fs:.1%}" for a, n, fr, fs in r.iter_rows()))
    fp = kept.filter(~pl.col("is_match"))
    print(f"\nfalse positives kept on VAL: {fp.height:,} (vs {t.filter(pl.col('status') == 'TP').height:,} TP)")


def main():
    split = cap_check()
    cfg, pfull, pa, kept, gt = budget(split)
    fn_breakdown(pa, kept, gt)


if __name__ == "__main__":
    main()
