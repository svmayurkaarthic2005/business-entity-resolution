"""Experiment: name-AMBIGUITY features (NEW4) on top of the cached stack3 feature table, plus stage 3.  VAL held-out, exp/stack4/.
Usage: python src/exp_stack4.py     (needs exp/stack3/feats_TRAIN.parquet written by exp_stack3.py)

Ablation showed 3 simple name-frequency counts are worth +0.0024, so this adds a fuller ambiguity family:
  amb_*  : how many S1 records share the candidate's core name (country / country+city), how many pool records share the S1's core name
           (country / country+city), how many S1 records in the city share the first name token
  idf_*  : rarest / mean token IDF of both names, max IDF among SHARED tokens and among tokens present on ONE side only
  ctx_*  : within the S1's candidate list: number of candidates with the same core name as this one, and this candidate's rank
           by house-number agreement among them
Variants: ref = stage2+NUM+QUAL+NEW3 (all), +NEW4, stage3(+NEW4 +owner-share).  Held-out split-halves vs base and vs ref.
"""
import pickle

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
from common import CACHE, log

N = CACHE / "norm"
OUT4 = CACHE.parent / "exp" / "stack4"
KEY = S.KEY
NEW4 = ["amb_s1_core_cty", "amb_s1_core_city", "amb_pool_core_cty", "amb_pool_core_city", "amb_s1_tok1_city",
        "idf_c_min", "idf_c_mean", "idf_s_min", "idf_s_mean", "idf_shared_max", "idf_diff_max", "ctx_same_core", "ctx_hn_rank"]


def ambiguity(d, pool):
    cand = pl.concat([pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id", "country", "name_core", "city_guess", "house_no"])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename(
        {"id": "cand_id", "name_core": "n2", "city_guess": "cy2", "house_no": "hn2"})
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "name_core", "city_guess", "house_no"]).rename(
        {"id": "s1_id", "name_core": "n1", "city_guess": "cy1", "house_no": "hn1"})
    s1c = s1.join(pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"}), on="s1_id")
    first = lambda c: pl.col(c).fill_null("").str.split(" ").list.first()
    f_core = s1c.group_by("country", "n1").len("amb_s1_core_cty").rename({"n1": "n2"})
    f_corecity = s1c.group_by("country", "n1", "cy1").len("amb_s1_core_city").rename({"n1": "n2", "cy1": "cy2"})
    f_tok = s1c.with_columns(t1=first("n1")).group_by("country", "t1", "cy1").len("amb_s1_tok1_city")
    p_core = cand.group_by("country", "n2").len("amb_pool_core_cty").rename({"n2": "n1"})
    p_corecity = cand.group_by("country", "n2", "cy2").len("amb_pool_core_city").rename({"n2": "n1", "cy2": "cy1"})
    t = (d.select(*KEY, "country").join(cand.select("cand_id", "source", "n2", "cy2", "hn2"), on=["cand_id", "source"], how="left")
          .join(s1, on="s1_id", how="left").with_columns(t1=first("n1"), t2=first("n2")))
    t = (t.join(f_core, on=["country", "n2"], how="left").join(f_corecity, on=["country", "n2", "cy2"], how="left")
          .join(p_core, on=["country", "n1"], how="left").join(p_corecity, on=["country", "n1", "cy1"], how="left")
          .join(f_tok.rename({"t1": "t2", "cy1": "cy2"}), on=["country", "t2", "cy2"], how="left"))
    # token IDF (document frequency over S1 core names)
    toks = s1.select(pl.col("n1").fill_null("").str.split(" ").list.unique().alias("tok")).explode("tok").drop_nulls("tok")
    idf = toks.group_by("tok").len().with_columns(idf=(len(s1) / pl.col("len")).log().cast(pl.Float32)).select("tok", "idf")
    big = float(np.log(len(s1)))
    tk = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    tt = t.select(*KEY, T1=tk("n1"), T2=tk("n2")).with_columns(sh=pl.col("T1").list.set_intersection("T2"),
                                                              di=pl.col("T1").list.set_symmetric_difference("T2"))
    tt = tt.with_row_index("r")

    def agg(col, name):
        e = (tt.select("r", col).explode(col).drop_nulls(col).join(idf, left_on=col, right_on="tok", how="left")
               .with_columns(pl.col("idf").fill_null(big)))
        return e.group_by("r").agg(**{f"{name}_min": pl.col("idf").min(), f"{name}_mean": pl.col("idf").mean(), f"{name}_max": pl.col("idf").max()})
    a2, a1, sh, di = agg("T2", "c"), agg("T1", "s"), agg("sh", "shr"), agg("di", "dif")
    tt = tt.select("r").join(a2, on="r", how="left").join(a1, on="r", how="left").join(sh, on="r", how="left").join(di, on="r", how="left").sort("r")
    out = t.select(*KEY, "amb_s1_core_cty", "amb_s1_core_city", "amb_pool_core_cty", "amb_pool_core_city", "amb_s1_tok1_city")
    out = out.with_columns(idf_c_min=tt["c_min"], idf_c_mean=tt["c_mean"], idf_s_min=tt["s_min"], idf_s_mean=tt["s_mean"],
                           idf_shared_max=tt["shr_max"], idf_diff_max=tt["dif_max"])
    out = out.with_columns([pl.col(c).fill_null(0) for c in ("amb_s1_core_cty", "amb_s1_core_city", "amb_pool_core_cty", "amb_pool_core_city",
                                                              "amb_s1_tok1_city")])
    # candidate-list context: same-core siblings inside this S1's list; house-number agreement rank among them
    hn = t.select(*KEY, "n2", hn_eq=(pl.col("hn1").fill_null("") != "") & (pl.col("hn1") == pl.col("hn2")))
    ctx = hn.with_columns(ctx_same_core=pl.len().over("s1_id", "n2").cast(pl.Int16),
                          ctx_hn_rank=pl.col("hn_eq").cast(pl.Int8).rank("dense", descending=True).over("s1_id", "n2").cast(pl.Int16))
    return out.join(ctx.select(*KEY, "ctx_same_core", "ctx_hn_rank"), on=KEY, how="left")


def main():
    OUT4.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUT4
    X.OUT3 = OUT4          # train_oof (exp_stack3) saves models here
    d = pl.read_parquet(CACHE.parent / "exp" / "stack3" / "feats_TRAIN.parquet")
    fit = d.filter(pl.col("role") == "fit")
    folds = list(GroupKFold(3).split(np.zeros(fit.height), fit["is_match"].to_numpy(), fit["s1_id"].to_numpy()))
    fit_ids = fit.select("s1_id")
    del fit
    log(f"loaded feature table {d.shape}")
    d = d.join(ambiguity(d, "train"), on=KEY, how="left")
    log("NEW4 features done")
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    s3all = base_cols + S.NEW + E.NEW2 + X.GA + X.GC + X.GD
    preds = {"base (production model)": d.select(*KEY, "role", "is_match", p="p1")}
    log("training: ref (all NEW3)")
    preds["ref stage2+all NEW3"], _ = X.train_oof(d, s3all, folds, fit_ids, "v_ref")
    log("training: +NEW4")
    preds["+NEW4"], p2 = X.train_oof(d, s3all + NEW4, folds, fit_ids, "v_new4")
    log("stage 3 with NEW4 + owner-share")
    d3 = d.drop([c for c in S.NEW if c in d.columns] + ["p1"]).with_columns(p1=p2)
    d3 = S.stage2_features(d3).with_columns(p2=p2)
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    d3 = d3.with_columns(o_sum=pl.col("p2").sum().over("cand_id", "source"), g_src_sum=pl.col("p2").sum().over("s1_id", "source"))
    d3 = d3.with_columns(o_share=pl.col("p2") / pl.col("o_sum").clip(lower_bound=1e-6), g_share=pl.col("p2") / pl.col("g_src_sum").clip(lower_bound=1e-6),
                         o_logit_gap=lg("p2") - lg("o_other_best"), g_logit_gap=lg("p2") - lg("g_other_best"))
    SHARE = ["o_share", "g_share", "o_logit_gap", "g_logit_gap"]
    preds["stage3 (+NEW4+share)"], _ = X.train_oof(d3, s3all + NEW4 + ["p2"] + SHARE, folds, fit_ids, "v_stage3")
    for k, v in preds.items():
        v.write_parquet(OUT4 / f"pred_TRAIN_{''.join(ch if ch.isalnum() else '_' for ch in k)}.parquet")
    S.evaluate(preds, ref="ref stage2+all NEW3")


if __name__ == "__main__":
    main()
