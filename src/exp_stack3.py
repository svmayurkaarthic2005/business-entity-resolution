"""Experiment: more error-mode features + a stage 3 (iterated stacking). VAL held-out, exp/stack3/.  Usage: python src/exp_stack3.py

NEW3 features (all computed for every candidate row):
  compact/leet   name_core with letter-digit-letter digits mapped to letters (r0ute->route, Banga1ore->Bangalore), spaces/punct removed:
                 c_ratio, c_partial, c_lev, c_contains, acr_ratio (candidate vs INITIALS of the S1 name), c_len_min
  house number   hn_absdiff (|first number of house_no| difference), hn_lev (Levenshtein between house_no strings)
  address words  ad_* : out-of-fold log-odds of address WORDS present on only one side (same construction as the name qualifiers)
  suffix pair    suf_pair_lo / suf_pair_n : out-of-fold log-odds of the (S1 suffix, candidate suffix) pair
Stage 3: recompute the group / owner / support / rarity features from stage-2 scores p2 (OUT-OF-FOLD for fit rows) and train again.
Variants (no is_india flag): ref = stage2+NUM+QUAL (0.97829 before); +A(compact,house) ; +C(addr words) ; +D(suffix pair) ; +all NEW3 ; stage3(+all).
"""
import pickle

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import exp_stack as S
import exp_stack2 as E
from common import CACHE, log
from train import engine, fit_one

N = CACHE / "norm"
OUT3 = CACHE.parent / "exp" / "stack3"
KEY = S.KEY
GA = ["c_ratio", "c_partial", "c_lev", "c_contains", "acr_ratio", "c_len_min", "hn_absdiff", "hn_lev"]
GC = ["ad_x_n", "ad_m_n", "ad_x_lo_min", "ad_x_lo_sum", "ad_m_lo_min", "ad_m_lo_sum"]
GD = ["suf_pair_lo", "suf_pair_n"]
LEET = (("0", "o"), ("1", "l"), ("3", "e"), ("4", "a"), ("5", "s"), ("8", "b"))


def compact_expr(c):
    s = pl.col(c).fill_null("")
    for dg, lt in LEET:
        for _ in range(2):  # overlapping neighbours
            s = s.str.replace_all(rf"([a-z]){dg}([a-z])", rf"${{1}}{lt}${{2}}")
    return s.str.replace_all(r"[^a-z0-9]", "")


def sim(a, b, scorer, div=100.0):
    return cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32) / div


def tables(pool):
    cand = pl.concat([pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id", "name_core", "name_suffix", "house_no", "addr_norm"])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename(
        {"id": "cand_id", "name_core": "n2", "name_suffix": "suf2", "house_no": "hn2", "addr_norm": "ad2"})
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "name_core", "name_suffix", "house_no", "addr_norm"]).rename(
        {"id": "s1_id", "name_core": "n1", "name_suffix": "suf1", "house_no": "hn1", "addr_norm": "ad1"})
    return s1, cand


def compact_features(d, pool):
    s1, cand = tables(pool)
    t = d.select(KEY).join(cand, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
    ini = pl.col("n1").fill_null("").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    t = t.with_columns(a=compact_expr("n1"), b=compact_expr("n2"), ini=ini)
    a, b, i = t["a"].to_list(), t["b"].to_list(), t["ini"].to_list()
    c_partial = sim(a, b, fuzz.partial_ratio)
    la, lb = t["a"].str.len_chars().to_numpy(), t["b"].str.len_chars().to_numpy()
    lmin = np.minimum(la, lb)
    acr = sim(b, i, fuzz.ratio)
    acr[lb > 6] = np.nan
    hn1, hn2 = t["hn1"].fill_null("").to_list(), t["hn2"].fill_null("").to_list()
    both = ((t["hn1"].fill_null("") != "") & (t["hn2"].fill_null("") != "")).to_numpy()
    n1 = t["hn1"].str.extract(r"(\d+)").cast(pl.Int64, strict=False).to_numpy(allow_copy=True).astype(float)
    n2 = t["hn2"].str.extract(r"(\d+)").cast(pl.Int64, strict=False).to_numpy(allow_copy=True).astype(float)
    hn_abs = np.clip(np.abs(n1 - n2), 0, 1000).astype(np.float32)
    hn_lev = cpdist(hn1, hn2, scorer=Levenshtein.distance, workers=-1).astype(np.float32)
    hn_lev[~both] = np.nan
    return t.select(*KEY).with_columns(
        c_ratio=pl.Series(sim(a, b, fuzz.ratio)), c_partial=pl.Series(c_partial), c_lev=pl.Series(sim(a, b, Levenshtein.normalized_similarity, 1.0)),
        c_contains=pl.Series(((c_partial >= 1.0) & (lmin >= 4)).astype(np.int8)), acr_ratio=pl.Series(acr),
        c_len_min=pl.Series(lmin.astype(np.int16)), hn_absdiff=pl.Series(hn_abs), hn_lev=pl.Series(hn_lev))


def addr_words(d, pool):
    s1, cand = tables(pool)
    t = d.select(KEY).join(cand, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
    toks = lambda c: (pl.col(c).fill_null("").str.replace_all(r"\d+", " ").str.replace_all(r"[,]", " ").str.split(" ")
                      .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique())
    t = t.with_columns(T1=toks("ad1"), T2=toks("ad2"))
    t = t.with_columns(X=pl.col("T2").list.set_difference("T1"), M=pl.col("T1").list.set_difference("T2"))
    return t.select(*KEY, "X", "M", ad_x_n=pl.col("X").list.len().cast(pl.Int16), ad_m_n=pl.col("M").list.len().cast(pl.Int16))


def suffix_pair(d, pool, lab, fold_of, p0):
    s1, cand = tables(pool)
    t = (d.select(KEY).join(cand, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
          .select(*KEY, pk=pl.col("suf1").fill_null("") + "|" + pl.col("suf2").fill_null(""))
          .join(lab, on=KEY, how="left").join(fold_of, on="s1_id", how="left"))
    fe = t.filter(pl.col("fold").is_not_null())
    tot = fe.group_by("pk").agg(pos=pl.col("is_match").sum(), neg=(~pl.col("is_match")).sum())
    byf = fe.group_by("pk", "fold").agg(pf=pl.col("is_match").sum(), nf=(~pl.col("is_match")).sum())
    t = (t.join(tot, on="pk", how="left").join(byf, on=["pk", "fold"], how="left")
          .with_columns(pos=pl.col("pos").fill_null(0) - pl.col("pf").fill_null(0), neg=pl.col("neg").fill_null(0) - pl.col("nf").fill_null(0)))
    t = E.log_odds(t, p0)
    return t.select(*KEY, suf_pair_lo=pl.col("lo"), suf_pair_n=(pl.col("pos") + pl.col("neg")).cast(pl.Int32))


def train_oof(d, cols, folds, fit_ids, tag):
    """Like exp_stack.train2, but also returns predictions where FIT rows carry their out-of-fold score (for stage 3)."""
    d = d.with_row_index("_i")
    fit = d.join(fit_ids, on="s1_id", how="semi").filter(pl.col("role") == "fit")
    fold_s1 = [set(fit_ids["s1_id"].to_numpy()[va]) for _, va in folds]
    fs = fit["s1_id"].to_numpy()
    x = fit.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
    y = fit["is_match"].to_numpy()
    eng, oof, models = engine(), np.zeros(len(y), np.float32), []
    for i, s in enumerate(fold_s1):
        va = np.isin(fs, list(s))
        m, pred = fit_one(eng, x[~va], y[~va], x[va], y[va], cols)
        oof[va] = pred(x[va])
        models.append(pred)
        m.save_model(str(OUT3 / f"{tag}_fold{i}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    pickle.dump({"iso": iso, "cols": cols, "engine": eng, "folds": len(fold_s1)}, open(OUT3 / f"{tag}_calib.pkl", "wb"))
    log(f"{tag}: OOF AUC {roc_auc_score(y, oof):.5f}")
    del x
    xa = d.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
    p = iso.predict(np.mean([m(xa) for m in models], axis=0)).astype(np.float32)
    p_oof = p.copy()
    p_oof[fit["_i"].to_numpy()] = iso.predict(oof).astype(np.float32)
    out = d.select(*KEY, "role", "is_match").with_columns(p=pl.Series(p))
    return out, pl.Series(p_oof)


def main():
    OUT3.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUT3
    allf = S.base_features()
    allf, folds, fit_ids = S.stage1(allf)
    d = S.stage2_features(allf)
    fit = allf.filter(pl.col("role") == "fit")
    fold_arr = np.empty(fit.height, np.int8)
    for i, (_, va) in enumerate(folds):
        fold_arr[va] = i
    fold_of = pl.DataFrame({"s1_id": fit["s1_id"], "fold": fold_arr}).unique("s1_id")
    lab = d.select(*KEY, "is_match")
    t = E.tok_num(d, "train")
    q, _, p0 = E.qualifier_features(t.select(*KEY, "X", "M"), lab, fold_of)
    d = d.join(t.drop("X", "M"), on=KEY, how="left").join(q, on=KEY, how="left")
    log("NEW2 features done")
    d = d.join(compact_features(d, "train"), on=KEY, how="left")
    log("compact/house features done")
    aw = addr_words(d, "train")
    qa, _, _ = E.qualifier_features(aw.select(*KEY, "X", "M"), lab, fold_of)
    qa = qa.rename({c: "ad_" + c for c in qa.columns if c not in KEY})
    d = d.join(aw.drop("X", "M"), on=KEY, how="left").join(qa, on=KEY, how="left")
    d = d.join(suffix_pair(d, "train", lab, fold_of, p0), on=KEY, how="left")
    log("address-word and suffix-pair features done")
    d.write_parquet(OUT3 / "feats_TRAIN.parquet")

    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    s2 = base_cols + S.NEW + E.NEW2
    preds = {"base (production model)": d.select(*KEY, "role", "is_match", p="p1")}
    p2 = None
    for name, extra in (("ref stage2+NUM+QUAL", []), ("+A compact/house", GA), ("+C addr words", GC), ("+D suffix pair", GD),
                        ("+all NEW3", GA + GC + GD)):
        log(f"training: {name}")
        preds[name], pp = train_oof(d, s2 + extra, folds, fit_ids, "v_" + "".join(ch if ch.isalnum() else "_" for ch in name))
        if name == "+all NEW3":
            p2 = pp
    log("stage 3: recompute cross-candidate features from OOF stage-2 scores")
    d3 = d.drop([c for c in S.NEW if c in d.columns] + ["p1"]).with_columns(p1=p2)
    d3 = S.stage2_features(d3)          # G/O/SUP/RAR from p2 (columns overwritten by name)
    d3 = d3.with_columns(p2=p2)
    preds["stage3 (+all NEW3)"], _ = train_oof(d3, s2 + GA + GC + GD + ["p2"], folds, fit_ids, "v_stage3")
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    d3 = d3.with_columns(o_sum=pl.col("p2").sum().over("cand_id", "source"), g_src_sum=pl.col("p2").sum().over("s1_id", "source"))
    d3 = d3.with_columns(o_share=pl.col("p2") / pl.col("o_sum").clip(lower_bound=1e-6), g_share=pl.col("p2") / pl.col("g_src_sum").clip(lower_bound=1e-6),
                         o_logit_gap=lg("p2") - lg("o_other_best"), g_logit_gap=lg("p2") - lg("g_other_best"))
    SHARE = ["o_share", "g_share", "o_logit_gap", "g_logit_gap"]
    preds["stage3+share"], _ = train_oof(d3, s2 + GA + GC + GD + ["p2"] + SHARE, folds, fit_ids, "v_stage3share")
    for k, v in preds.items():
        v.write_parquet(OUT3 / f"pred_TRAIN_{''.join(ch if ch.isalnum() else '_' for ch in k)}.parquet")
    S.evaluate(preds, ref="ref stage2+NUM+QUAL")


if __name__ == "__main__":
    main()
