"""Experiment: stage-2 (stacked) matcher with cross-candidate, name-rarity and contested-owner features.

Usage: python src/exp_stack.py          (writes only exp/stack/; production models/files untouched)

Stage 1 = the production model. Its scores p1 are OUT-OF-FOLD for training rows (production GroupKFold folds are
reconstructed and each fold model scores only its held-out S1) and fold-averaged for every other row (VAL, competitors),
exactly like at test time. Stage-2 features from p1:
  group (same S1):   g_sum, g_n5, g_rank, g_other_best, g_gap         (how many strong candidates the S1 has)
  record (owners):   o_other_best, o_margin, o_n5, o_rank             (is another S1 a better owner of this record?)
  support:           sup_name, sup_both, sup_x, n_sup                 (is this candidate similar to other confident
                                                                        candidates of the same S1, esp. from the other source?)
  rarity:            cand_name_freq, s1_name_freq, grp_dup             (how common the core name is)
Stage 2 = XGBoost on base + stage-2 features, same GroupKFold folds, isotonic on OOF. Evaluation on the untouched VAL S1:
decision rule re-chosen on one random half and scored on the other (20 runs), for base / stage2 / stage2+is_india.
"""
import itertools
import json
import pickle
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import features as F
from common import CACHE, log
from decide import choose, owner_adjust
from exp_budget import val_truth
from exp_decide import per_s1
from predict import load_models
from train import engine, fit_one

N = CACHE / "norm"
OUTD = CACHE.parent / "exp" / "stack"
KEY = ["s1_id", "cand_id", "source"]
NEW = ["g_sum", "g_n5", "g_rank", "g_other_best", "g_gap", "o_other_best", "o_margin", "o_n5", "o_rank",
       "sup_name", "sup_both", "sup_x", "n_sup", "cand_name_freq", "s1_name_freq", "grp_dup"]
GRID = dict(alpha=(0.3, 0.5, 0.7), miss=(0.4, 0.6, 0.8), gamma=(1.5, 2.0, 3.0, 4.0), floor=(0.1, 0.15))


def base_features():
    """Base features for EVERY candidates_TRAIN row (features_TRAIN only has fit/val/comp rows)."""
    ft = pl.read_parquet(CACHE / "features_TRAIN.parquet")
    c = F.competition(pl.read_parquet(CACHE / "candidates_TRAIN.parquet"))
    miss = c.join(ft.select(KEY), on=KEY, how="anti")
    log(f"features_TRAIN rows {ft.height:,}; extra candidate rows to featurise {miss.height:,}")
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country", *F.TXT]).rename(
        lambda x: x if x in ("id", "country") else x + "_1").rename({"id": "s1_id"})
    cand = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "name_raw", *F.TXT])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id", "name_raw": "name_raw_c"})
    toks = s1.select(pl.col("name_core_1").str.split(" ").list.unique().alias("tok")).explode("tok")
    idf = toks.group_by("tok").len().with_columns(idf=(len(s1) / pl.col("len")).log().cast(pl.Float32)).select("tok", "idf")
    out = []
    for i in range(0, len(miss), F.CHUNK):
        d = miss.slice(i, F.CHUNK).join(s1, on="s1_id", how="left").join(cand, on=["cand_id", "source"], how="left")
        d = F.pair_features(d, idf)
        out.append(d.drop([x for x in d.columns if x.endswith("_1") or x in F.TXT or x == "name_raw_c"]))
    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").select("s1_id", cand_id="match_id", source="match_source", is_match=pl.lit(True))
    extra = pl.concat(out).with_columns(role=pl.lit("other")).join(gt, on=KEY, how="left").with_columns(pl.col("is_match").fill_null(False))
    return pl.concat([ft, extra.select(ft.columns)], how="vertical_relaxed")


def stage1(allf):
    """p1: OOF for fit rows (reconstructed production folds), fold mean elsewhere."""
    cal = pickle.load(open(CACHE / "calib.pkl", "rb"))
    models = load_models(cal)
    fit = allf.filter(pl.col("role") == "fit")
    folds = list(GroupKFold(cal["folds"]).split(np.zeros(fit.height), fit["is_match"].to_numpy(), fit["s1_id"].to_numpy()))
    fold_of = np.empty(fit.height, np.int8)
    for i, (_, va) in enumerate(folds):
        fold_of[va] = i
    fmap = pl.DataFrame({"s1_id": fit["s1_id"], "fold": fold_of}).unique("s1_id")
    x = allf.select([pl.col(k).cast(pl.Float32) for k in cal["cols"]]).to_numpy()
    raw = np.stack([m(x) for m in models])            # (folds, rows)
    del x
    fold = allf.select("s1_id").join(fmap, on="s1_id", how="left")["fold"].to_numpy()
    pick = np.where(np.isnan(fold.astype(float)), -1, fold).astype(int)
    r = np.where(pick >= 0, raw[np.clip(pick, 0, None), np.arange(raw.shape[1])], raw.mean(0))
    p1 = cal["iso"].predict(r).astype(np.float32)
    isfit = pick >= 0
    log(f"stage-1 OOF AUC on fit rows (should be ~0.99830): {roc_auc_score(allf['is_match'].to_numpy()[isfit], r[isfit]):.5f}")
    return allf.with_columns(p1=pl.Series(p1)), folds, fit.select("s1_id")


def stage2_features(allf, pool="train"):
    t0 = time.time()
    rk = lambda part: pl.col("p1").rank("ordinal", descending=True).over(part)
    top2 = lambda part: pl.col("p1").top_k(2).min().over(part)
    d = allf.with_columns(g_rank=rk("s1_id"), o_rank=rk(["cand_id", "source"]),
                          g_top1=pl.col("p1").max().over("s1_id"), g_top2=top2("s1_id"),
                          o_top1=pl.col("p1").max().over("cand_id", "source"), o_top2=top2(["cand_id", "source"]),
                          g_n=pl.len().over("s1_id"), o_n=pl.len().over("cand_id", "source"))
    d = d.with_columns(
        g_sum=pl.col("p1").sum().over("s1_id"), g_n5=(pl.col("p1") > 0.5).sum().over("s1_id").cast(pl.Int16),
        g_other_best=pl.when(pl.col("g_n") == 1).then(0.0).when(pl.col("g_rank") == 1).then(pl.col("g_top2")).otherwise(pl.col("g_top1")),
        o_other_best=pl.when(pl.col("o_n") == 1).then(0.0).when(pl.col("o_rank") == 1).then(pl.col("o_top2")).otherwise(pl.col("o_top1")),
        o_n5=(pl.col("p1") > 0.5).sum().over("cand_id", "source").cast(pl.Int16))
    d = d.with_columns(g_gap=pl.col("p1") - pl.col("g_other_best"), o_margin=pl.col("p1") - pl.col("o_other_best")).drop(
        "g_top1", "g_top2", "o_top1", "o_top2", "g_n", "o_n")

    # names/addresses of candidate records and S1
    cand = pl.concat([pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id", "name_core", "addr_norm"])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "name_core"]).rename({"id": "s1_id", "name_core": "s1_core"})
    freq_c = cand.group_by("name_core").len("cand_name_freq")
    freq_s = s1.group_by("s1_core").len("s1_name_freq")
    nm = (d.select(KEY).join(cand, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
            .join(freq_c, on="name_core", how="left").join(freq_s, on="s1_core", how="left")
            .with_columns(grp_dup=pl.len().over("s1_id", "name_core").cast(pl.Int16)))

    # support: similarity to other confident candidates of the same S1
    sup = nm.join(d.select(*KEY, "p1"), on=KEY).filter(pl.col("p1") >= 0.05).select(
        "s1_id", c2="cand_id", s2="source", p2="p1", n2="name_core", a2="addr_norm")
    # only candidates with p1 >= 0.01 get support features (others 0) -- same rule at test time
    rows = nm.join(d.select(*KEY, "p1"), on=KEY).filter(pl.col("p1") >= 0.01)
    pr = rows.select(*KEY, "name_core", "addr_norm").join(sup, on="s1_id").filter(
        ~((pl.col("cand_id") == pl.col("c2")) & (pl.col("source") == pl.col("s2"))))
    log(f"support pairs: {pr.height:,}")
    ns = cpdist(pr["name_core"].fill_null("").to_list(), pr["n2"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    ad = cpdist(pr["addr_norm"].fill_null("").to_list(), pr["a2"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    pr = pr.select(*KEY, "p2", "s2").with_columns(ns=pl.Series(ns.astype(np.float32)), asim=pl.Series(ad.astype(np.float32)))
    agg = pr.group_by(KEY).agg(
        sup_name=(pl.col("p2") * pl.col("ns")).max(),
        sup_both=(pl.col("p2") * pl.min_horizontal("ns", "asim")).max(),
        sup_x=(pl.col("p2") * pl.col("ns")).filter(pl.col("s2") != pl.col("source")).max(),
        n_sup=((pl.col("p2") > 0.5) & (pl.col("ns") >= 0.9)).sum().cast(pl.Int16))
    d = (d.join(nm.select(*KEY, "cand_name_freq", "s1_name_freq", "grp_dup"), on=KEY, how="left")
          .join(agg, on=KEY, how="left")
          .with_columns([pl.col(c).fill_null(0) for c in ("sup_name", "sup_both", "sup_x", "n_sup")]))
    log(f"stage-2 features in {time.time() - t0:.0f}s")
    return d


def train2(d, cols, folds, fit_ids, tag):
    fit = d.join(fit_ids, on="s1_id", how="semi").filter(pl.col("role") == "fit")
    # keep the production fold order: rebuild folds by S1 membership
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
        m.save_model(str(OUTD / f"{tag}_fold{i}.json"))
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    pickle.dump({"iso": iso, "cols": cols, "engine": eng, "folds": len(fold_s1)}, open(OUTD / f"{tag}_calib.pkl", "wb"))
    log(f"{tag}: OOF AUC {roc_auc_score(y, oof):.5f}")
    del x
    xa = d.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
    p = iso.predict(np.mean([m(xa) for m in models], axis=0)).astype(np.float32)
    return d.select(*KEY, "role", "is_match").with_columns(p=pl.Series(p))


def scores(pred, truth_n):
    pa = {a: owner_adjust(pred, a).filter(pl.col("role") == "val") for a in GRID["alpha"]}
    return {cfg: per_s1(choose(pa[cfg[0]], *cfg[1:]).filter("take"), truth_n) for cfg in itertools.product(*GRID.values())}


def evaluate(preds, ref=None):
    """Print VAL metrics per model and held-out split-half gains vs base (and vs ref)."""
    _, truth_n, gt = val_truth()
    ctry = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    india = truth_n.join(ctry, on="s1_id")["country"].to_numpy() == "India"
    cur = json.load(open(CACHE / "decide.json"))["cfg"]
    kc = (cur["alpha"], cur["miss"], cur["gamma"], cur["floor"])
    n_true = gt.height
    res = {}
    print(f"\n{'model':24s} rule        F0.5     P      R    pred/S1 modelRecall | India    US")
    for name, pred in preds.items():
        s = scores(pred, truth_n)
        res[name] = s
        retrieved = pred.filter((pl.col("role") == "val") & pl.col("is_match")).height
        for label, cfg in (("production", kc), ("best-on-VAL", max(s, key=lambda c: s[c][0].mean()))):
            f, p, r, k = s[cfg]
            kept = choose(owner_adjust(pred, cfg[0]).filter(pl.col("role") == "val"), *cfg[1:]).filter("take")
            mr = kept.filter(pl.col("is_match")).height / retrieved
            print(f"{name:24s} {label:11s} {f.mean():.5f} {p.mean():.4f} {r.mean():.4f} {k.mean():.3f}  {mr:.4f}     | "
                  f"{f[india].mean():.5f} {f[~india].mean():.5f}   {cfg}")
    print(f"blocking recall (VAL pairs retrieved): {retrieved / n_true:.4f}")
    for base in ["base (production model)"] + ([ref] if ref else []):
      for name in res:
        if name in (base, "base (production model)"):
            continue
        g = []
        for seed in range(10):
            m = np.random.default_rng(seed).random(len(truth_n)) < 0.5
            for A, B in ((m, ~m), (~m, m)):
                pick = {n: max(res[n], key=lambda c: res[n][c][0][A].mean()) for n in (name, base)}
                g.append(res[name][pick[name]][0][B].mean() - res[base][pick[base]][0][B].mean())
        g = np.array(g)
        dd = res[name][kc][0] - res[base][kc][0]
        print(f"HELD-OUT gain {name} vs {base}: mean {g.mean():+.5f} min {g.min():+.5f} max {g.max():+.5f} (20 split-halves); "
              f"paired at production rule {dd.mean():+.5f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.5f}")



def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    allf = base_features()
    allf, folds, fit_ids = stage1(allf)
    d = stage2_features(allf)
    d.select(*KEY, "role", *NEW, "p1").write_parquet(OUTD / "stage2_features_TRAIN.parquet")
    cal = pickle.load(open(CACHE / "calib.pkl", "rb"))
    base_cols = cal["cols"]
    d = d.with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32))
    preds = {"base (production model)": d.select(*KEY, "role", "is_match", p="p1"),
             "stage2": train2(d, base_cols + NEW, folds, fit_ids, "stage2"),
             "stage2+is_india": train2(d, base_cols + NEW + ["is_india"], folds, fit_ids, "stage2_india")}
    for k, v in preds.items():
        v.write_parquet(OUTD / f"pred_TRAIN_{k.split()[0].replace('+', '_')}.parquet")

    evaluate(preds)

if __name__ == "__main__":
    main()
