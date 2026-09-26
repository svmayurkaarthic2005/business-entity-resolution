"""FINAL ablation table (VAL 33,172 labelled S1, same 20 held-out split-halves).  Usage: python src/exp_final.py  (exp/final/)

Configurations (all with the is_india flag; selection is on held-out macro F0.5, never AUC):
  current       base + stage-2 groups + number-set + name-qualifier + is_india          (reproduces the 0.97854 model)
  +NEW3         current + compact/leet, house-number, address-word, suffix-pair features
  stage 3       +NEW3, then stage-3 features recomputed from OUT-OF-FOLD stage-2 scores (+ owner-share / logit-gap)
  +NEW4 (amb)   current + name-ambiguity / IDF / candidate-context features
  stage3+amb    +NEW3 +NEW4 with stage 3
  +empty        current on the candidate set UNIONED with the empty-address name channel (top-3 per source, name cosine >= 0.6)
  all+empty     stage3+amb on the unioned candidate set
The empty-address rows are added to fit / val / competitor roles like every other candidate; their blocking columns are missing
(NaN) and flagged by ch_empty.  For each configuration also: pair recall of the candidate set and candidates per S1.
"""
import pickle

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

import exp_block2 as B
import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
import exp_stack4 as X4
import features as F
from common import CACHE, log
from exp_budget import val_truth

N = CACHE / "norm"
OUTF = CACHE.parent / "exp" / "final"
KEY = S.KEY


def empty_rows(allf_old):
    """Empty-address name channel for every queried TRAIN S1: top-3 per source, cosine >= 0.6 (char and word retrievals, unioned)."""
    q_ids = pl.read_parquet(CACHE / "queried_TRAIN.parquet")
    s1 = pl.read_parquet(N / "train_source1.parquet").join(q_ids.rename({"s1_id": "id"}), on="id")
    pools = {s: pl.read_parquet(N / f"train_source{s}.parquet") for s in (2, 3)}
    out = []
    for c in sorted(s1["country"].unique().to_list()):
        q = s1.filter(pl.col("country") == c)
        pc = {s: pools[s].filter((pl.col("country") == c) & pl.col("addr_missing")) for s in (2, 3)}
        corpus = pl.concat([pc[2]["name_core"], pc[3]["name_core"]]).fill_null("").to_list()
        n2 = len(pc[2])
        for kw in (dict(analyzer="char_wb", ngram_range=(3, 4), min_df=1), dict(analyzer="word", token_pattern=r"\S+", min_df=1)):
            v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
            x = v.fit_transform(corpus)
            qm = v.transform(q["name_core"].fill_null("").to_list())
            for s, m in ((2, x[:n2]), (3, x[n2:])):
                mT = m.T.tocsr()
                for c0 in range(0, len(q), 50_000):
                    r, cc = B.topn(qm[c0:c0 + 50_000], mT, 3, 0.6)
                    out.append(B.pairs_df(q["id"].to_numpy(), pc[s]["id"].to_numpy(), r + c0, cc, s))
        log(f"  empty channel {c}: {sum(len(o) for o in out):,} raw pairs so far")
    new = pl.concat(out).unique()
    old_keys = pl.read_parquet(CACHE / "candidates_TRAIN.parquet", columns=KEY)
    new = new.join(old_keys, on=KEY, how="anti")
    log(f"empty-address channel: {new.height:,} pairs not already in candidates_TRAIN")
    roles = allf_old.select("s1_id", "role").unique("s1_id").filter(pl.col("role").is_in(["fit", "val"]))
    new = new.join(roles, on="s1_id", how="left")
    val_rec = pl.concat([allf_old.filter(pl.col("role") == "val").select("cand_id", "source"),
                         new.filter(pl.col("role") == "val").select("cand_id", "source")]).unique()
    comp = new.filter(pl.col("role").is_null()).join(val_rec, on=["cand_id", "source"], how="semi").with_columns(role=pl.lit("comp"))
    new = pl.concat([new.filter(pl.col("role").is_not_null()), comp], how="vertical_relaxed")
    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").select("s1_id", cand_id="match_id", source="match_source", is_match=pl.lit(True))
    new = new.join(gt, on=KEY, how="left").with_columns(pl.col("is_match").fill_null(False))
    log(f"new rows by role {dict(new['role'].value_counts().iter_rows())}; positives {new['is_match'].mean():.3%}")
    return new


def featurise_new(new, allf_old):
    """Pair features + competition features for the new rows; blocking columns missing."""
    oldc = pl.read_parquet(CACHE / "candidates_TRAIN.parquet", columns=[*KEY, "score"])
    comp = F.competition(pl.concat([oldc, new.select(KEY).with_columns(score=pl.lit(0.0, pl.Float32))], how="vertical_relaxed"))
    comp = comp.join(new.select(KEY), on=KEY, how="semi").select(*KEY, "cand_best", "cand_n_owners", "owner_rank", "s1_best",
                                                                 "s1_n_strong", "gap_next", "gap_owner", "gap_s1")
    del oldc
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country", *F.TXT]).rename(
        lambda x: x if x in ("id", "country") else x + "_1").rename({"id": "s1_id"})
    cand = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "name_raw", *F.TXT])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id", "name_raw": "name_raw_c"})
    toks = s1.select(pl.col("name_core_1").str.split(" ").list.unique().alias("tok")).explode("tok")
    idf = toks.group_by("tok").len().with_columns(idf=(len(s1) / pl.col("len")).log().cast(pl.Float32)).select("tok", "idf")
    c = new.join(comp, on=KEY, how="left")
    out = []
    for i in range(0, len(c), F.CHUNK):
        d = c.slice(i, F.CHUNK).join(s1, on="s1_id", how="left").join(cand, on=["cand_id", "source"], how="left")
        d = F.pair_features(d, idf)
        out.append(d.drop([x for x in d.columns if x.endswith("_1") or x in F.TXT or x == "name_raw_c"]))
    f = pl.concat(out)
    schema = allf_old.schema
    for col, dt in schema.items():
        if col not in f.columns:
            f = f.with_columns(pl.lit(None).cast(dt).alias(col))
    f = f.with_columns(am=pl.lit(True), score=pl.lit(0.0).cast(schema["score"]))
    return f.select(allf_old.columns)


def assemble(allf, tag):
    """Stage 1 -> stage-2 groups -> NEW2 -> NEW3 -> NEW4 -> is_india.  Returns d, folds, fit_ids.  Extra columns (ch_empty) ride along."""
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
    d = d.join(X.compact_features(d, "train"), on=KEY, how="left")
    aw = X.addr_words(d, "train")
    qa, _, _ = E.qualifier_features(aw.select(*KEY, "X", "M"), lab, fold_of)
    qa = qa.rename({c: "ad_" + c for c in qa.columns if c not in KEY})
    d = d.join(aw.drop("X", "M"), on=KEY, how="left").join(qa, on=KEY, how="left")
    d = d.join(X.suffix_pair(d, "train", lab, fold_of, p0), on=KEY, how="left")
    d = d.join(X4.ambiguity(d, "train"), on=KEY, how="left")
    d = d.with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32))
    log(f"[{tag}] features assembled {d.shape}")
    return d, folds, fit_ids


def stage3(d, p2, cols, folds, fit_ids, tag):
    d3 = d.drop([c for c in S.NEW if c in d.columns] + ["p1"]).with_columns(p1=p2)
    d3 = S.stage2_features(d3).with_columns(p2=p2)
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    d3 = d3.with_columns(o_sum=pl.col("p2").sum().over("cand_id", "source"), g_src_sum=pl.col("p2").sum().over("s1_id", "source"))
    d3 = d3.with_columns(o_share=pl.col("p2") / pl.col("o_sum").clip(lower_bound=1e-6), g_share=pl.col("p2") / pl.col("g_src_sum").clip(lower_bound=1e-6),
                         o_logit_gap=lg("p2") - lg("o_other_best"), g_logit_gap=lg("p2") - lg("g_other_best"))
    return X.train_oof(d3, cols + ["p2", "o_share", "g_share", "o_logit_gap", "g_logit_gap"], folds, fit_ids, tag)[0]


def set_stats(d, name, val_ids, n_true, stats):
    v = d.filter(pl.col("role") == "val")
    stats[name] = (v.filter(pl.col("is_match")).height / n_true, v.height / val_ids)


def main():
    OUTF.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUTF
    X.OUT3 = OUTF
    _, truth_n, gt = val_truth()
    n_true, n_s1 = gt.height, truth_n.height
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    cur = base_cols + S.NEW + E.NEW2 + ["is_india"]
    n3 = X.GA + X.GC + X.GD
    stats, preds = {}, {}

    allf_old = S.base_features().with_columns(ch_empty=pl.lit(0, pl.Int8))
    log("=== old candidate set")
    d, folds, fit_ids = assemble(allf_old.drop("ch_empty"), "old")
    preds["base (production model)"] = d.select(*KEY, "role", "is_match", p="p1")
    stats["base (production model)"] = None
    set_stats(d, "current", n_s1, n_true, stats)
    log("training: current")
    preds["current"], _ = X.train_oof(d, cur, folds, fit_ids, "f_current")
    log("training: +NEW3")
    preds["+NEW3"], p2a = X.train_oof(d, cur + n3, folds, fit_ids, "f_new3")
    log("stage 3 (NEW3)")
    preds["stage 3 (+NEW3)"] = stage3(d, p2a, cur + n3, folds, fit_ids, "f_stage3")
    log("training: +NEW4 (ambiguity)")
    preds["+NEW4 (amb)"], _ = X.train_oof(d, cur + X4.NEW4, folds, fit_ids, "f_new4")
    log("training: +NEW3+NEW4")
    preds["+NEW3+NEW4"], p2b = X.train_oof(d, cur + n3 + X4.NEW4, folds, fit_ids, "f_new34")
    log("stage 3 (NEW3+NEW4)")
    preds["stage3+amb"] = stage3(d, p2b, cur + n3 + X4.NEW4, folds, fit_ids, "f_stage3amb")
    for k in ("+NEW3", "stage 3 (+NEW3)", "+NEW4 (amb)", "+NEW3+NEW4", "stage3+amb"):
        stats[k] = stats["current"]
    del d

    log("=== unioned candidate set (+ empty-address channel)")
    new = empty_rows(allf_old.drop("ch_empty"))
    newf = featurise_new(new, allf_old.drop("ch_empty")).with_columns(ch_empty=pl.lit(1, pl.Int8))
    allf_u = pl.concat([allf_old, newf], how="vertical_relaxed")      # ch_empty rides along as a column
    del allf_old, newf
    du, folds_u, fit_ids_u = assemble(allf_u, "union")
    du = du.with_columns(ch_empty=pl.col("ch_empty").cast(pl.Float32))
    set_stats(du, "+empty", n_s1, n_true, stats)
    log("training: +empty (current features)")
    preds["+empty"], _ = X.train_oof(du, cur + ["ch_empty"], folds_u, fit_ids_u, "f_empty")
    log("training: all + empty (NEW3+NEW4)")
    preds["all+empty (stage3+amb)"] = None
    _, p2u = X.train_oof(du, cur + n3 + X4.NEW4 + ["ch_empty"], folds_u, fit_ids_u, "f_allempty_s2")
    preds["all+empty (stage3+amb)"] = stage3(du, p2u, cur + n3 + X4.NEW4 + ["ch_empty"], folds_u, fit_ids_u, "f_allempty")
    stats["all+empty (stage3+amb)"] = stats["+empty"]

    for k, v in preds.items():
        v.write_parquet(OUTF / f"pred_TRAIN_{''.join(ch_ if ch_.isalnum() else '_' for ch_ in k)}.parquet")
    S.evaluate(preds, ref="current")
    print("\ncandidate set per configuration: pair recall / candidates per S1 (VAL)")
    for k, v in stats.items():
        if v:
            print(f"  {k:26s} {v[0]:.4f} / {v[1]:.2f}")


if __name__ == "__main__":
    main()
