"""Experiment: stage-2 model + address-number-set and learned name-qualifier features.

Usage: python src/exp_stack2.py     (writes only exp/stack2/)
New features (NEW2), computed for every candidate row:
  numbers: all digit groups of both addresses (leading zeros stripped) -> jaccard, share of S1 numbers found in the
           candidate and vice versa, conflict flag (both have numbers, none shared), counts.
           (house_state only compares the FIRST number, which breaks on reordered Indian addresses)
  qualifiers: tokens of the candidate name not in the S1 name (X) and vice versa (M); each token has a smoothed
           log-odds of "match" learned from training rows -- OUT-OF-FOLD for training rows (production folds), from all
           training rows for VAL/test. min/sum per row. Catches sibling distractors ("... Participations", "Holding").
Compared on the untouched VAL S1 against stage2 (same run): base / stage2 / stage2+new, each with is_india.
"""
import numpy as np
import polars as pl

import exp_stack as S
from common import CACHE, log

N = CACHE / "norm"
OUT2 = CACHE.parent / "exp" / "stack2"
KEY = S.KEY
NEW2 = ["num_jacc", "num_s1_in_c", "num_c_in_s1", "num_conflict", "n_num_s1", "n_num_c",
        "x_n", "x_lo_min", "x_lo_sum", "m_n", "m_lo_min", "m_lo_sum"]
SMOOTH = 20.0


def tok_num(d, pool):
    cand = pl.concat([pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id", "name_latin", "addr_norm"])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "name_latin", "addr_norm"]).rename(
        {"id": "s1_id", "name_latin": "n1", "addr_norm": "a1"})
    nums = lambda c: pl.col(c).fill_null("").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.unique()
    toks = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    t = (d.select(KEY).join(cand, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
          .with_columns(A=nums("a1"), B=nums("addr_norm"), T1=toks("n1"), T2=toks("name_latin")))
    t = t.with_columns(ni=pl.col("A").list.set_intersection("B").list.len(), nu=pl.col("A").list.set_union("B").list.len(),
                       na=pl.col("A").list.len(), nb=pl.col("B").list.len(),
                       X=pl.col("T2").list.set_difference("T1"), M=pl.col("T1").list.set_difference("T2"))
    return t.select(*KEY, "X", "M",
                    num_jacc=pl.when(pl.col("nu") > 0).then(pl.col("ni") / pl.col("nu")).otherwise(-1.0),
                    num_s1_in_c=pl.when(pl.col("na") > 0).then(pl.col("ni") / pl.col("na")).otherwise(-1.0),
                    num_c_in_s1=pl.when(pl.col("nb") > 0).then(pl.col("ni") / pl.col("nb")).otherwise(-1.0),
                    num_conflict=((pl.col("na") > 0) & (pl.col("nb") > 0) & (pl.col("ni") == 0)).cast(pl.Int8),
                    n_num_s1=pl.col("na").cast(pl.Int16), n_num_c=pl.col("nb").cast(pl.Int16),
                    x_n=pl.col("X").list.len().cast(pl.Int16), m_n=pl.col("M").list.len().cast(pl.Int16))


def log_odds(tab, p0):
    return tab.with_columns(lo=((pl.col("pos") + SMOOTH * p0) / (pl.col("neg") + SMOOTH * (1 - p0))).log() - np.log(p0 / (1 - p0)))


def qualifier_features(t, lab, fold_of):
    """OOF token log-odds for fit rows, full-fit log-odds for the rest. Returns per-row min/sum and the vocab tables."""
    t = t.join(lab, on=KEY, how="left").join(fold_of, on="s1_id", how="left")
    p0 = t.filter(pl.col("fold").is_not_null())["is_match"].mean()
    out, vocab = t.select(KEY), {}
    for col, pre in (("X", "x"), ("M", "m")):
        e = t.select(*KEY, "is_match", "fold", tok=col).explode("tok").drop_nulls("tok")
        fe = e.filter(pl.col("fold").is_not_null())
        tot = fe.group_by("tok").agg(pos=pl.col("is_match").sum(), neg=(~pl.col("is_match")).sum())
        byf = fe.group_by("tok", "fold").agg(pf=pl.col("is_match").sum(), nf=(~pl.col("is_match")).sum())
        vocab[col] = log_odds(tot, p0).select("tok", "lo")
        e = (e.join(tot, on="tok", how="left").join(byf, on=["tok", "fold"], how="left")
               .with_columns(pos=pl.col("pos").fill_null(0) - pl.col("pf").fill_null(0),
                             neg=pl.col("neg").fill_null(0) - pl.col("nf").fill_null(0)))
        e = log_odds(e, p0)
        agg = e.group_by(KEY).agg(**{f"{pre}_lo_min": pl.col("lo").min(), f"{pre}_lo_sum": pl.col("lo").sum()})
        out = out.join(agg, on=KEY, how="left")
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in ("x_lo_min", "x_lo_sum", "m_lo_min", "m_lo_sum")])
    return out, vocab, p0


def main():
    OUT2.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUT2                                  # train2 saves its models here
    allf = S.base_features()
    allf, folds, fit_ids = S.stage1(allf)
    d = S.stage2_features(allf)
    fit = allf.filter(pl.col("role") == "fit")
    fold_arr = np.empty(fit.height, np.int8)
    for i, (_, va) in enumerate(folds):
        fold_arr[va] = i
    fold_of = pl.DataFrame({"s1_id": fit["s1_id"], "fold": fold_arr}).unique("s1_id")
    t = tok_num(d, "train")
    q, vocab, p0 = qualifier_features(t.select(*KEY, "X", "M"), d.select(*KEY, "is_match"), fold_of)
    for k, v in vocab.items():
        v.write_parquet(OUT2 / f"vocab_{k}.parquet")
    (pl.DataFrame({"p0": [p0]})).write_parquet(OUT2 / "vocab_p0.parquet")
    d = d.join(t.drop("X", "M"), on=KEY, how="left").join(q, on=KEY, how="left")
    d = d.with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32))
    top = vocab["X"].join(vocab["X"].select("tok"), on="tok").sort("lo").head(15)
    log(f"most 'non-match' extra candidate-name tokens: {top.rows()}")
    import pickle
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    preds = {"base (production model)": d.select(*KEY, "role", "is_match", p="p1"),
             "stage2+is_india": S.train2(d, base_cols + S.NEW + ["is_india"], folds, fit_ids, "s2_india"),
             "stage2+new": S.train2(d, base_cols + S.NEW + NEW2, folds, fit_ids, "s2new"),
             "stage2+new+is_india": S.train2(d, base_cols + S.NEW + NEW2 + ["is_india"], folds, fit_ids, "s2new_india")}
    for k, v in preds.items():
        v.write_parquet(OUT2 / f"pred_TRAIN_{k.split()[0].replace('+', '_')}.parquet")
    S.evaluate(preds, ref="stage2+is_india")


if __name__ == "__main__":
    main()
