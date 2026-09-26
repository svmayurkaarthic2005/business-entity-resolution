"""Name-robustness feature groups on the UNION model, same pinned 20 splits / production rule.  Usage: VARIANTS=NAMEA,NAMEB python src/exp_name.py   (reuses exp/rank2/du.parquet from exp_rank2.py)
NAMEA  typo / edit / token-fuzzy similarity, label-free, computed on name_core and the leet-normalised compact name (exp_stack3.compact_expr):
       Jaro-Winkler, OSA (Damerau) similarity, LCS similarity, partial-token-set / partial-token-sort / WRatio, first-token and last-token Jaro-Winkler,
       common-prefix length share, first-4-characters equal.
NAMEB  website / handle / domain-style candidate names: domain root of the candidate raw name (www., .com/.net/.org/.in/.co/.io/.biz/.info, '@', '#') compared with the S1
       compact name (ratio, partial ratio, prefix relation, root inside S1 name); flag = candidate name is a domain/handle.  NaN elsewhere except the flag.
Each group is a separate variant against CTRL (E retrained, exp/rank2/pred_CTRL.parquet) so gains are attributable.  Output: the same table as exp_rank2.report.
"""
import os
import pickle

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq, OSA
from rapidfuzz.process import cpdist

import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
import exp_stack4 as X4
from common import CACHE, log
from exp_rank2 import DENSE, EX, KEY, N, OUTR, OUTU, load_du, report
from exp_stack3 import compact_expr

NA = ["nm_jw", "nm_osa", "nm_lcs", "nm_ptset", "nm_ptsort", "nm_wr", "nm_jw_c", "nm_osa_c", "nm_tok1_jw", "nm_tokl_jw", "nm_pref", "nm_first4"]
NB = ["nm_dom", "nm_dom_ratio", "nm_dom_partial", "nm_dom_prefix", "nm_dom_in"]
WEB = r"(?:^|[^a-z0-9])(?:www\.|https?:|@|#)|\.(?:com|net|org|in|co|io|biz|info)(?:$|[^a-z])"


def names(d):
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "name_core"]).rename({"id": "s1_id", "name_core": "n1"})
    cd = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "name_core", "name_raw"]).with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename(
        {"id": "cand_id", "name_core": "n2", "name_raw": "raw2"})
    t = d.select(KEY).join(cd, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
    return t.with_columns(pl.col("n1").fill_null(""), pl.col("n2").fill_null(""), pl.col("raw2").fill_null(""))


def name_a(d):
    t = names(d).with_columns(c1=compact_expr("n1"), c2=compact_expr("n2"))
    tok = lambda c, i: pl.col(c).str.split(" ").list.get(i, null_on_oob=True).fill_null("")
    t = t.with_columns(f1=tok("n1", 0), f2=tok("n2", 0), l1=tok("n1", -1), l2=tok("n2", -1))
    a, b, ca, cb = t["n1"].to_list(), t["n2"].to_list(), t["c1"].to_list(), t["c2"].to_list()
    sim = lambda x, y, sc: cpdist(x, y, scorer=sc, workers=-1).astype(np.float32)
    out = {"nm_jw": sim(a, b, JaroWinkler.similarity), "nm_osa": sim(a, b, OSA.normalized_similarity), "nm_lcs": sim(a, b, LCSseq.normalized_similarity),
           "nm_ptset": sim(a, b, fuzz.partial_token_set_ratio) / 100, "nm_ptsort": sim(a, b, fuzz.partial_token_sort_ratio) / 100, "nm_wr": sim(a, b, fuzz.WRatio) / 100,
           "nm_jw_c": sim(ca, cb, JaroWinkler.similarity), "nm_osa_c": sim(ca, cb, OSA.normalized_similarity),
           "nm_tok1_jw": sim(t["f1"].to_list(), t["f2"].to_list(), JaroWinkler.similarity), "nm_tokl_jw": sim(t["l1"].to_list(), t["l2"].to_list(), JaroWinkler.similarity)}
    n1, n2 = t["c1"].str.len_chars().to_numpy(), t["c2"].str.len_chars().to_numpy()
    pre = np.array([len(os.path.commonprefix([x, y])) for x, y in zip(ca, cb)], np.float32)
    out["nm_pref"] = pre / np.maximum(np.minimum(n1, n2), 1)
    out["nm_first4"] = ((t["c1"].str.slice(0, 4) == t["c2"].str.slice(0, 4)).to_numpy() & (n1 >= 4) & (n2 >= 4)).astype(np.float32)
    empty = (n1 == 0) | (n2 == 0)
    for k in out:
        out[k][empty] = np.nan
    return t.select(*KEY).with_columns([pl.Series(k, v.astype(np.float32)) for k, v in out.items()]).select(*KEY, *NA)


def name_b(d):
    t = names(d)
    low = pl.col("raw2").str.to_lowercase()
    dom = low.str.replace(r"^(?:https?:)?/*", "").str.replace(r"^(?:www\.|@|#)+", "").str.replace(r"\.(?:com|net|org|in|co|io|biz|info)\b.*$", "").str.replace_all(r"[^a-z0-9]", "")
    t = t.with_columns(isdom=low.str.contains(WEB), dom=dom, c1=compact_expr("n1"))
    t = t.with_columns(dom=pl.when(pl.col("isdom")).then(pl.col("dom")).otherwise(pl.lit("")))
    sub = t.filter(pl.col("isdom") & (pl.col("dom") != "") & (pl.col("c1") != "")).select(*KEY, "dom", "c1", "n1")
    log(f"NAMEB: {sub.height:,} domain/handle-style rows of {t.height:,}")
    x, y = sub["dom"].to_list(), sub["c1"].to_list()
    f1 = sub["n1"].str.split(" ").list.first().fill_null("")
    r = pl.DataFrame({**{k: sub[k] for k in KEY},
                      "nm_dom_ratio": pl.Series(cpdist(x, y, scorer=fuzz.ratio, workers=-1).astype(np.float32) / 100),
                      "nm_dom_partial": pl.Series(cpdist(x, y, scorer=fuzz.partial_ratio, workers=-1).astype(np.float32) / 100)})
    r = r.with_columns(nm_dom_prefix=(sub["dom"].str.starts_with(f1.str.replace_all(r"[^a-z0-9]", "")) & (f1.str.len_chars() >= 3)).cast(pl.Float32),
                       nm_dom_in=pl.Series(np.array([float(a in b) for a, b in zip(x, y)], np.float32)))
    out = t.select(*KEY, nm_dom=pl.col("isdom").cast(pl.Float32)).join(r, on=KEY, how="left")
    return out.select(*KEY, *NB)


def main():
    OUTR.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUTR
    X.OUT3 = OUTR
    du = load_du()
    fmap = dict(pl.read_parquet(OUTU / "fold_map.parquet").iter_rows())
    fit_ids = du.filter(pl.col("role") == "fit").select("s1_id")
    fo = np.array([fmap[g] for g in fit_ids["s1_id"].to_numpy()])
    folds = [(np.flatnonzero(fo != i), np.flatnonzero(fo == i)) for i in range(3)]
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    ref = base_cols + S.NEW + E.NEW2 + ["is_india"] + X.GA + X.GC + X.GD + X4.NEW4 + DENSE
    files = {"A saved 0.98108": EX / "final" / "pred_TRAIN__NEW3_NEW4.parquet", "E union 0.98643": OUTU / "pred_TRAIN_union_rev_k1.parquet", "CTRL": OUTR / "pred_CTRL.parquet"}
    groups = {"NAMEA": (name_a, NA), "NAMEB": (name_b, NB)}
    for v in os.environ.get("VARIANTS", "NAMEA,NAMEB").split(","):
        fn, cols = groups[v]
        f = OUTR / f"{v.lower()}.parquet"
        if f.exists():
            g = pl.read_parquet(f)
        else:
            log(f"computing {v} features")
            g = fn(du)
            g.write_parquet(f)
        assert g.height == du.height and g.select(KEY).equals(du.select(KEY)), f"{v} rows not aligned with du"
        dv = du.join(g, on=KEY, how="left")
        log(f"{v}: non-null share " + str({c: round(1 - dv[c].null_count() / dv.height, 3) for c in cols}))
        log(f"=== training {v}")
        pv, _ = X.train_oof(dv, ref + cols, folds, fit_ids, f"r_{v}")
        pv.write_parquet(OUTR / f"pred_{v}.parquet")
        files[v] = OUTR / f"pred_{v}.parquet"
        del dv
    report(files)
    print("DONE")


if __name__ == "__main__":
    main()
