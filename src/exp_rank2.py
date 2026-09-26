"""Ranking-feature experiments on the UNION model (E = 0.98643), same 20 pinned split-halves, same production rule.  Usage: VARIANTS=CTRL,ADDR2 python src/exp_rank2.py
Starting point = exp_union.py's model: allf_union.parquet -> assemble (production folds pinned) -> train_oof(ref + DENSE).  Assembled matrix cached in exp/rank2/du.parquet.
CTRL   = ref + DENSE retrained unchanged (measures retrain noise, so a variant's gain is judged against E AND against CTRL)
ADDR2  = CTRL + address-consistency features for candidates with a non-empty address (a2_*):
         numeric tokens of the WHOLE address (leading zeros stripped): shared / only-S1 / only-cand counts, Jaccard, set-equal;
         longest-number Levenshtein and prefix relation (digit truncation 1738 vs 173, '003441' vs '3441'); first-number Levenshtein;
         address token-set / token-sort / plain ratio and the same on the number-free text (street + city words, typo-tolerant, order-free).
         Label-free (only normalised address strings) -> no validation label enters a feature.
Prints, per model: production-rule F0.5 / P / R / India / US, blocking recall, cand/S1, held-out mean / worst / best over the 20 split-halves, gain vs A (0.98108) and vs E
(mean/min/max, #positive of 20), paired vs CTRL.
"""
import json
import os
import pickle

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
import exp_stack4 as X4
from common import CACHE, log
from exp_budget import val_truth
from exp_final import assemble
from exp_union import DENSE

N = CACHE / "norm"
EX = CACHE.parent / "exp"
OUTU, OUTR = EX / "union", EX / "rank2"
KEY = S.KEY
A2 = ["a2_both", "a2_shared", "a2_only1", "a2_only2", "a2_jacc", "a2_exact", "a2_lg_lev", "a2_lg_prefix", "a2_f_lev", "a2_tset", "a2_tsort", "a2_tset_nn", "a2_ratio"]


def addr2_features(d):
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "addr_norm"]).rename({"id": "s1_id", "addr_norm": "ad1"})
    cd = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "addr_norm"]).with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename(
        {"id": "cand_id", "addr_norm": "ad2"})
    t = d.select(KEY).join(cd, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left").with_columns(pl.col("ad1").fill_null(""), pl.col("ad2").fill_null(""))
    nums = lambda c: pl.col(c).str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.element().filter(pl.element() != "")).list.unique()
    longest = lambda c: pl.col(c).list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True).first()).list.first().fill_null("")
    first = lambda c: pl.col(c).str.extract(r"(\d+)").str.strip_chars_start("0").fill_null("")
    t = t.with_columns(n1=nums("ad1"), n2=nums("ad2"), f1=first("ad1"), f2=first("ad2"), nn1=pl.col("ad1").str.replace_all(r"\d+", " "), nn2=pl.col("ad2").str.replace_all(r"\d+", " "))
    t = t.with_columns(l1=longest("n1"), l2=longest("n2"))
    both = ((pl.col("ad1") != "") & (pl.col("ad2") != ""))
    bn = both & (pl.col("l1") != "") & (pl.col("l2") != "")
    inter, uni = pl.col("n1").list.set_intersection("n2").list.len(), pl.col("n1").list.set_union("n2").list.len()
    t = t.with_columns(a2_both=both.cast(pl.Float32), a2_shared=pl.when(both).then(inter).cast(pl.Float32),
                       a2_only1=pl.when(both).then(pl.col("n1").list.set_difference("n2").list.len()).cast(pl.Float32),
                       a2_only2=pl.when(both).then(pl.col("n2").list.set_difference("n1").list.len()).cast(pl.Float32),
                       a2_jacc=pl.when(both & (uni > 0)).then(inter / uni).cast(pl.Float32),
                       a2_exact=pl.when(bn).then((pl.col("n1").list.set_symmetric_difference("n2").list.len() == 0)).cast(pl.Float32),
                       a2_lg_prefix=pl.when(bn).then(pl.col("l1").str.starts_with(pl.col("l2")) | pl.col("l2").str.starts_with(pl.col("l1"))).cast(pl.Float32))
    a1, a2, nn1, nn2 = t["ad1"].to_list(), t["ad2"].to_list(), t["nn1"].to_list(), t["nn2"].to_list()
    l1, l2, f1, f2 = t["l1"].to_list(), t["l2"].to_list(), t["f1"].to_list(), t["f2"].to_list()
    bm, bnm = t.select(x=both)["x"].to_numpy(), t.select(x=bn)["x"].to_numpy()
    sim = lambda x, y, sc: cpdist(x, y, scorer=sc, workers=-1).astype(np.float32) / 100.0
    lev = lambda x, y: cpdist(x, y, scorer=Levenshtein.distance, workers=-1).astype(np.float32)
    out = {"a2_tset": sim(a1, a2, fuzz.token_set_ratio), "a2_tsort": sim(a1, a2, fuzz.token_sort_ratio), "a2_ratio": sim(a1, a2, fuzz.ratio),
           "a2_tset_nn": sim(nn1, nn2, fuzz.token_set_ratio), "a2_lg_lev": lev(l1, l2), "a2_f_lev": lev(f1, f2)}
    for k in ("a2_tset", "a2_tsort", "a2_ratio", "a2_tset_nn"):
        out[k][~bm] = np.nan
    out["a2_lg_lev"][~bnm] = np.nan
    out["a2_f_lev"][~(bnm & (np.array(f1, dtype=object) != "") & (np.array(f2, dtype=object) != ""))] = np.nan
    t = t.select(*KEY, *[c for c in A2 if c in t.columns]).with_columns([pl.Series(k, v) for k, v in out.items()]).select(*KEY, *A2)
    return t.with_columns(pl.col(c).cast(pl.Float32) for c in A2)


def load_du():
    if (OUTR / "du.parquet").exists():
        log("loading cached du")
        return pl.read_parquet(OUTR / "du.parquet")
    allf_u = pl.read_parquet(OUTU / "allf_union.parquet")
    fmap = dict(pl.read_parquet(OUTU / "fold_map.parquet").iter_rows())

    class ProductionFolds:
        def __init__(self, n):
            self.n = n

        def split(self, X_, y, groups):
            f = np.array([fmap[g] for g in groups])
            for i in range(self.n):
                yield np.flatnonzero(f != i), np.flatnonzero(f == i)

    S.GroupKFold = ProductionFolds
    du, folds, fit_ids = assemble(allf_u, "union")
    del allf_u
    du.write_parquet(OUTR / "du.parquet")
    return du


def report(files):
    _, truth_n, gt = val_truth()
    n_true, n_s1 = gt.height, truth_n.height
    ctry = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    india = truth_n.join(ctry, on="s1_id")["country"].to_numpy() == "India"
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    kc = (cfg["alpha"], cfg["miss"], cfg["gamma"], cfg["floor"])
    res, meta, held = {}, {}, {}
    for name, f in files.items():
        p = pl.read_parquet(f)
        res[name] = S.scores(p, truth_n)
        v = p.filter(pl.col("role") == "val")
        meta[name] = (v.filter(pl.col("is_match")).height / n_true, v.height / n_s1)
        vals = []
        for seed in range(10):
            m = np.random.default_rng(seed).random(len(truth_n)) < 0.5
            for a, b in ((m, ~m), (~m, m)):
                pick = max(res[name], key=lambda c: res[name][c][0][a].mean())
                vals.append(res[name][pick][0][b].mean())
        held[name] = np.array(vals)
    names = list(res)
    A, E_ = names[0], names[1]
    print(f"\n{'model':22s} {'F0.5':>8s} {'P':>7s} {'R':>7s} {'India':>8s} {'US':>8s} | {'blk.rec':>7s} {'cand/S1':>7s} | heldout mean  worst   best | vs 0.98108: mean min max | vs E: mean min max #pos/20")
    for n in names:
        f, pr, r, k = res[n][kc]
        ga, ge = held[n] - held[A], held[n] - held[E_]
        print(f"{n:22s} {f.mean():8.5f} {pr.mean():7.4f} {r.mean():7.4f} {f[india].mean():8.5f} {f[~india].mean():8.5f} | {meta[n][0]:7.4f} {meta[n][1]:7.2f} | "
              f"{held[n].mean():.5f} {held[n].min():.5f} {held[n].max():.5f} | {ga.mean():+.5f} {ga.min():+.5f} {ga.max():+.5f} | {ge.mean():+.5f} {ge.min():+.5f} {ge.max():+.5f} {int((ge > 0).sum())}/20"
              f"   (production-rule vs A {f.mean() - res[A][kc][0].mean():+.5f}, vs E {f.mean() - res[E_][kc][0].mean():+.5f})")
    if "CTRL" in held:
        print("\npaired vs CTRL (retrain noise removed):")
        for n in names:
            if n not in ("CTRL", A, E_):
                g = held[n] - held["CTRL"]
                print(f"  {n:20s} mean {g.mean():+.5f} min {g.min():+.5f} max {g.max():+.5f} positive {int((g > 0).sum())}/20 | production-rule {res[n][kc][0].mean() - res['CTRL'][kc][0].mean():+.5f}")
    for n in names:
        print(f"  20 held-out {n:20s}", " ".join(f"{x:.5f}" for x in np.sort(held[n])))


def main():
    OUTR.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUTR
    X.OUT3 = OUTR
    du = load_du()
    fmap = dict(pl.read_parquet(OUTU / "fold_map.parquet").iter_rows())
    fit = du.filter(pl.col("role") == "fit")
    fit_ids = fit.select("s1_id")                   # one row per fit row (train_oof only uses fold membership of these S1)
    fo = np.array([fmap[g] for g in fit_ids["s1_id"].to_numpy()])
    folds = [(np.flatnonzero(fo != i), np.flatnonzero(fo == i)) for i in range(3)]
    del fit
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    ref = base_cols + S.NEW + E.NEW2 + ["is_india"] + X.GA + X.GC + X.GD + X4.NEW4 + DENSE
    variants = os.environ.get("VARIANTS", "CTRL,ADDR2").split(",")
    files = {"A saved 0.98108": EX / "final" / "pred_TRAIN__NEW3_NEW4.parquet", "E union 0.98643": OUTU / "pred_TRAIN_union_rev_k1.parquet"}
    for v in variants:
        extra = []
        if v == "ADDR2":
            f = OUTR / "addr2.parquet"
            if f.exists():
                a2 = pl.read_parquet(f)
            else:
                log("computing a2_* address features")
                a2 = addr2_features(du)
                a2.write_parquet(f)
            assert a2.height == du.height and a2.select(KEY).equals(du.select(KEY)), "a2 rows not aligned with du"
            dv = du.join(a2, on=KEY, how="left")
            extra = A2
            nn = dv.filter(pl.col("a2_both") == 1).select([pl.col(c).is_null().mean() for c in A2[2:]]).to_dicts()[0]
            log(f"a2 rows with both addresses: {dv.filter(pl.col('a2_both') == 1).height:,} of {dv.height:,}; null share among those {nn}")
        else:
            dv = du
        log(f"=== training {v}: {len(ref) + len(extra)} features")
        pv, _ = X.train_oof(dv, ref + extra, folds, fit_ids, f"r_{v}")
        pv.write_parquet(OUTR / f"pred_{v}.parquet")
        files[v] = OUTR / f"pred_{v}.parquet"
        del dv
    report(files)
    print("DONE")


if __name__ == "__main__":
    main()
