"""Compact dense-blocking report (VAL 33,172 labelled S1).  Usage: python src/exp_dense_report.py   (needs exp/dense/{fwd,rev}_train.parquet)
Rows: 1 current TF-IDF | 2 dense only k4/8/16/32 | 3 current + dense k4/8/16/32 | 4 dense forward / reverse separately | 5 reverse TF-IDF.
dense k = forward top-k per source  UNION  reverse top-min(k,16) S1 per S2/S3 record.
Columns: recall | recall on the pairs the CURRENT blocker misses | new true pairs vs current | cands/S1 | extra cands/S1 | S1 with >=1 true pair found |
S1 fully covered | perfect-matcher F0.5 ceiling | ceiling gain vs current | new true pairs per 1,000 extra candidates | marginal (vs previous k) same ratio.
Also: overlap of dense with current candidates and how many dense-recovered true pairs are new.
"""
import polars as pl

from common import CACHE
from exp_block2 import ceiling
from exp_budget import val_truth
from exp_dense import KEY, KR, OUTD as D

KR = min(KR, 16)


def main():
    pl.Config.set_tbl_rows(60)
    val_ids, truth_n, gt = val_truth()
    n_s1, n_true = truth_n.height, gt.height
    n_pos = truth_n.filter(pl.col("n") > 0).height
    cv = pl.read_parquet(CACHE / "candidates_VAL.parquet").join(val_ids, on="s1_id", how="semi")
    ex = cv.filter((pl.col("rank") < 10) & (pl.col("score") >= 0.003)).select(KEY).unique()
    missed = gt.join(ex, on=KEY, how="anti")
    fwd = pl.read_parquet(D / "fwd_train.parquet").join(val_ids, on="s1_id", how="semi")
    rev = pl.read_parquet(D / "rev_train.parquet").join(val_ids, on="s1_id", how="semi")
    tr = CACHE.parent / "exp" / "reverse" / "rev_name.parquet"
    trev = pl.read_parquet(tr).join(val_ids, on="s1_id", how="semi") if tr.exists() else None
    c0 = ceiling(ex, truth_n, gt)
    print(f"VAL S1 {n_s1:,} | true pairs {n_true:,} | missed by current blocker {missed.height:,} ({missed.height / n_true:.2%}) | current: {ex.height / n_s1:.2f} cands/S1, ceiling {c0:.5f}\n")
    hdr = f"{'configuration':34s} {'recall':>7s} {'missedRec':>10s} {'newTrue':>8s} {'cand/S1':>8s} {'+cand/S1':>8s} {'S1>=1':>7s} {'S1full':>7s} {'ceiling':>8s} {'dCeil':>8s} {'new/1kExtra':>11s} {'marginal':>9s}"
    print(hdr)
    prev = {}

    def row(name, f, chain=None):
        f = f.select(KEY).unique()
        hit = gt.join(f, on=KEY, how="semi")
        new_true = hit.join(ex, on=KEY, how="anti").height
        extra = f.join(ex, on=KEY, how="anti").height
        rec_m = missed.join(f, on=KEY, how="semi").height
        s1_any = hit["s1_id"].n_unique() / n_pos
        full = truth_n.filter(pl.col("n") > 0).join(hit.group_by("s1_id").len("h"), on="s1_id", how="left").fill_null(0).select((pl.col("h") == pl.col("n")).mean()).item()
        c = ceiling(f, truth_n, gt)
        per_k = 1000 * new_true / extra if extra else float("nan")
        marg = ""
        if chain is not None:
            if chain in prev:
                dn, de = new_true - prev[chain][0], extra - prev[chain][1]
                marg = f"{1000 * dn / de:9.2f}" if de > 0 else "      n/a"
            prev[chain] = (new_true, extra)
        print(f"{name:34s} {hit.height / n_true:7.4f} {rec_m / max(missed.height, 1):9.1%} {new_true:8,} {f.height / n_s1:8.2f} {extra / n_s1:8.2f} {s1_any:7.4f} {full:7.4f} {c:8.5f} {c - c0:+8.5f} {per_k:11.2f} {marg:>9s}")
        return f

    row("1 current TF-IDF (K10, floor .003)", ex)
    print("-- 2 dense only")
    dk = lambda k: pl.concat([fwd.filter(pl.col("drank") < k).select(KEY), rev.filter(pl.col("rrank") < min(k, KR)).select(KEY)])
    for k in (4, 8, 16, 32):
        row(f"2 dense only k{k}", dk(k), "d")
    print("-- 3 current + dense")
    for k in (4, 8, 16, 32):
        row(f"3 current + dense k{k}", pl.concat([ex, dk(k)]), "u")
    print("-- 4 dense directions separately (alone)")
    for k in (4, 8, 16, 32):
        row(f"4 dense forward k{k}/source", fwd.filter(pl.col("drank") < k), "f")
    for k in (1, 2, 4, 8, 16):
        row(f"4 dense reverse k{k}", rev.filter(pl.col("rrank") < k), "r")
    print("-- 4b current + one direction")
    for k in (4, 8, 16, 32):
        row(f"4b current + dense fwd k{k}", pl.concat([ex, fwd.filter(pl.col('drank') < k).select(KEY)]), "cf")
    for k in (1, 2, 4, 8, 16):
        row(f"4b current + dense rev k{k}", pl.concat([ex, rev.filter(pl.col('rrank') < k).select(KEY)]), "cr")
    if trev is not None:
        print("-- 5 reverse TF-IDF (name char 3-5, S2/S3 -> top-k S1)")
        for k in (1, 2, 4, 8):
            row(f"5 tfidf reverse k{k} (alone)", trev.filter(pl.col("rrank") < k), "t")
        for k in (1, 2, 4, 8):
            row(f"5 current + tfidf reverse k{k}", pl.concat([ex, trev.filter(pl.col('rrank') < k).select(KEY)]), "tu")
        print("-- 6 current + dense k8 + tfidf reverse k4")
        row("6 current + dense k8 + trev k4", pl.concat([ex, dk(8), trev.filter(pl.col("rrank") < 4).select(KEY)]))

    print("\n-- overlap of dense with the current candidate set")
    for k in (4, 8, 16, 32):
        d = dk(k).unique()
        inter = d.join(ex, on=KEY, how="semi").height
        dh = gt.join(d, on=KEY, how="semi")
        new = dh.join(ex, on=KEY, how="anti").height
        print(f"  dense k{k}: {d.height / n_s1:6.2f} cands/S1 | already in current {inter / d.height:6.1%} | new candidates {d.height - inter:,} | "
              f"true pairs found {dh.height:,} of which already in current {dh.height - new:,} and genuinely NEW {new:,}")
    print("DONE")


if __name__ == "__main__":
    main()
