"""Follow-up to exp_block2: how few candidates does the empty-address channel need?  (VAL, labelled; ~2 min)
Usage: python src/exp_block2b.py
Sweeps top-k of the empty-address name retrieval (char + word) and reports the added pair recall, extra candidates per S1,
recall per extra candidate, and the F0.5 ceiling on top of the existing set. Also a 'tight' variant that keeps a pair only if
the name cosine is high (>= 0.6), which is what a candidate with no address can honestly rely on.
"""
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

import exp_block2 as B
from common import CACHE, log

KEY = B.KEY


def empty_channel(s1, pools, k, thr):
    out = []
    for c in sorted(s1["country"].unique().to_list()):
        q = s1.filter(pl.col("country") == c)
        pc = {s: pools[s].filter((pl.col("country") == c) & pl.col("addr_missing")) for s in (2, 3)}
        for kw in (dict(analyzer="char_wb", ngram_range=(3, 4), min_df=1), dict(analyzer="word", token_pattern=r"\S+", min_df=1)):
            corpus = pl.concat([pc[2]["name_core"], pc[3]["name_core"]]).fill_null("").to_list()
            v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
            x = v.fit_transform(corpus)
            n2 = len(pc[2])
            qm = v.transform(q["name_core"].fill_null("").to_list())
            for s, m in ((2, x[:n2]), (3, x[n2:])):
                r, cc = B.topn(qm, m.T.tocsr(), k, thr)
                out.append(B.pairs_df(q["id"].to_numpy(), pc[s]["id"].to_numpy(), r, cc, s))
    return pl.concat(out).unique()


def main():
    s1, pools, truth_n, gt, val_ids = B.load()
    n_s1, n_true = truth_n.height, gt.height
    cv = pl.read_parquet(CACHE / "candidates_VAL.parquet").join(val_ids, on="s1_id", how="semi")
    existing = cv.filter((pl.col("rank") < 10) & (pl.col("score") >= 0.003)).select(KEY)
    print(f"{'variant':26s} addedTrue  addedRecall  extraCands/S1  trueGainPer1000Cands  ceilingF0.5  pairRecall")
    base_c = B.ceiling(existing, truth_n, gt)
    for k, thr in ((1, 0.05), (2, 0.05), (3, 0.05), (5, 0.05), (10, 0.05), (3, 0.6), (5, 0.6), (10, 0.6), (5, 0.8)):
        ch = empty_channel(s1, pools, k, thr)
        u = pl.concat([existing, ch.select(KEY)]).unique()
        extra = u.height - existing.height
        new_true = gt.join(u, on=KEY, how="semi").height - gt.join(existing, on=KEY, how="semi").height
        print(f"empty top{k:<2d} thr{thr:<4}         {new_true:6,}   {new_true / n_true:.4f}      {extra / n_s1:7.2f}       "
              f"{1000 * new_true / max(extra, 1):8.1f}          {B.ceiling(u, truth_n, gt):.5f}   {gt.join(u, on=KEY, how='semi').height / n_true:.4f}", flush=True)
    print(f"(existing ceiling {base_c:.5f}, pair recall {gt.join(existing, on=KEY, how='semi').height / n_true:.4f})")


if __name__ == "__main__":
    main()
