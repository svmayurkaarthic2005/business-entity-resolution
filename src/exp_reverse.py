"""REVERSE blocking (target -> S1): every S2/S3 record retrieves its top-k S1 records (same country, ALL train S1 as index).
Each target has at most one owner, so its best S1 is usually the owner even when that S1 has many look-alike targets.
Usage: python src/exp_reverse.py        (writes exp/reverse/; no model, no TEST, output_full untouched)
Views: name = char_wb(3,5) on name_latin ; hyb = char_wb(3,5) on name_core + address.  Reports on the labelled VAL S1 (same as exp_block2):
pair recall, candidates/S1, new true pairs vs the production set, perfect-matcher F0.5 ceiling -- per view and k, alone and unioned.
"""
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from common import CACHE, log
from exp_block2 import KEY, report, topn
from exp_budget import val_truth

N = CACHE / "norm"
OUTR = CACHE.parent / "exp" / "reverse"
K = 8
VIEWS = {"name": pl.col("name_latin").fill_null(""),
         "hyb": pl.concat_str([pl.col("name_core").fill_null(""), pl.col("addr_norm").fill_null("")], separator=" | ")}


def reverse(view, expr):
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country", "name_latin", "name_core", "addr_norm"])
    out = []
    for c in sorted(s1["country"].unique().to_list()):
        q1 = s1.filter(pl.col("country") == c)
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.005, dtype=np.float32, sublinear_tf=True)
        idxT = v.fit_transform(q1.select(expr).to_series().to_list()).T.tocsr()
        ids1 = q1["id"].to_numpy()
        for s in (2, 3):
            t = time.time()
            p = pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "country", "name_latin", "name_core", "addr_norm"]).filter(pl.col("country") == c)
            qm = v.transform(p.select(expr).to_series().to_list())
            pid = p["id"].to_numpy()
            for c0 in range(0, qm.shape[0], 50_000):
                r, cc = topn(qm[c0:c0 + 50_000], idxT, K, thr=0.02)
                d = pl.DataFrame({"cand_id": pid[r + c0], "s1_id": ids1[cc]}).with_columns(source=pl.lit(s, pl.Int8))
                out.append(d.with_columns(rrank=pl.int_range(pl.len()).over("cand_id").cast(pl.Int8)))  # topn rows are sorted by score
            log(f"  {view} {c} S{s}: {len(p):,} queries {time.time() - t:.0f}s")
    return pl.concat(out)


def main():
    OUTR.mkdir(parents=True, exist_ok=True)
    val_ids, truth_n, gt = val_truth()
    n_s1, n_true = truth_n.height, gt.height
    cv = pl.read_parquet(CACHE / "candidates_VAL.parquet").join(val_ids, on="s1_id", how="semi")
    existing = cv.filter((pl.col("rank") < 10) & (pl.col("score") >= 0.003)).select(KEY)
    report("existing (K10, floor .003)", existing, existing, truth_n, gt, n_s1, n_true)
    views = {}
    for name, expr in VIEWS.items():
        log(f"view {name}")
        r = reverse(name, expr)
        r.write_parquet(OUTR / f"rev_{name}.parquet")
        views[name] = r.join(val_ids, on="s1_id", how="semi")
    for name, r in views.items():
        for k in (1, 2, 4, 8):
            f = r.filter(pl.col("rrank") < k).select(KEY)
            report(f"rev {name} k{k} (alone)", f, existing, truth_n, gt, n_s1, n_true)
            report(f"existing + rev {name} k{k}", pl.concat([existing, f]).unique(), existing, truth_n, gt, n_s1, n_true)
    both = pl.concat([v.select(*KEY, "rrank") for v in views.values()])
    for k in (1, 2, 4, 8):
        f = both.filter(pl.col("rrank") < k).select(KEY)
        report(f"existing + rev both k{k}", pl.concat([existing, f]).unique(), existing, truth_n, gt, n_s1, n_true)
    print("DONE")


if __name__ == "__main__":
    main()
