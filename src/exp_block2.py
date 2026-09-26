"""Phase 1: candidate-recall ceiling with NEW blocking channels, on the labelled VAL S1 (same 33,172 S1 as before).

Usage: python src/exp_block2.py       (writes exp/block2/: channel pairs + log; no model training, no TEST)
Existing set = production kept candidates on VAL (candidates_VAL.parquet: rank < 10 and re-ranker score >= 0.003).
New channels (per country, per source S2/S3, against the FULL train pools):
  exact_core   name_core equal (block <= 200 as in production, and separately UNCAPPED count)
  exact_latin  name_latin equal (keeps legal/noise words that name_core strips)
  core_city    name_core + city_guess equal, any block size (weak signal instead of discarding common names)
  core_state   name_core + state_guess equal
  acronym      candidate name is the initials of the S1 name (EC for "EHPAD Conscrits") + same city
  tok_name     word TF-IDF on name_core only (top 20)
  char_name    char_wb(3,5) TF-IDF on name_latin (top 20; corrupted names, keeps legal words)
  empty_addr   pool restricted to candidates with EMPTY address; char+word name TF-IDF, no df pruning (top 20)
Reports per channel: pair recall, S1 with >=1 true found, S1 fully covered, candidates per S1, NEW true pairs vs the
existing set, overlap; then unions and the theoretical F0.5 ceiling (choose exactly the true pairs that are retrieved).
"""
import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import CACHE, log
from exp_budget import val_truth

N = CACHE / "norm"
OUTD = CACHE.parent / "exp" / "block2"
KEY = ["s1_id", "cand_id", "source"]
THREADS = os.cpu_count()
TOPK = 20


def load():
    val_ids, truth_n, gt = val_truth()
    s1 = pl.read_parquet(N / "train_source1.parquet").join(val_ids.rename({"s1_id": "id"}), on="id")
    pools = {s: pl.read_parquet(N / f"train_source{s}.parquet") for s in (2, 3)}
    return s1, pools, truth_n, gt, val_ids


def topn(q, pT, k, thr=0.05):
    c = sp_matmul_topn(q, pT, top_n=k, threshold=thr, sort=True, n_threads=THREADS).tocoo()
    return c.row.astype(np.int64), c.col.astype(np.int64)


def pairs_df(qid, pid, rows, cols, s):
    return pl.DataFrame({"s1_id": qid[rows], "cand_id": pid[cols]}).with_columns(source=pl.lit(s, pl.Int8))


def tfidf_channel(q, pools_c, textcol, kw, k=TOPK, sub=None):
    """Top-k neighbours per S1 row for each source. sub: optional filter on the pool (pl.Expr)."""
    P = {s: (pools_c[s].filter(sub) if sub is not None else pools_c[s]) for s in (2, 3)}
    corpus = pl.concat([P[2][textcol], P[3][textcol]]).fill_null("").to_list()
    v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
    x = v.fit_transform(corpus)
    n2 = len(P[2])
    mats = {2: x[:n2], 3: x[n2:]}
    qm = v.transform(q[textcol].fill_null("").to_list())
    out = []
    for s in (2, 3):
        pT = mats[s].T.tocsr()
        for c0 in range(0, len(q), 20_000):
            r, c = topn(qm[c0:c0 + 20_000], pT, k)
            out.append(pairs_df(q["id"].to_numpy(), P[s]["id"].to_numpy(), r + c0, c, s))
    return pl.concat(out)


def key_channel(q, pools_c, key, cap=200, uncapped=False):
    """Exact key join. key: list of columns or pl.Expr. Skips pool blocks larger than cap unless uncapped."""
    kq = q.with_row_index("qi").select("qi", "id", key=key).drop_nulls().filter(pl.col("key") != "")
    out, sizes = [], []
    for s in (2, 3):
        kp = pools_c[s].select("id", key=key).drop_nulls().filter(pl.col("key") != "")
        cnt = kp.group_by("key").len("blk")
        kp = kp.join(cnt, on="key")
        sizes.append(kp["blk"])
        if not uncapped:
            kp = kp.filter(pl.col("blk") <= cap)
        j = kq.join(kp, on="key", suffix="_c")
        out.append(j.select(s1_id="id", cand_id="id_c").with_columns(source=pl.lit(s, pl.Int8)))
    return pl.concat(out), pl.concat(sizes)


def run_channels(s1, pools):
    chans = {k: [] for k in ("exact_core", "exact_core_uncapped", "exact_latin", "core_city", "core_state", "acronym",
                             "tok_name", "char_name", "empty_addr")}
    stats = {}
    for c in sorted(s1["country"].unique().to_list()):
        q = s1.filter(pl.col("country") == c)
        pc = {s: pools[s].filter(pl.col("country") == c) for s in (2, 3)}
        log(f"{c}: {len(q):,} S1, S2 {len(pc[2]):,}, S3 {len(pc[3]):,}")
        core = pl.col("name_core").filter(pl.col("name_core").str.len_chars() >= 2)
        t = time.time()
        chans["exact_core"].append(key_channel(q, pc, pl.when(pl.col("name_core").str.len_chars() >= 2).then(pl.col("name_core")))[0])
        p, sz = key_channel(q, pc, pl.when(pl.col("name_core").str.len_chars() >= 2).then(pl.col("name_core")), uncapped=True)
        chans["exact_core_uncapped"].append(p)
        stats[f"{c} core block size p50/p90/p99/max"] = [float(sz.quantile(x)) for x in (0.5, 0.9, 0.99, 1.0)]
        chans["exact_latin"].append(key_channel(q, pc, pl.when(pl.col("name_latin").str.len_chars() >= 2).then(pl.col("name_latin")))[0])
        chans["core_city"].append(key_channel(q, pc, pl.concat_str(["name_core", "city_guess"], separator="|"), cap=5000)[0])
        chans["core_state"].append(key_channel(q, pc, pl.concat_str(["name_core", "state_guess"], separator="|"), cap=5000)[0])
        # acronym: candidate name_core (no space, 2-6 letters) == initials of S1 core (>=2 tokens); same city
        qa = q.with_columns(ini=pl.col("name_core").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join(""),
                            ntok=pl.col("name_core").str.split(" ").list.len()).filter(pl.col("ntok") >= 2)
        outa = []
        for s in (2, 3):
            cp = pc[s].filter(~pl.col("name_core").str.contains(" ") & pl.col("name_core").str.len_chars().is_between(2, 6))
            j = qa.select("id", "ini", "city_guess").join(cp.select(cid="id", ini="name_core", cg="city_guess"), on="ini")
            j = j.filter(pl.col("city_guess").is_not_null() & (pl.col("city_guess") == pl.col("cg")))
            outa.append(j.select(s1_id="id", cand_id="cid").with_columns(source=pl.lit(s, pl.Int8)))
        chans["acronym"].append(pl.concat(outa))
        log(f"  key channels {time.time() - t:.0f}s")
        t = time.time()
        chans["tok_name"].append(tfidf_channel(q, pc, "name_core", dict(analyzer="word", token_pattern=r"\S+", min_df=1, max_df=0.02)))
        log(f"  tok_name {time.time() - t:.0f}s")
        t = time.time()
        chans["char_name"].append(tfidf_channel(q, pc, "name_latin", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.005)))
        log(f"  char_name {time.time() - t:.0f}s")
        t = time.time()
        # empty-address pool: small, so no df pruning; char + word evidence combined by union of both retrievals
        ea = [tfidf_channel(q, pc, "name_core", dict(analyzer="char_wb", ngram_range=(3, 4), min_df=1), k=TOPK, sub=pl.col("addr_missing")),
              tfidf_channel(q, pc, "name_core", dict(analyzer="word", token_pattern=r"\S+", min_df=1), k=TOPK, sub=pl.col("addr_missing"))]
        chans["empty_addr"].append(pl.concat(ea).unique())
        log(f"  empty_addr {time.time() - t:.0f}s")
    return {k: pl.concat(v).unique() for k, v in chans.items()}, stats


def ceiling(found, truth_n, gt):
    """Perfect-matcher macro F0.5 if exactly the retrieved true pairs are predicted (singletons predicted empty)."""
    tp = gt.join(found.select(KEY).unique(), on=KEY, how="semi").group_by("s1_id").len("tp")
    t = truth_n.join(tp, on="s1_id", how="left").fill_null(0)
    f = (pl.when(pl.col("n") == 0).then(1.0).when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n") + pl.col("tp"))))
    return float(t.select(f.mean()).item())


def report(name, found, existing, truth_n, gt, n_s1, n_true, new_only=False):
    f = found.select(KEY).unique()
    hit = gt.join(f, on=KEY, how="semi")
    hit_old = gt.join(existing, on=KEY, how="semi")
    new_true = hit.join(existing, on=KEY, how="anti")
    s1_any = hit["s1_id"].n_unique() / truth_n.filter(pl.col("n") > 0).height
    full = (truth_n.filter(pl.col("n") > 0).join(hit.group_by("s1_id").len("h"), on="s1_id", how="left").fill_null(0)
            .select((pl.col("h") == pl.col("n")).mean()).item())
    extra = f.join(existing, on=KEY, how="anti").height
    print(f"{name:34s} pairRecall {hit.height / n_true:.4f} S1>=1 {s1_any:.4f} S1full {full:.4f} cands/S1 {f.height / n_s1:7.2f} "
          f"| new true {new_true.height:6,} ({new_true.height / n_true:.4f}) extra cands/S1 {extra / n_s1:7.2f} "
          f"| ceilingF0.5 {ceiling(f, truth_n, gt):.5f}", flush=True)
    return hit.height / n_true


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    s1, pools, truth_n, gt, val_ids = load()
    n_s1, n_true = truth_n.height, gt.height
    cv = pl.read_parquet(CACHE / "candidates_VAL.parquet").join(val_ids, on="s1_id", how="semi")
    existing = cv.filter((pl.col("rank") < 10) & (pl.col("score") >= 0.003)).select(KEY)
    k30 = cv.select(KEY)
    log(f"VAL S1 {n_s1:,}, true pairs {n_true:,}; existing kept set {existing.height:,} rows")
    print(f"\n{'channel':34s}")
    report("existing (K10, floor .003)", existing, existing, truth_n, gt, n_s1, n_true)
    report("existing K30 (no floor)", k30, existing, truth_n, gt, n_s1, n_true)
    ch, stats = run_channels(s1, pools)
    for k, v in stats.items():
        print(k, v)
    for k, v in ch.items():
        v.write_parquet(OUTD / f"chan_{k}.parquet")
    print("\n--- each new channel alone, and its effect on top of the existing set")
    for k, v in ch.items():
        report(f"{k} (alone)", v, existing, truth_n, gt, n_s1, n_true)
    print("\n--- existing + one channel")
    for k, v in ch.items():
        report(f"existing + {k}", pl.concat([existing, v.select(KEY)]).unique(), existing, truth_n, gt, n_s1, n_true)
    print("\n--- cumulative union")
    acc = existing
    for k in ("exact_core", "exact_latin", "core_city", "core_state", "acronym", "tok_name", "char_name", "empty_addr"):
        acc = pl.concat([acc, ch[k].select(KEY)]).unique()
        report(f"+ {k}", acc, existing, truth_n, gt, n_s1, n_true)
    report("existing + ALL channels", acc, existing, truth_n, gt, n_s1, n_true)
    # what remains lost
    lost = gt.join(acc, on=KEY, how="anti")
    meta = pl.concat([pools[s].select("id", "addr_missing", "name_raw").with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    lost = lost.join(meta, on=["cand_id", "source"])
    print(f"\nstill lost: {lost.height:,} pairs ({lost.height / n_true:.2%}); with empty candidate address {lost['addr_missing'].mean():.1%}")
    s1n = s1.select(pl.col("id").alias("s1_id"), pl.col("name_raw").alias("n1"), pl.col("addr_norm").alias("a1"))
    for r in lost.join(s1n, on="s1_id").sample(min(25, lost.height), seed=5).iter_rows(named=True):
        print(f"  S1 {r['n1']} | {r['a1']}   <-  S{r['source']} {r['name_raw']} (addr_missing={r['addr_missing']})")


if __name__ == "__main__":
    main()
