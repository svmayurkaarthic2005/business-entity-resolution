"""Phase 3: blocking / candidate generation.

Usage:  python src/block.py --split FIT --frac 0.01 --raw    # training rows for the re-ranker
        python src/block.py --fit-rerank                     # learn the re-ranker (cache/rerank.pkl)
        python src/block.py --split VAL [--frac 0.2]         # train S1 (VAL ids) vs full train S2/S3 pool
        python src/block.py --split TRAIN                    # all train S1 (model training + reverse features)
        python src/block.py --split TEST                     # test S1 vs test S2/S3
Output: cache/candidates_<split>.parquet
        s1_id, cand_id, source(2/3), channel(bitmask 1=name-char 2=word 4=exact 8=addr-char), rank, score,
        + ranking features (cosines, per-S1 maxima/ranks)   (rank 0 = best within (s1, source))
Per country (open set of labels), per source, S1 in chunks of 50K:
  1  char_wb 3-4gram TF-IDF on name_core                      -> top RETR
  2  word TF-IDF on name_core + house_no + street + city      -> top RETR
  8  char_wb 3-4gram TF-IDF on the full normalised address    -> top RETR
  4  exact keys (name_core; house_no+street token+city; postcode+first name token), block <= 200
  union -> exact cosines for every pair -> learned re-ranker (or cos sum) -> keep top KEEP (--k)
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import CACHE, log

N = CACHE / "norm"
TF = CACHE / "tfidf"
RETR, CHUNK, MAX_BLOCK, THREADS = 30, 20_000, 200, os.cpu_count()  # RETR = top-n per channel
WORKERS = int(os.environ.get("BLOCK_WORKERS", 1))  # sparse_dot_topn holds the GIL -> 1 worker, all cores to top-n
KEEP = 30  # candidates kept per (S1, source) after re-ranking; --k
FLOOR = 0.0  # ...and only if re-ranker probability >= FLOOR; --floor (VAL: k=10, floor=0.003 -> 97.4% recall, 11 cands/S1)
MAX_DF = 0.01  # drop n-grams/words in >1% of names: ~8x faster top-n, negligible signal loss
POOL = {"VAL": "train", "LOCO": "train", "FIT": "train", "TRAIN": "train", "TEST": "test"}
CHANNELS = ((1, "char"), (2, "word"), (8, "addr"))
FEATS = ["char_cos", "word_cos", "addr_cos", "ch_char", "ch_word", "ch_exact", "ch_addr", "am",
         "cc_max", "wc_max", "ac_max", "r_cc", "r_wc", "r_ac", "n_u"]
RERANK = CACHE / "rerank.pkl"


def texts(df):
    """The strings each record is indexed by."""
    return df.with_columns(
        t_char=pl.col("name_core").fill_null(""),
        t_word=pl.concat_str([pl.col("name_core"), pl.col("house_no"), pl.col("street_tokens"),
                              pl.col("city_guess")], separator=" ", ignore_nulls=True),
        t_addr=pl.col("addr_norm").fill_null(""))


def load(split, frac):
    """S1 query rows for the split and the S2/S3 pools (with texts)."""
    pool = POOL[split]
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet")
    if split not in ("TEST", "TRAIN"):  # TRAIN = every train S1 (mirrors TEST: all S1 queried)
        col, val = {"VAL": ("rand", "VAL"), "FIT": ("rand", "FIT"), "LOCO": ("loco", "VAL")}[split]
        ids = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col(col) == val).select(pl.col("s1_id").alias("id"))
        s1 = s1.join(ids, on="id")
    if frac:
        s1 = s1.sample(fraction=frac, seed=42)
    pools = {s: texts(pl.read_parquet(N / f"{pool}_source{s}.parquet")) for s in (2, 3)}
    return texts(s1), pools


def fit_tfidf(pool, country, pools, channels, tag):
    """Fit (or load cached) TF-IDF for the given channels of one pool+country."""
    path = TF / f"{pool}_{country}_{tag}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    TF.mkdir(parents=True, exist_ok=True)
    s1 = texts(pl.read_parquet(N / f"{pool}_source1.parquet").filter(pl.col("country") == country))
    out = {"vec": {}, "mat": {}}
    for ch in channels:
        kw = (dict(analyzer="word", token_pattern=r"\S+", min_df=1) if ch == "word"
              else dict(analyzer="char_wb", ngram_range=(3, 4), min_df=2))
        corpus = pl.concat([s1[f"t_{ch}"]] + [pools[s][f"t_{ch}"] for s in (2, 3)]).to_list()
        v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, max_df=MAX_DF, **kw)
        x = v.fit_transform(corpus)  # one pass; pool rows are sliced out below
        out["vec"][ch] = v
        n1, n2 = len(s1), len(pools[2])
        out["mat"][(ch, 2)], out["mat"][(ch, 3)] = x[n1:n1 + n2], x[n1 + n2:]
        del x
        log(f"  tfidf {ch} {country}: vocab {len(v.vocabulary_):,}")
    pickle.dump(out, open(path, "wb"), protocol=5)
    return out


def vectorizers(pool, country, pools):
    """Name char + word TF-IDF (cached as df<MAX_DF>) plus address char TF-IDF (cached as addr)."""
    tf = fit_tfidf(pool, country, pools, ("char", "word"), f"df{MAX_DF}")
    ta = fit_tfidf(pool, country, pools, ("addr",), "addr")
    tf["vec"].update(ta["vec"])
    tf["mat"].update(ta["mat"])
    return tf


def topn(q, poolT, k):
    """Top-k cosine neighbours for every query row -> (query_row, pool_row)."""
    c = sp_matmul_topn(q, poolT, top_n=k, threshold=0.05, sort=True, n_threads=THREADS).tocoo()
    return c.row.astype(np.int32), c.col.astype(np.int32)


def exact_pairs(q, pool):
    """Exact-key candidates -> (query_row, pool_row). Keys with pool block > MAX_BLOCK skipped."""
    keys = [pl.when(pl.col("name_core").str.len_chars() >= 3).then(pl.col("name_core")),
            pl.concat_str([pl.col("house_no"), pl.col("street_tokens").str.split(" ").list.first(),
                           pl.col("city_guess")], separator="|"),
            pl.concat_str([pl.col("postcode"), pl.col("name_core").str.split(" ").list.first()], separator="|")]
    out = []
    for k in keys:
        a = q.with_row_index("qi").select("qi", key=k).drop_nulls().filter(pl.col("key") != "")
        b = pool.with_row_index("pi").select("pi", key=k).drop_nulls().filter(pl.col("key") != "")
        b = b.filter(pl.len().over("key") <= MAX_BLOCK)
        out.append(a.join(b, on="key").select("qi", "pi"))
    e = pl.concat(out).unique()
    return e["qi"].to_numpy().astype(np.int32), e["pi"].to_numpy().astype(np.int32)


try:  # nogil sparse row-pair dot product: lets the worker threads run truly in parallel
    from numba import njit

    @njit(nogil=True, cache=True)
    def _pair_dot(ip1, ix1, d1, ip2, ix2, d2, qi, pi, out):
        for k in range(len(qi)):
            a, a1, b, b1 = ip1[qi[k]], ip1[qi[k] + 1], ip2[pi[k]], ip2[pi[k] + 1]
            s = 0.0
            while a < a1 and b < b1:
                if ix1[a] == ix2[b]:
                    s += d1[a] * d2[b]
                    a += 1
                    b += 1
                elif ix1[a] < ix2[b]:
                    a += 1
                else:
                    b += 1
            out[k] = s
except ImportError:  # pragma: no cover
    _pair_dot = None


def pair_cos(qm, pm, qi, pi):
    """Row-wise cosine between query row qi and pool row pi (both L2-normalised, sorted indices)."""
    out = np.zeros(len(qi), np.float32)
    if _pair_dot is not None:
        _pair_dot(qm.indptr, qm.indices, qm.data, pm.indptr, pm.indices, pm.data,
                  np.asarray(qi, np.int64), np.asarray(pi, np.int64), out)
        return out
    for s in range(0, len(qi), 2_000_000):
        e = slice(s, s + 2_000_000)
        out[e] = np.asarray(qm[qi[e]].multiply(pm[pi[e]]).sum(1)).ravel()
    return out


def add_features(u):
    """Ranking features for each (query, candidate) pair of one source."""
    r = lambda c: pl.col(c).rank("ordinal", descending=True).over("qi").cast(pl.Int16)
    return u.with_columns(
        ch_char=(pl.col("channel") & 1) > 0, ch_word=(pl.col("channel") & 2) > 0,
        ch_exact=(pl.col("channel") & 4) > 0, ch_addr=(pl.col("channel") & 8) > 0,
        cc_max=pl.col("char_cos").max().over("qi"), wc_max=pl.col("word_cos").max().over("qi"),
        ac_max=pl.col("addr_cos").max().over("qi"),
        r_cc=r("char_cos"), r_wc=r("word_cos"), r_ac=r("addr_cos"),
        n_u=pl.len().over("qi").cast(pl.Int16))


def score(u, model):
    """Learned match probability if a re-ranker exists, else the plain cosine sum."""
    if model is None:
        return u.with_columns(score=pl.col("char_cos") + pl.col("word_cos") + pl.col("addr_cos"))
    p = model.predict_proba(u.select(FEATS).to_numpy().astype(np.float32))[:, 1]
    return u.with_columns(score=pl.Series(p, dtype=pl.Float32))


_W = {}  # state shared by the worker threads: matrices, pools, model


def _chunk(args):
    """Candidates for query rows c0:c1 against one source (runs in a worker thread)."""
    s, c0, c1 = args
    qm, tf, am, ex_q, ex_p, pT = _W["qm"], _W["tf"], _W["am"][s], _W["ex"][s][0], _W["ex"][s][1], _W["pT"][s]
    parts = []
    for bit, ch in CHANNELS:
        r, p = topn(qm[ch][c0:c1], pT[ch], RETR)
        parts.append(pl.DataFrame({"qi": r + c0, "pi": p, "channel": np.full(len(r), bit, np.int8)}))
    m = (ex_q >= c0) & (ex_q < c1)
    parts.append(pl.DataFrame({"qi": ex_q[m], "pi": ex_p[m], "channel": np.full(m.sum(), 4, np.int8)}))
    u = pl.concat(parts).group_by("qi", "pi").agg(pl.col("channel").sum())
    qi, pi = u["qi"].to_numpy(), u["pi"].to_numpy()
    u = u.with_columns(**{f"{ch}_cos": pair_cos(qm[ch], tf["mat"][(ch, s)], qi, pi) for _, ch in CHANNELS},
                       am=pl.Series(am[pi]))
    u = score(add_features(u), _W["model"])
    u = u.with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("qi") - 1)
    if not _W["raw"]:
        u = u.filter((pl.col("rank") < KEEP) & (pl.col("score") >= FLOOR))
    return u.with_columns(
        s1_id=pl.Series(_W["qid"][u["qi"].to_numpy()]),
        cand_id=pl.Series(_W["pid"][s][u["pi"].to_numpy()]),
        source=pl.lit(s, pl.Int8)).drop("qi", "pi")


def block_country(q, pools, tf, model, raw):
    """All candidates for the S1 rows of one country; chunks run in parallel threads.
    (Forked processes deadlock once polars' thread pool is running, so threads are used.)"""
    global THREADS
    for m in tf["mat"].values():
        m.sort_indices()
    qm = {ch: tf["vec"][ch].transform(q[f"t_{ch}"].to_list()) for _, ch in CHANNELS}
    for m in qm.values():
        m.sort_indices()
    _W.update(tf=tf, model=model, raw=raw, qid=q["id"].to_numpy(), qm=qm,
              pT={s: {ch: tf["mat"][(ch, s)].T.tocsr() for _, ch in CHANNELS} for s in (2, 3)},
              am={s: pools[s]["addr_missing"].to_numpy() for s in (2, 3)},
              pid={s: pools[s]["id"].to_numpy() for s in (2, 3)},
              ex={s: exact_pairs(q, pools[s]) for s in (2, 3)})
    jobs = [(s, c0, min(c0 + CHUNK, len(q))) for s in (2, 3) for c0 in range(0, len(q), CHUNK)]
    THREADS = max(1, os.cpu_count() // WORKERS)
    res, t0 = [], time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(_chunk, jobs), 1):  # in order; log progress every ~10%
            res.append(r)
            if i % max(1, len(jobs) // 10) == 0:
                log(f"    {i}/{len(jobs)} chunks ({time.time() - t0:.0f}s)")
    _W.clear()
    log(f"  {len(jobs)} chunks -> {sum(len(r) for r in res):,} candidate rows")
    return pl.concat(res).select("s1_id", "cand_id", "source", "channel",
                                 pl.col("rank").cast(pl.Int16), "score", *FEATS)


def label(cands):
    """Add is_match from the ground truth."""
    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").select(
        pl.col("s1_id"), pl.col("match_id").alias("cand_id"), pl.col("match_source").alias("source"),
        is_match=pl.lit(True))
    return cands.join(gt, on=["s1_id", "cand_id", "source"], how="left").with_columns(pl.col("is_match").fill_null(False))


def recall_at(c, col, total, ks=(5, 10, 20, 30)):
    """Pair recall@K (per source) when candidates are ordered by `col`; `total` = all true pairs of these S1."""
    c = c.with_columns(rk=pl.col(col).rank("ordinal", descending=True).over("s1_id", "source"))
    return {k: round(c.filter((pl.col("rk") <= k) & pl.col("is_match")).height / total, 4) for k in ks}


def fit_rerank():
    """Train the re-ranker on raw FIT candidates; report recall@K on a held-out S1 group."""
    c = label(pl.read_parquet(CACHE / "candidates_FIT_raw.parquet")).with_columns(
        base=pl.col("char_cos") + pl.col("word_cos") + pl.col("addr_cos"))
    ids = c["s1_id"].unique().sort()
    hold = ids.sample(fraction=0.2, seed=42)
    tr, te = c.filter(~pl.col("s1_id").is_in(hold)), c.filter(pl.col("s1_id").is_in(hold))
    log(f"re-ranker: {len(tr):,} train pairs ({tr['is_match'].mean():.3%} positive), {len(te):,} held-out")
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                           early_stopping=True, random_state=42)
    model.fit(tr.select(FEATS).to_numpy().astype(np.float32), tr["is_match"].to_numpy())
    te = te.with_columns(learned=pl.Series(model.predict_proba(te.select(FEATS).to_numpy().astype(np.float32))[:, 1]))
    total = pl.read_parquet(CACHE / "gt_pairs.parquet").filter(pl.col("s1_id").is_in(hold)).height
    print("held-out recall@K per source  cos-sum:", recall_at(te, "base", total), " learned:", recall_at(te, "learned", total))
    print("ceiling (all union candidates):", round(te["is_match"].sum() / total, 4),
          "| union size per S1+source:", round(len(te) / te.select("s1_id", "source").n_unique(), 1))
    pickle.dump(model, open(RERANK, "wb"))
    log(f"saved {RERANK.name}")


def evaluate(cands, split, ks=(5, 10, 20, 30)):
    """Recall / size table vs the ground truth for a labelled split."""
    col = "rand" if split == "VAL" else "loco"
    s1ids = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col(col) == "VAL").select("s1_id")
    s1ids = s1ids.join(cands.select("s1_id").unique(), on="s1_id", how="semi") if FRAC else s1ids
    truth = pl.read_parquet(CACHE / "gt_pairs.parquet").join(s1ids, on="s1_id") \
              .rename({"match_id": "cand_id", "match_source": "source"})
    meta = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet",
                                      columns=["id", "name_raw", "addr_missing", "country"])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    truth = truth.join(meta, on=["cand_id", "source"]).with_columns(
        indic=pl.col("name_raw").str.contains(r"[ऀ-෿]"))
    rows = []
    for k in ks:
        c = cands.filter(pl.col("rank") < k)
        t = truth.join(c.select("s1_id", "cand_id", "source", hit=pl.lit(True)),
                       on=["s1_id", "cand_id", "source"], how="left").with_columns(pl.col("hit").fill_null(False))
        size = s1ids.join(c.group_by("s1_id").len(), on="s1_id", how="left").fill_null(0)["len"]
        full = t.group_by("s1_id").agg(pl.col("hit").all())["hit"].mean()
        grp = lambda g: {str(kk): round(v, 4) for kk, v in t.group_by(g).agg(pl.col("hit").mean()).iter_rows()}
        rows.append({"K_per_source": k, "pair_recall": round(t["hit"].mean(), 4), "s1_all_found": round(full, 4),
                     "avg_cands": round(size.mean(), 2), "p95_cands": float(size.quantile(0.95)),
                     "by_country": grp("country"), "by_source": grp("source"),
                     "indic_name": round(t.filter(pl.col("indic"))["hit"].mean(), 4),
                     "empty_addr": round(t.filter(pl.col("addr_missing"))["hit"].mean(), 4)})
    for r in rows:
        print(r)
    return rows, truth


def show_missed(truth, cands, k=10, n=20):
    """Print n true pairs that blocking lost at K=k."""
    c = cands.filter(pl.col("rank") < k).select("s1_id", "cand_id", "source")
    miss = truth.join(c, on=["s1_id", "cand_id", "source"], how="anti").sample(n, seed=1)
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["name_raw", "name_core", "addr_norm"])
    src = {s: pl.read_parquet(N / f"train_source{s}.parquet", columns=["name_raw", "name_core", "addr_norm"])
           for s in (2, 3)}
    for r in miss.iter_rows(named=True):
        a, b = s1.row(r["s1_id"], named=True), src[r["source"]].row(r["cand_id"], named=True)
        print(f"\nS1 : {a['name_raw']} | core={a['name_core']} | {a['addr_norm']}")
        print(f"S{r['source']} : {b['name_raw']} | core={b['name_core']} | {b['addr_norm']}")


FRAC = None


def main():
    global FRAC, KEEP, FLOOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="VAL", choices=list(POOL))
    ap.add_argument("--frac", type=float, default=None, help="use a fraction of the split's S1 rows")
    ap.add_argument("--raw", action="store_true", help="keep the whole union (re-ranker training rows)")
    ap.add_argument("--fit-rerank", action="store_true", help="train the re-ranker from candidates_FIT_raw")
    ap.add_argument("--k", type=int, default=30, help="candidates kept per (S1, source)")
    ap.add_argument("--floor", type=float, default=0.0, help="minimum re-ranker probability to keep a candidate")
    a = ap.parse_args()
    if a.fit_rerank:
        return fit_rerank()
    FRAC, KEEP, FLOOR = a.frac, a.k, a.floor
    t0 = time.time()
    model = None if a.raw or not RERANK.exists() else pickle.load(open(RERANK, "rb"))
    log(f"re-ranker: {'learned' if model else 'cosine sum'}")
    q, pools = load(a.split, a.frac)
    log(f"{a.split}: {len(q):,} S1 rows; pools S2 {len(pools[2]):,} S3 {len(pools[3]):,}")
    out = []
    for country in q["country"].unique().sort().to_list():
        pc = {s: pools[s].filter(pl.col("country") == country) for s in (2, 3)}
        log(f"{country}: {int((q['country'] == country).sum()):,} S1, S2 {len(pc[2]):,}, S3 {len(pc[3]):,}")
        tf = vectorizers(POOL[a.split], country, pc)
        out.append(block_country(q.filter(pl.col("country") == country), pc, tf, model, a.raw))
        del tf
    cands = pl.concat(out)
    q.select(pl.col("id").alias("s1_id")).write_parquet(CACHE / f"queried_{a.split}.parquet")  # S1 actually queried
    path = CACHE / f"candidates_{a.split}{'_frac' if a.frac and not a.raw else ''}{'_raw' if a.raw else ''}.parquet"
    cands.write_parquet(path)
    log(f"wrote {path.name}: {len(cands):,} rows in {time.time() - t0:.0f}s")
    if a.split in ("VAL", "LOCO", "TRAIN"):
        split = "VAL" if a.split == "TRAIN" else a.split
        if a.split == "TRAIN":
            val = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
            cands = cands.join(val, on="s1_id", how="semi")
        _, truth = evaluate(cands, split)
        show_missed(truth, cands)


if __name__ == "__main__":
    main()
