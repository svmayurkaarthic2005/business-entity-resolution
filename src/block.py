"""Phase 3: blocking / candidate generation.

Usage:  python src/block.py --split VAL [--frac 0.2]     # train S1 (VAL ids) vs full train S2/S3 pool
        python src/block.py --split TEST                 # test S1 vs test S2/S3
Output: cache/candidates_<split>.parquet
        s1_id, cand_id, source(2/3), channel(bitmask 1=char 2=word 4=exact), rank, score,
        char_cos, word_cos        (rank 0 = best within (s1, source); KMAX kept per source)
Per country (open set of labels), per source, S1 in chunks of 50K:
  A  char_wb 3-4gram TF-IDF on name_core      -> top KMAX
  B  word TF-IDF on name_core + house_no + street + city -> top KMAX
  C  exact keys (name_core; house_no+street token+city; postcode+first name token), block <= 200
  union -> exact char & word cosine for every pair -> score = char + word -> keep top KMAX
"""
import argparse
import os
import pickle
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import CACHE, log

N = CACHE / "norm"
TF = CACHE / "tfidf"
KMAX, CHUNK, MAX_BLOCK, THREADS = 30, 50_000, 200, os.cpu_count()
MAX_DF = 0.01  # drop n-grams/words in >1% of names: ~8x faster top-n, negligible signal loss
POOL = {"VAL": "train", "LOCO": "train", "TEST": "test"}


def texts(df):
    """The two strings each record is indexed by."""
    return df.with_columns(
        t_char=pl.col("name_core").fill_null(""),
        t_word=pl.concat_str([pl.col("name_core"), pl.col("house_no"), pl.col("street_tokens"),
                              pl.col("city_guess")], separator=" ", ignore_nulls=True))


def load(split, frac):
    """S1 query rows for the split and the S2/S3 pools (with texts)."""
    pool = POOL[split]
    s1 = pl.read_parquet(N / f"{pool}_source1.parquet")
    if split != "TEST":
        col = "rand" if split == "VAL" else "loco"
        ids = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col(col) == "VAL").select(pl.col("s1_id").alias("id"))
        s1 = s1.join(ids, on="id")
    if frac:
        s1 = s1.sample(fraction=frac, seed=42)
    pools = {s: texts(pl.read_parquet(N / f"{pool}_source{s}.parquet")) for s in (2, 3)}
    return texts(s1), pools


def vectorizers(pool, country, pools):
    """Fit (or load cached) char and word TF-IDF for one pool+country; return vecs and pool matrices."""
    path = TF / f"{pool}_{country}_df{MAX_DF}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    TF.mkdir(parents=True, exist_ok=True)
    s1 = texts(pl.read_parquet(N / f"{pool}_source1.parquet").filter(pl.col("country") == country))
    out = {"vec": {}, "mat": {}}
    for ch, kw in (("char", dict(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=MAX_DF)),
                   ("word", dict(analyzer="word", token_pattern=r"\S+", min_df=1, max_df=MAX_DF))):
        corpus = pl.concat([s1[f"t_{ch}"]] + [pools[s][f"t_{ch}"] for s in (2, 3)]).to_list()
        v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
        x = v.fit_transform(corpus)  # one pass; pool rows are sliced out below
        out["vec"][ch] = v
        n1, n2 = len(s1), len(pools[2])
        out["mat"][(ch, 2)], out["mat"][(ch, 3)] = x[n1:n1 + n2], x[n1 + n2:]
        del x
        log(f"  tfidf {ch} {country}: vocab {len(v.vocabulary_):,}")
    pickle.dump(out, open(path, "wb"), protocol=5)
    return out


def topn(q, poolT, k):
    """Top-k cosine neighbours for every query row -> (query_row, pool_row, cos)."""
    c = sp_matmul_topn(q, poolT, top_n=k, threshold=0.05, sort=True, n_threads=THREADS).tocoo()
    return c.row.astype(np.int32), c.col.astype(np.int32), c.data.astype(np.float32)


def exact_pairs(q, pool):
    """Exact-key candidates -> (query_row, pool_row). Keys with pool block > MAX_BLOCK skipped."""
    keys = [pl.when(pl.col("name_core").str.len_chars() >= 3).then(pl.col("name_core")),
            pl.concat_str([pl.col("house_no"), pl.col("street_tokens").str.split(" ").list.first(),
                           pl.col("city_guess")], separator="|"),
            pl.concat_str([pl.col("postcode"), pl.col("name_core").str.split(" ").list.first()], separator="|")]
    out = []
    for i, k in enumerate(keys):
        a = q.with_row_index("qi").select("qi", key=k).drop_nulls().filter(pl.col("key") != "")
        b = pool.with_row_index("pi").select("pi", key=k).drop_nulls().filter(pl.col("key") != "")
        b = b.filter(pl.len().over("key") <= MAX_BLOCK)
        out.append(a.join(b, on="key").select("qi", "pi"))
    e = pl.concat(out).unique()
    return e["qi"].to_numpy().astype(np.int32), e["pi"].to_numpy().astype(np.int32)


def pair_cos(qm, pm, qi, pi):
    """Row-wise cosine between query row qi and pool row pi (both L2-normalised)."""
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), 2_000_000):
        e = slice(s, s + 2_000_000)
        out[e] = np.asarray(qm[qi[e]].multiply(pm[pi[e]]).sum(1)).ravel()
    return out


def block_country(q, pools, tf):
    """All candidates for the S1 rows of one country."""
    res = []
    qm = {ch: tf["vec"][ch].transform(q[f"t_{ch}"].to_list()) for ch in ("char", "word")}
    for s in (2, 3):
        pT = {ch: tf["mat"][(ch, s)].T.tocsr() for ch in ("char", "word")}
        ex_q, ex_p = exact_pairs(q, pools[s])
        for c0 in range(0, len(q), CHUNK):
            c1 = min(c0 + CHUNK, len(q))
            parts = []
            for bit, ch in ((1, "char"), (2, "word")):
                r, p, _ = topn(qm[ch][c0:c1], pT[ch], KMAX)
                parts.append(pl.DataFrame({"qi": r + c0, "pi": p, "channel": np.full(len(r), bit, np.int8)}))
            m = (ex_q >= c0) & (ex_q < c1)
            parts.append(pl.DataFrame({"qi": ex_q[m], "pi": ex_p[m], "channel": np.full(m.sum(), 4, np.int8)}))
            u = pl.concat(parts).group_by("qi", "pi").agg(pl.col("channel").sum())
            qi, pi = u["qi"].to_numpy(), u["pi"].to_numpy()
            u = u.with_columns(char_cos=pair_cos(qm["char"], tf["mat"][("char", s)], qi, pi),
                               word_cos=pair_cos(qm["word"], tf["mat"][("word", s)], qi, pi))
            u = (u.with_columns(score=pl.col("char_cos") + pl.col("word_cos"))
                  .with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("qi") - 1)
                  .filter(pl.col("rank") < KMAX))
            res.append(u.with_columns(
                s1_id=pl.Series(q["id"].to_numpy()[u["qi"].to_numpy()]),
                cand_id=pl.Series(pools[s]["id"].to_numpy()[u["pi"].to_numpy()]),
                source=pl.lit(s, pl.Int8)))
        log(f"  source {s}: {sum(len(r) for r in res):,} candidate rows so far")
    return pl.concat(res).select("s1_id", "cand_id", "source", "channel",
                                 pl.col("rank").cast(pl.Int16), "score", "char_cos", "word_cos")


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
        grp = lambda g: {str(kk[0]): round(v, 4) for kk, v in t.group_by(g).agg(pl.col("hit").mean()).iter_rows()}
        rows.append({"K_per_source": k, "pair_recall": round(t["hit"].mean(), 4), "s1_all_found": round(full, 4),
                     "avg_cands": round(size.mean(), 2), "p95_cands": float(size.quantile(0.95)),
                     "by_country": grp("country"), "by_source": grp("source"),
                     "indic_name": round(t.filter(pl.col("indic"))["hit"].mean(), 4),
                     "empty_addr": round(t.filter(pl.col("addr_missing"))["hit"].mean(), 4)})
    for r in rows:
        print(r)
    return rows, truth


def show_missed(truth, cands, split, k=10, n=20):
    """Print n true pairs that blocking lost at K=k."""
    c = cands.filter(pl.col("rank") < k).select("s1_id", "cand_id", "source")
    miss = truth.join(c, on=["s1_id", "cand_id", "source"], how="anti").sample(n, seed=1)
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "name_raw", "name_core", "addr_norm"])
    for r in miss.iter_rows(named=True):
        a = s1.row(r["s1_id"], named=True)
        b = pl.read_parquet(N / f"train_source{r['source']}.parquet",
                            columns=["name_raw", "name_core", "addr_norm"]).row(r["cand_id"], named=True)
        print(f"\nS1 : {a['name_raw']} | core={a['name_core']} | {a['addr_norm']}")
        print(f"S{r['source']} : {b['name_raw']} | core={b['name_core']} | {b['addr_norm']}")


FRAC = None


def main():
    global FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="VAL", choices=["VAL", "LOCO", "TEST"])
    ap.add_argument("--frac", type=float, default=None, help="use a fraction of the split's S1 rows")
    a = ap.parse_args()
    FRAC = a.frac
    t0 = time.time()
    q, pools = load(a.split, a.frac)
    log(f"{a.split}: {len(q):,} S1 rows; pools S2 {len(pools[2]):,} S3 {len(pools[3]):,}")
    out = []
    for country in q["country"].unique().sort().to_list():
        pc = {s: pools[s].filter(pl.col("country") == country) for s in (2, 3)}
        log(f"{country}: {int((q['country'] == country).sum()):,} S1, S2 {len(pc[2]):,}, S3 {len(pc[3]):,}")
        tf = vectorizers(POOL[a.split], country, pc)
        out.append(block_country(q.filter(pl.col("country") == country), pc, tf))
        del tf
    cands = pl.concat(out)
    path = CACHE / f"candidates_{a.split}{'_frac' if a.frac else ''}.parquet"
    cands.write_parquet(path)
    log(f"wrote {path.name}: {len(cands):,} rows in {time.time() - t0:.0f}s")
    if a.split != "TEST":
        _, truth = evaluate(cands, a.split)
        show_missed(truth, cands, a.split)


if __name__ == "__main__":
    main()
