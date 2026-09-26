"""DENSE-embedding blocking with intfloat/multilingual-e5-small (MIT, 118M), both directions, same-country only.
Usage: python src/exp_dense.py [pool]      pool = train (default: encode + search + VAL report) | test (encode + search only)
Writes exp/dense/: emb_<pool>_s{1,2,3}.npy (fp16), fwd_<pool>.parquet (S1 -> top-32 per source), rev_<pool>.parquet (S2/S3 -> top-16 S1).
Report (train): on the labelled VAL S1 (same as exp_block2 / exp_reverse), for current TF-IDF, TF-IDF reverse, dense forward k, dense
reverse k and unions: pair recall, NEW true pairs, recall AMONG the pairs the current blocker misses, candidates/S1, oracle F0.5.
Text = "query: <raw name> | <normalised address>" on both sides (symmetric task -> same prefix), mean pooling, L2-normalised.
"""
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from common import CACHE, log

N = CACHE / "norm"
OUTD = CACHE.parent / "exp" / "dense"
KEY = ["s1_id", "cand_id", "source"]
MODEL = "intfloat/multilingual-e5-small"
KF, KR = 32, 16          # forward top-k per source, reverse top-k S1


def texts(pool, s):
    d = pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id", "country", "name_raw", "addr_norm"])
    t = d.select(pl.concat_str([pl.lit("query: "), pl.col("name_raw").fill_null(""), pl.lit(" | "), pl.col("addr_norm").fill_null("")]))
    return d.select("id", "country"), t.to_series().to_list()


@torch.no_grad()
def encode(tok, model, txt, bs=2048):
    order = np.argsort([len(x) for x in txt])
    out = np.empty((len(txt), 384), np.float16)
    for i in range(0, len(txt), bs):
        idx = order[i:i + bs]
        b = tok([txt[j] for j in idx], max_length=64, truncation=True, padding=True, return_tensors="pt").to("cuda")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
        out[idx] = e.cpu().numpy()
    return out


@torch.no_grad()
def topk(q, keys, k, qb=1024, kb=1_000_000):
    """Exact cosine top-k of every row of q (np fp16) against keys (np fp16). Returns (idx int64, score fp16)."""
    kt = [torch.from_numpy(keys[j:j + kb]).cuda() for j in range(0, len(keys), kb)]
    I, S = np.empty((len(q), k), np.int64), np.empty((len(q), k), np.float16)
    for i in range(0, len(q), qb):
        qq = torch.from_numpy(q[i:i + qb]).cuda()
        vs, ix = [], []
        for j, kk in enumerate(kt):
            s = qq @ kk.T
            v, x = s.topk(min(k, s.shape[1]), dim=1)
            vs.append(v)
            ix.append(x + j * kb)
        v, x = torch.cat(vs, 1), torch.cat(ix, 1)
        v2, p = v.topk(k, dim=1)
        I[i:i + qb], S[i:i + qb] = torch.gather(x, 1, p).cpu().numpy(), v2.cpu().numpy()
    del kt
    torch.cuda.empty_cache()
    return I, S


def build(pool):
    OUTD.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    meta, emb = {}, {}
    for s in (1, 2, 3):
        f = OUTD / f"emb_{pool}_s{s}.npy"
        meta[s], txt = texts(pool, s)
        if f.exists():
            emb[s] = np.load(f)
        else:
            t = time.time()
            emb[s] = encode(tok, model, txt)
            np.save(f, emb[s])
            log(f"encoded {pool} S{s}: {len(txt):,} in {time.time() - t:.0f}s")
    del model
    torch.cuda.empty_cache()
    fwd, rev = [], []
    for c in sorted(meta[1]["country"].unique().to_list()):
        m1 = (meta[1]["country"] == c).to_numpy()
        ids1, e1 = meta[1]["id"].to_numpy()[m1], emb[1][m1]
        pools = {s: ((meta[s]["country"] == c).to_numpy()) for s in (2, 3)}
        for s in (2, 3):
            idp, ep = meta[s]["id"].to_numpy()[pools[s]], emb[s][pools[s]]
            t = time.time()
            I, S = topk(e1, ep, KF)
            fwd.append(pl.DataFrame({"s1_id": np.repeat(ids1, KF), "cand_id": idp[I.ravel()], "dscore": S.ravel(),
                                     "drank": np.tile(np.arange(KF, dtype=np.int8), len(ids1))}).with_columns(source=pl.lit(s, pl.Int8)))
            I, S = topk(ep, e1, KR)
            rev.append(pl.DataFrame({"cand_id": np.repeat(idp, KR), "s1_id": ids1[I.ravel()], "rscore": S.ravel(),
                                     "rrank": np.tile(np.arange(KR, dtype=np.int8), len(idp))}).with_columns(source=pl.lit(s, pl.Int8)))
            log(f"  {c} S{s}: fwd {len(ids1):,} x {len(idp):,}, rev done in {time.time() - t:.0f}s")
    fwd, rev = pl.concat(fwd), pl.concat(rev)
    fwd.write_parquet(OUTD / f"fwd_{pool}.parquet")
    rev.write_parquet(OUTD / f"rev_{pool}.parquet")
    return fwd, rev


def report():
    from exp_block2 import report as rep
    from exp_budget import val_truth
    val_ids, truth_n, gt = val_truth()
    n_s1, n_true = truth_n.height, gt.height
    cv = pl.read_parquet(CACHE / "candidates_VAL.parquet").join(val_ids, on="s1_id", how="semi")
    ex = cv.filter((pl.col("rank") < 10) & (pl.col("score") >= 0.003)).select(KEY)
    missed = gt.join(ex, on=KEY, how="anti")
    fwd = pl.read_parquet(OUTD / "fwd_train.parquet").join(val_ids, on="s1_id", how="semi")
    rev = pl.read_parquet(OUTD / "rev_train.parquet").join(val_ids, on="s1_id", how="semi")
    tr = CACHE.parent / "exp" / "reverse"
    trev = {k: pl.read_parquet(f).join(val_ids, on="s1_id", how="semi") for k, f in (("name", tr / "rev_name.parquet"), ("hyb", tr / "rev_hyb.parquet")) if f.exists()}
    print(f"\nVAL S1 {n_s1:,}, true pairs {n_true:,}, missed by current blocker {missed.height:,} ({missed.height / n_true:.2%})")

    def row(name, f):
        f = f.select(KEY).unique()
        rec_m = missed.join(f, on=KEY, how="semi").height
        print(f"  [missed recovered {rec_m:5,} / {missed.height:,} = {rec_m / max(missed.height, 1):6.1%}]", end=" ")
        rep(name, f, ex, truth_n, gt, n_s1, n_true)

    dk = lambda k: pl.concat([fwd.filter(pl.col("drank") < k).select(KEY), rev.filter(pl.col("rrank") < min(k, KR)).select(KEY)])
    print("\n--- 1. current   2. dense only (fwd top-k per source + rev top-min(k,16))   3. current + dense")
    row("1 current TF-IDF (K10)", ex)
    for k in (4, 8, 16, 32):
        row(f"2 dense only k{k}", dk(k))
    for k in (4, 8, 16, 32):
        row(f"3 current + dense k{k}", pl.concat([ex, dk(k)]))
    print("\n--- directions separately")
    for k in (4, 8, 16, 32):
        row(f"dense fwd only k{k}/source", fwd.filter(pl.col("drank") < k))
    for k in (1, 2, 4, 8, 16):
        row(f"dense rev only k{k}", rev.filter(pl.col("rrank") < k))
    for k, r in trev.items():
        for kk in (2, 4, 8):
            row(f"tfidf-reverse {k} k{kk}", r.filter(pl.col("rrank") < kk))
    print("\n--- current + asymmetric dense k (fwd kf, rev kr), + tfidf-reverse top-4")
    trev_all = pl.concat([r.filter(pl.col("rrank") < 4).select(KEY) for r in trev.values()]) if trev else ex.head(0)
    for kf in (4, 8, 16, 32):
        for kr in (2, 4, 8, 16):
            u = pl.concat([ex, fwd.filter(pl.col("drank") < kf).select(KEY), rev.filter(pl.col("rrank") < kr).select(KEY)])
            row(f"current + dfwd{kf} + drev{kr}", u)
            if trev:
                row(f"current + trev4 + dfwd{kf} + drev{kr}", pl.concat([u, trev_all]))
    print("DONE")


if __name__ == "__main__":
    pool = sys.argv[1] if len(sys.argv) > 1 else "train"
    if not (OUTD / f"rev_{pool}.parquet").exists():
        build(pool)
    if pool == "train":
        report()
