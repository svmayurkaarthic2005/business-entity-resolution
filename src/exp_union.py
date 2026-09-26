"""Matcher on the candidate set  current TF-IDF (candidates_TRAIN)  UNION  reverse dense k1  (each S2/S3 record's single nearest S1, e5).  VAL held-out.
Usage: python src/exp_union.py [--from-cache]      -> exp/union/

Checks built in (printed):  1 union == current + rev-k1 exactly (row counts, no duplicates, VAL recall / cands per S1 vs the Pareto sweep 0.9861 / 11.47);
2 every new row gets REAL blocking features (TF-IDF name-char / word / address cosines, ch_*, am, per-S1 maxima and ranks, the production
  re-ranker score, rank recomputed over the union) -- null check on new rows;  3 dense features on ALL rows: dcos (e5 cosine), d_rev (rank of the
  S1 in the record's reverse top-16, 99 if absent), d_rgap (record's best reverse cosine minus this pair's cosine), ch_dense (row only from the dense channel);
4 owner-competition features (cand_best, cand_n_owners, owner_rank, s1_best, ...) recomputed over the whole union for ALL rows, then stage-1 -> stage-2
  groups -> NUM/QUAL/NEW3/NEW4 exactly as the 0.98108 model;  5 same 20 split-halves + production rule (S.evaluate);  6 no validation label enters a
  feature / statistic (token log-odds are out-of-fold as before; rev-dense features are label-free);  7 mean and worst held-out F0.5 added;
8 original-candidate vs newly-added-candidate breakdown of the union model against the saved 0.98108 model (exp/final).
Not retrained: the production stage-1 models (applied to the union rows).  Approximation: r_cc / r_wc / r_ac / n_u of NEW rows are ranked/counted against the KEPT
old rows (the production values were computed on the raw pre-filter union).
"""
import pickle
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import block as BL
import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
import exp_stack4 as X4
import features as F
from common import CACHE, log
from exp_budget import val_truth
from exp_final import assemble, featurise_new

N = CACHE / "norm"
D = CACHE.parent / "exp" / "dense"
OUTU = CACHE.parent / "exp" / "union"
FIN = CACHE.parent / "exp" / "final"
KEY = S.KEY
DENSE = ["dcos", "d_rev", "d_rgap", "ch_dense"]
COMP = ["cand_best", "cand_n_owners", "owner_rank", "s1_best", "s1_n_strong", "gap_next", "gap_owner", "gap_s1"]


def dcos(rows, pool="train"):
    """e5 cosine of every (s1, cand, source) row (embeddings are L2-normalised fp16)."""
    e1 = np.load(D / f"emb_{pool}_s1.npy", mmap_mode="r")
    pos1 = (pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id"]).with_row_index("i1").rename({"id": "s1_id"})
            .with_columns(pl.col("s1_id").cast(rows.schema["s1_id"])))
    r = rows.select(KEY).with_row_index("r").join(pos1, on="s1_id", how="left")
    out = np.zeros(rows.height, np.float32)
    for s in (2, 3):
        es = np.load(D / f"emb_{pool}_s{s}.npy", mmap_mode="r")
        pos = (pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["id"]).with_row_index("i2").rename({"id": "cand_id"})
               .with_columns(pl.col("cand_id").cast(rows.schema["cand_id"])))
        t = r.filter(pl.col("source") == s).join(pos, on="cand_id", how="left")
        ri, a, b = t["r"].to_numpy(), t["i1"].to_numpy(), t["i2"].to_numpy()
        for c0 in range(0, len(ri), 1_000_000):
            sl = slice(c0, c0 + 1_000_000)
            out[ri[sl]] = np.einsum("ij,ij->i", e1[a[sl]].astype(np.float32), es[b[sl]].astype(np.float32))
    return pl.Series("dcos", out)


def blocking_features(new, old, pool="train"):
    """Real blocking columns (block.FEATS + score + rank + channel) for the new rows: same vectorizers, same re-ranker as production."""
    model = pickle.load(open(BL.RERANK, "rb"))
    s1 = BL.texts(pl.read_parquet(N / f"{pool}_source1.parquet"))
    pools = {s: BL.texts(pl.read_parquet(N / f"{pool}_source{s}.parquet")) for s in (2, 3)}
    out = []
    for c in sorted(s1["country"].unique().to_list()):
        pc = {s: pools[s].filter(pl.col("country") == c) for s in (2, 3)}
        tf = BL.vectorizers(pool, c, pc)
        for m in tf["mat"].values():
            m.sort_indices()
        q = s1.filter(pl.col("country") == c)
        qm = {ch: tf["vec"][ch].transform(q[f"t_{ch}"].to_list()) for _, ch in BL.CHANNELS}
        for m in qm.values():
            m.sort_indices()
        qpos = q.select(s1_id="id").with_row_index("qi")
        for s in (2, 3):
            ppos = pc[s].select(cand_id="id", am="addr_missing").with_row_index("pi")
            u = new.filter(pl.col("source") == s).join(qpos, on="s1_id", how="inner").join(ppos, on="cand_id", how="inner")
            if u.height == 0:
                continue
            qi, pi = u["qi"].to_numpy(), u["pi"].to_numpy()
            u = u.with_columns(**{f"{ch}_cos": BL.pair_cos(qm[ch], tf["mat"][(ch, s)], qi, pi) for _, ch in BL.CHANNELS})
            out.append(u.drop("qi", "pi"))
        del tf, qm
        log(f"  blocking features {c}: {sum(o.height for o in out):,} rows so far")
    u = pl.concat(out)
    assert u.height == new.height, (u.height, new.height)
    u = u.with_columns(channel=pl.lit(16, pl.Int8), ch_char=pl.lit(False), ch_word=pl.lit(False), ch_exact=pl.lit(False), ch_addr=pl.lit(False))
    o = old.select("s1_id", "source", "char_cos", "word_cos", "addr_cos", "cc_max", "wc_max", "ac_max", "n_u")
    g = o.group_by("s1_id", "source").agg(pl.col("cc_max").max(), pl.col("wc_max").max(), pl.col("ac_max").max(), pl.col("n_u").max())
    u = u.join(g, on=["s1_id", "source"], how="left")
    u = u.with_columns(cc_max=pl.max_horizontal(pl.col("cc_max").fill_null(0), pl.col("char_cos").max().over("s1_id", "source")),
                       wc_max=pl.max_horizontal(pl.col("wc_max").fill_null(0), pl.col("word_cos").max().over("s1_id", "source")),
                       ac_max=pl.max_horizontal(pl.col("ac_max").fill_null(0), pl.col("addr_cos").max().over("s1_id", "source")),
                       n_u=(pl.col("n_u").fill_null(0) + pl.len().over("s1_id", "source")).cast(pl.Int16))
    u = u.with_row_index("uid")
    cols = ["s1_id", "source", "char_cos", "word_cos", "addr_cos"]
    both = pl.concat([o.select(cols).with_columns(uid=pl.lit(None, pl.UInt32)), u.select(*cols, "uid")])
    r = lambda c: pl.col(c).rank("ordinal", descending=True).over("s1_id", "source").cast(pl.Int16)
    both = both.with_columns(r_cc=r("char_cos"), r_wc=r("word_cos"), r_ac=r("addr_cos")).filter(pl.col("uid").is_not_null())
    u = u.join(both.select("uid", "r_cc", "r_wc", "r_ac"), on="uid").drop("uid")
    p = model.predict_proba(u.select(BL.FEATS).to_numpy().astype(np.float32))[:, 1]
    return u.with_columns(score=pl.Series(p, dtype=pl.Float32))


def build_union_features():
    _, truth_n, gt = val_truth()
    old = pl.read_parquet(CACHE / "candidates_TRAIN.parquet")
    rev = pl.read_parquet(D / "rev_train.parquet")                       # cand_id, s1_id, rscore, rrank(0..15), source  -- label-free
    rev = rev.with_columns([pl.col(c).cast(old.schema[c]) for c in KEY])
    k1 = rev.filter(pl.col("rrank") == 0).select(KEY)
    new = k1.join(old.select(KEY), on=KEY, how="anti")
    log(f"current {old.height:,} rows | rev-dense k1 {k1.height:,} | new (not in current) {new.height:,}")
    nb = blocking_features(new, old)
    nb.select(KEY).write_parquet(OUTU / "new_candidates.parquet")
    null_bl = {c: int(nb[c].null_count()) for c in BL.FEATS + ["score", "channel"]}
    assert not any(null_bl.values()), f"NULL blocking features on new rows: {null_bl}"
    nb = nb.with_columns(rank=(pl.col("score").rank("ordinal", descending=True).over("s1_id", "source")).cast(pl.Int16))  # temporary; recomputed over the union below
    union_c = pl.concat([old.select(*KEY, "score"), nb.select(*KEY, "score")], how="vertical_relaxed")
    assert union_c.height == old.height + new.height and union_c.unique(KEY).height == union_c.height, "union is not current + rev-k1 exactly"
    log("check 1/2 OK: union = current + new rev-k1 rows, no duplicates, no null blocking features on new rows")
    del union_c
    allf_old = S.base_features()
    # PRODUCTION fold of every fit S1 (GroupKFold on the ORIGINAL fit rows).  S.stage1 rebuilds folds from the fit rows it is given; with new rows the
    # group sizes change and S1 would move between folds (its stage-1 score would then come from a model that trained on it) -> pinned in main().
    cal = pickle.load(open(CACHE / "calib.pkl", "rb"))
    fit0 = allf_old.filter(pl.col("role") == "fit")
    fo = np.empty(fit0.height, np.int8)
    for i, (_, va) in enumerate(GroupKFold(cal["folds"]).split(np.zeros(fit0.height), fit0["is_match"].to_numpy(), fit0["s1_id"].to_numpy())):
        fo[va] = i
    pl.DataFrame({"s1_id": fit0["s1_id"], "fold": fo}).unique("s1_id").write_parquet(OUTU / "fold_map.parquet")
    del fit0
    roles = allf_old.select("s1_id", "role").unique("s1_id").filter(pl.col("role").is_in(["fit", "val"]))
    newr = nb.select(KEY).join(roles, on="s1_id", how="left")
    val_rec = pl.concat([allf_old.filter(pl.col("role") == "val").select("cand_id", "source"),
                         newr.filter(pl.col("role") == "val").select("cand_id", "source")]).unique()
    comp = newr.filter(pl.col("role").is_null()).join(val_rec, on=["cand_id", "source"], how="semi").with_columns(role=pl.lit("comp"))
    newr = pl.concat([newr.filter(pl.col("role").is_not_null()), comp], how="vertical_relaxed")
    g = pl.read_parquet(CACHE / "gt_pairs.parquet").select("s1_id", cand_id="match_id", source="match_source", is_match=pl.lit(True))
    newr = newr.join(g, on=KEY, how="left").with_columns(pl.col("is_match").fill_null(False))     # labels only as the training target
    log(f"new rows by role {dict(newr['role'].value_counts().iter_rows())}")
    newf = featurise_new(newr, allf_old)
    blk = [c for c in BL.FEATS + ["score", "rank", "channel"] if c in newf.columns]
    newf = newf.drop(blk).join(nb.select(*KEY, *blk), on=KEY, how="left").select(allf_old.columns)
    allf_u = pl.concat([allf_old, newf], how="vertical_relaxed")
    del allf_old, newf
    # rank recomputed over the union for ALL rows, owner competition recomputed over the union for ALL rows
    allf_u = allf_u.with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("s1_id", "source").cast(pl.Int16))
    cu = F.competition(pl.concat([old.select(*KEY, "score"), nb.select(*KEY, "score")], how="vertical_relaxed")).select(*KEY, *COMP)
    allf_u = allf_u.drop(COMP).join(cu, on=KEY, how="left")
    # dense features for ALL rows (label-free)
    rmax = rev.filter(pl.col("rrank") == 0).select("cand_id", "source", rmax="rscore")
    allf_u = (allf_u.join(rev.select(*KEY, d_rev="rrank"), on=KEY, how="left").join(rmax, on=["cand_id", "source"], how="left")
              .join(nb.select(*KEY, ch_dense=pl.lit(1.0, pl.Float32)), on=KEY, how="left"))
    allf_u = allf_u.with_columns(dcos(allf_u), pl.col("d_rev").fill_null(99).cast(pl.Float32), pl.col("ch_dense").fill_null(0.0))
    allf_u = allf_u.with_columns(d_rgap=(pl.col("rmax").fill_null(pl.col("dcos")) - pl.col("dcos")).cast(pl.Float32)).drop("rmax")
    isnew = allf_u.select(*KEY).join(nb.select(*KEY, n=pl.lit(True)), on=KEY, how="left")["n"].fill_null(False).to_numpy()
    nn = allf_u.filter(pl.Series(isnew)).select([pl.col(c).is_null().mean() for c in allf_u.columns if allf_u.schema[c].is_numeric()])
    no = allf_u.filter(pl.Series(~isnew)).head(2_000_000).select([pl.col(c).is_null().mean() for c in allf_u.columns if allf_u.schema[c].is_numeric()])
    bad = [(c, round(nn[c][0], 3), round(no[c][0], 3)) for c in nn.columns if nn[c][0] > 0.001 and nn[c][0] > no[c][0] + 0.001]
    log(f"check 2: numeric columns with MORE nulls on new rows than old rows (new share, old share): {bad}")
    allf_u.write_parquet(OUTU / "allf_union.parquet")
    return allf_u, nb.select(KEY)


def worst_split(preds, ref):
    """Held-out (rule chosen on the other half) F0.5 per split-half: mean and worst for each model."""
    _, truth_n, _ = val_truth()
    res = {n: S.scores(p, truth_n) for n, p in preds.items()}
    print(f"\n{'model':30s} held-out F0.5 over 20 split-halves: mean    worst    best")
    for n, r in res.items():
        v = []
        for seed in range(10):
            m = np.random.default_rng(seed).random(len(truth_n)) < 0.5
            for A, B in ((m, ~m), (~m, m)):
                pick = max(r, key=lambda c: r[c][0][A].mean())
                v.append(r[pick][0][B].mean())
        print(f"{n:30s}                                 {np.mean(v):.5f}  {np.min(v):.5f}  {np.max(v):.5f}")


def group_report(pu, pr, new_keys):
    """Union model vs the saved 0.98108 model, split into ORIGINAL candidates and NEWLY ADDED reverse-dense candidates (VAL, production rule)."""
    import json
    _, truth_n, gt = val_truth()
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    kc = (cfg["alpha"], cfg["miss"], cfg["gamma"], cfg["floor"])
    nk = new_keys.with_columns(isnew=pl.lit(True))
    def kept(p):
        return S.choose(S.owner_adjust(p, kc[0]).filter(pl.col("role") == "val"), *kc[1:]).filter("take").join(nk, on=KEY, how="left").with_columns(pl.col("isnew").fill_null(False))
    ku, kr = kept(pu), kept(pr)
    vu = pu.filter(pl.col("role") == "val").join(nk, on=KEY, how="left").with_columns(pl.col("isnew").fill_null(False))
    t_new, t_old = vu.filter(pl.col("is_match") & pl.col("isnew")).height, vu.filter(pl.col("is_match") & ~pl.col("isnew")).height
    print(f"\nVAL true pairs retrievable: original candidates {t_old:,} | only via new reverse-dense candidates {t_new:,}")
    print(f"{'model / candidate group':40s} {'kept':>8s} {'TP':>8s} {'FP':>6s} {'precision':>9s} {'recall of group':>16s}")
    for name, k in (("saved 0.98108 model: original", kr.filter(~pl.col('isnew'))), ("union model: original", ku.filter(~pl.col('isnew'))),
                    ("union model: NEW reverse-dense", ku.filter(pl.col('isnew')))):
        tp = k.filter(pl.col("is_match")).height
        tot = t_new if "NEW" in name else t_old
        print(f"{name:40s} {k.height:8,} {tp:8,} {k.height - tp:6,} {tp / max(k.height, 1):9.4f} {tp / tot:16.4f}")
    vn = vu.filter(pl.col("isnew"))
    q = lambda s: "/".join(f"{s.quantile(x):.3f}" for x in (.1, .25, .5, .75, .9))
    print(f"new candidates in VAL: {vn.height:,} ({vn['is_match'].sum():,} true). p quantiles 10/25/50/75/90 -- true {q(vn.filter(pl.col('is_match'))['p'])} | false {q(vn.filter(~pl.col('is_match'))['p'])}")
    print(f"new true pairs with p>=0.5: {vn.filter(pl.col('is_match') & (pl.col('p') >= .5)).height:,} of {t_new:,}")
    fu, fr = S.per_s1(ku, truth_n)[0], S.per_s1(kr, truth_n)[0]
    ids_new = set(vu.filter(pl.col("is_match") & pl.col("isnew"))["s1_id"].to_list())
    m = np.array([s in ids_new for s in truth_n["s1_id"].to_list()])
    print(f"S1 with >=1 true pair only reachable via the new candidates: {m.sum():,} | F0.5 saved model {fr[m].mean():.4f} -> union {fu[m].mean():.4f} ({fu[m].mean() - fr[m].mean():+.4f})")
    print(f"all other S1: {(~m).sum():,} | F0.5 saved model {fr[~m].mean():.5f} -> union {fu[~m].mean():.5f} ({fu[~m].mean() - fr[~m].mean():+.5f})")


def main():
    OUTU.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUTU
    X.OUT3 = OUTU
    _, truth_n, gt = val_truth()
    n_true, n_s1 = gt.height, truth_n.height
    if "--from-cache" in sys.argv:
        allf_u, new_keys = pl.read_parquet(OUTU / "allf_union.parquet"), pl.read_parquet(OUTU / "new_candidates.parquet")
    else:
        allf_u, new_keys = build_union_features()
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    ref = base_cols + S.NEW + E.NEW2 + ["is_india"] + X.GA + X.GC + X.GD + X4.NEW4
    fmap = dict(pl.read_parquet(OUTU / "fold_map.parquet").iter_rows())

    class ProductionFolds:                       # same S1 -> fold assignment as the production stage-1 models were trained with
        def __init__(self, n):
            self.n = n

        def split(self, X, y, groups):
            f = np.array([fmap[g] for g in groups])
            for i in range(self.n):
                yield np.flatnonzero(f != i), np.flatnonzero(f == i)

    S.GroupKFold = ProductionFolds
    log("=== union candidate set: stage 1 (production folds pinned) -> stage-2 groups -> NUM/QUAL/NEW3/NEW4")
    du, folds, fit_ids = assemble(allf_u, "union")
    del allf_u
    v = du.filter(pl.col("role") == "val")
    log(f"VAL union: pair recall {v.filter(pl.col('is_match')).height / n_true:.4f} | {v.height / n_s1:.2f} candidates per S1 (Pareto sweep: 0.9861 / 11.47)")
    log("training: union + dense features")
    pu, _ = X.train_oof(du, ref + DENSE, folds, fit_ids, "u_union")
    pu.write_parquet(OUTU / "pred_TRAIN_union_rev_k1.parquet")
    fn = sorted(FIN.glob("pred_TRAIN__NEW3_NEW4.parquet"))
    assert fn, "saved 0.98108 predictions (exp/final) not found"
    pr = pl.read_parquet(fn[0])
    preds = {"base (production model)": du.select(*KEY, "role", "is_match", p="p1"), "saved 0.98108 (old set)": pr, "union rev-k1 + dense": pu}
    S.evaluate(preds, ref="saved 0.98108 (old set)")
    worst_split(preds, "saved 0.98108 (old set)")
    group_report(pu, pr, new_keys)
    print("DONE")


if __name__ == "__main__":
    main()
