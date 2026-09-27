"""TEST build of the FWD4 model (current + reverse-dense k1 + forward-dense k4; exp_fwd.py) -> exp/fwd4_test/ (RANK0=1: 0-based rank + exp_rankfix model -> exp/fwd4r0_test/).  Derived from exp_union_test.py:
TEST submission from the UNION model (exp/union: candidates_TEST  UNION  reverse dense k1, stage 1 -> stage-2 groups -> NUM/QUAL/NEW3/NEW4 + dense)
-> exp/union_test/.   output_full and candidates_TEST are never touched.

Usage: python src/exp_union_test.py [--from-cache] [--fr-map]
  --from-cache  reuse exp/union_test/allf_union_test.parquet + new_candidates.parquet (steps 1-5)
  --fr-map      France rows: French name-qualifier tokens mapped to their English equivalents (exp_frmap.FRMAP, as in exp/stack2fr_test)
                before the name log-odds lookup -> writes exp/union_frmap_test/ instead

Mirrors exp_union.build_union_features + exp_final.assemble on the test pool, row for row:
  1 reverse dense k1 on TEST (exp/dense/rev_test.parquet; built with exp_dense.build("test") if missing -- needs the GPU) -> new rows = k1 not in candidates_TEST
  2 real blocking features for the new rows (exp_union.blocking_features, pool="test": same TF-IDF vectorizers, same re-ranker), null check
  3 pair features for the new rows (features.pair_features, test pool, IDF over test S1 exactly like features.py --split TEST)
  4 rank recomputed over the union (1-based, as exp_union.py line 148 did for training), owner competition recomputed over the union
  5 dense features dcos / d_rev / d_rgap / ch_dense (label-free)
  6 p1 = production stage-1 fold mean + isotonic (every TEST row is "non-fit", like VAL in training)
  7 stage-2 groups (pool="test"), number-set, name-qualifier / address-word / suffix-pair log-odds with FULL-FIT tables rebuilt from the union
    FIT rows (exp/union/allf_union.parquet + fold_map.parquet; identical to what VAL rows saw in training; unseen token/pair -> 0), compact/leet,
    house number, ambiguity, is_india
  8 score with exp/union/u_union_fold{i}.json + u_union_calib.pkl.  France gets is_india=0 (only the is_india model exists for the union).
  9 production decision rule (cache/decide.json) -> matching_results.tsv + candidate_pairs.tsv (candidate_pairs now includes the new rows)
Printed: new-row counts, feature null shares new vs old rows, pred/S1 and empty share per country vs the submitted / stack2 / stack2fr files.
Validate afterwards: python validate_submission.py --matching exp/union_test/matching_results.tsv --candidate exp/union_test/candidate_pairs.tsv --test-dir <data> --check-ids
"""
import json
import pickle
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

import block as BL
import exp_stack as S
import exp_stack2 as E
import exp_stack3 as X
import exp_stack4 as X4
import exp_union as U
import features as F
from common import CACHE, OUT, log
from decide import apply
from predict import load_models

N = CACHE / "norm"
EXP = CACHE.parent / "exp"
RANK0 = os.environ.get("RANK0") == "1"                        # RANK0=1: rank 0-based (as the production stage-1 models were trained) + the exp_rankfix stage-2 model
UN = EXP / "union_fwd4_ctl1"                                   # FROZEN production model = FWD4 control run #1
MD = EXP / "union_fwd4_ctl1"                                   # stage-2 model directory (frozen, never modified)
TAG = "r_rank0" if RANK0 else "u_fwd4"
DEST = EXP / ("fwd4r0_test" if RANK0 else "fwd4_test")
FRMAP = False
OUTD = EXP / "union_frmap_test" if FRMAP else DEST       # the feature cache (DEST) is shared by both variants
KEY = S.KEY
CH = 4_000_000


def rev_test():
    f = U.D / "rev_test.parquet"
    if not f.exists():
        log("rev_test.parquet missing -> exp_dense.build('test') (encodes test S1/S2/S3 with e5, GPU)")
        import exp_dense
        exp_dense.build("test")
    return pl.read_parquet(f)


def pair_rows(new, cols, schema):
    """Base pair features for new TEST rows, same code path as features.py --split TEST (IDF over test S1 core names)."""
    s1 = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country", *F.TXT]).rename(
        lambda x: x if x in ("id", "country") else x + "_1").rename({"id": "s1_id"})
    cand = pl.concat([pl.read_parquet(N / f"test_source{s}.parquet", columns=["id", "name_raw", *F.TXT])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id", "name_raw": "name_raw_c"})
    toks = s1.select(pl.col("name_core_1").str.split(" ").list.unique().alias("tok")).explode("tok")
    idf = toks.group_by("tok").len().with_columns(idf=(len(s1) / pl.col("len")).log().cast(pl.Float32)).select("tok", "idf")
    out = []
    for i in range(0, new.height, F.CHUNK):
        d = new.slice(i, F.CHUNK).join(s1, on="s1_id", how="left").join(cand, on=["cand_id", "source"], how="left")
        d = F.pair_features(d, idf)
        out.append(d.drop([x for x in d.columns if x.endswith("_1") or x in F.TXT or x == "name_raw_c"]))
    f = pl.concat(out)
    for col in cols:
        if col not in f.columns:
            f = f.with_columns(pl.lit(None).cast(schema[col]).alias(col))
    return f.select(cols)


def build_union_test():
    old = pl.read_parquet(CACHE / "candidates_TEST.parquet")
    rev = rev_test()
    rev = rev.with_columns([pl.col(c).cast(old.schema[c]) for c in KEY])
    fwd = pl.read_parquet(U.D / "fwd_test.parquet")
    fwd4 = fwd.filter(pl.col("drank") < 4).select([pl.col(c).cast(old.schema[c]) for c in KEY])
    k1 = pl.concat([rev.filter(pl.col("rrank") == 0).select(KEY), fwd4]).unique()      # rev k1 UNION fwd k4 (as exp_fwd.py)
    new = k1.join(old.select(KEY), on=KEY, how="anti")
    n_s1 = pl.read_parquet(N / "test_source1.parquet", columns=["id"]).height
    log(f"TEST current {old.height:,} rows ({old.height / n_s1:.2f}/S1) | rev-dense k1 {k1.height:,} | new {new.height:,} "
        f"(+{new.height / n_s1:.2f}/S1; VAL was +0.42)")
    nb = U.blocking_features(new, old, pool="test")
    nb = nb.with_columns(rank=(pl.col("score").rank("ordinal", descending=True).over("s1_id", "source")).cast(pl.Int16))  # temporary; recomputed over the union below (exactly as exp_union.py)
    null_bl = {c: int(nb[c].null_count()) for c in BL.FEATS + ["score", "channel"]}
    assert not any(null_bl.values()), f"NULL blocking features on new rows: {null_bl}"
    nb.select(KEY).write_parquet(DEST / "new_candidates.parquet")

    ft = pl.read_parquet(CACHE / "features_TEST.parquet")
    assert ft.height == old.height, "features_TEST does not cover candidates_TEST row for row"
    ft = ft.with_columns(role=pl.lit("test"), is_match=pl.lit(False))
    cols, schema = ft.columns, ft.schema
    newf = pair_rows(nb.select(KEY), cols, schema).with_columns(role=pl.lit("test"), is_match=pl.lit(False))
    blk = [c for c in BL.FEATS + ["score", "rank", "channel"] if c in cols]
    newf = newf.drop(blk).join(nb.select(*KEY, *[c for c in blk if c in nb.columns]), on=KEY, how="left").select(cols)
    allf = pl.concat([ft, newf], how="vertical_relaxed")
    del ft, newf
    assert allf.unique(KEY).height == allf.height == old.height + new.height, "union is not candidates_TEST + new rev-k1 rows exactly"
    # exactly as exp_union.py (training): rank over the union (1-based), owner competition over the union
    allf = allf.with_columns(rank=(pl.col("score").rank("ordinal", descending=True).over("s1_id", "source") - (1 if RANK0 else 0)).cast(pl.Int16))
    cu = F.competition(pl.concat([old.select(*KEY, "score"), nb.select(*KEY, "score")], how="vertical_relaxed")).select(*KEY, *U.COMP)
    allf = allf.drop(U.COMP).join(cu, on=KEY, how="left")
    rmax = rev.filter(pl.col("rrank") == 0).select("cand_id", "source", rmax="rscore")
    allf = (allf.join(rev.select(*KEY, d_rev="rrank"), on=KEY, how="left").join(rmax, on=["cand_id", "source"], how="left")
            .join(nb.select(*KEY, ch_dense=pl.lit(1.0, pl.Float32)), on=KEY, how="left"))
    allf = allf.with_columns(U.dcos(allf, pool="test"), pl.col("d_rev").fill_null(99).cast(pl.Float32), pl.col("ch_dense").fill_null(0.0))
    allf = allf.with_columns(d_rgap=(pl.col("rmax").fill_null(pl.col("dcos")) - pl.col("dcos")).cast(pl.Float32)).drop("rmax")
    allf.write_parquet(DEST / "allf_union_test.parquet")
    return allf, nb.select(KEY)


def stage1_test(allf):
    """p1 = production fold mean + isotonic (no TEST row belongs to a fit fold)."""
    cal = pickle.load(open(CACHE / "calib.pkl", "rb"))
    models = load_models(cal)
    p1 = np.empty(allf.height, np.float32)
    for i in range(0, allf.height, CH):
        x = allf.slice(i, CH).select([pl.col(k).cast(pl.Float32) for k in cal["cols"]]).to_numpy()
        p1[i:i + CH] = cal["iso"].predict(np.mean([m(x) for m in models], axis=0))
    return allf.with_columns(p1=pl.Series(p1))


def fit_tables():
    """FULL-FIT log-odds tables of the union training run, rebuilt from its FIT rows (same p0 / smoothing / token definitions as exp_final.assemble)."""
    fmap = pl.read_parquet(UN / "fold_map.parquet").select("s1_id")
    fit = pl.read_parquet(UN / "allf_union.parquet", columns=[*KEY, "role", "is_match"]).filter(pl.col("role") == "fit")
    fit = fit.join(fmap.with_columns(pl.col("s1_id").cast(fit.schema["s1_id"])), on="s1_id", how="semi").select(*KEY, "is_match")
    p0 = fit["is_match"].mean()
    log(f"union FIT rows {fit.height:,}, p0 {p0:.5f}")

    def vocab(t):
        out = {}
        for col in ("X", "M"):
            e = t.select(*KEY, tok=col).explode("tok").drop_nulls("tok").join(fit, on=KEY, how="left")
            tot = e.group_by("tok").agg(pos=pl.col("is_match").sum(), neg=(~pl.col("is_match")).sum())
            out[col] = E.log_odds(tot, p0).select("tok", "lo")
        return out
    parts_n, parts_a = [], []
    for i in range(0, fit.height, CH):
        sl = fit.slice(i, CH).select(KEY)
        parts_n.append(E.tok_num(sl, "train").select(*KEY, "X", "M"))
        parts_a.append(X.addr_words(sl, "train").select(*KEY, "X", "M"))
    v_name, v_addr = vocab(pl.concat(parts_n)), vocab(pl.concat(parts_a))
    del parts_n, parts_a
    s1, cand = X.tables("train")
    pk = (fit.select(KEY).join(cand.select("cand_id", "source", "suf2"), on=["cand_id", "source"], how="left")
          .join(s1.select("s1_id", "suf1"), on="s1_id", how="left")
          .select(*KEY, pk=pl.col("suf1").fill_null("") + "|" + pl.col("suf2").fill_null("")).join(fit, on=KEY, how="left"))
    suf = pk.group_by("pk").agg(pos=pl.col("is_match").sum(), neg=(~pl.col("is_match")).sum())
    return v_name, v_addr, suf, p0


def lo_feats(t, vocab, pre=""):
    """min/sum of token log-odds per row; unseen tokens -> 0 (= what VAL rows got for tokens absent from the FIT rows)."""
    out = t.select(KEY)
    for col, p in (("X", "x"), ("M", "m")):
        e = t.select(*KEY, tok=col).explode("tok").drop_nulls("tok").join(vocab[col], on="tok", how="left").with_columns(pl.col("lo").fill_null(0.0))
        out = out.join(e.group_by(KEY).agg(**{f"{pre}{p}_lo_min": pl.col("lo").min(), f"{pre}{p}_lo_sum": pl.col("lo").sum()}), on=KEY, how="left")
    return out.with_columns([pl.col(f"{pre}{c}").fill_null(0.0) for c in ("x_lo_min", "x_lo_sum", "m_lo_min", "m_lo_sum")])


def assemble_test(allf):
    """exp_final.assemble on the test pool with the full-fit tables."""
    d = S.stage2_features(stage1_test(allf), pool="test")
    v_name, v_addr, suf, p0 = fit_tables()
    if FRMAP:
        from exp_frmap import map_fr
        cty = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
        cty = cty.with_columns(pl.col("s1_id").cast(d.schema["s1_id"]))
    parts = []
    for i in range(0, d.height, CH):
        sl = d.slice(i, CH).select(KEY)
        t = E.tok_num(sl, "test")
        aw = X.addr_words(sl, "test")
        tq = map_fr(t.select(*KEY, "X", "M"), cty) if FRMAP else t
        parts.append(t.drop("X", "M").join(lo_feats(tq, v_name), on=KEY, how="left")
                     .join(aw.drop("X", "M"), on=KEY, how="left").join(lo_feats(aw, v_addr, "ad_"), on=KEY, how="left")
                     .join(X.compact_features(sl, "test"), on=KEY, how="left"))
        log(f"  NUM/QUAL/ADDR/compact {min(i + CH, d.height):,}/{d.height:,}")
    d = d.join(pl.concat(parts), on=KEY, how="left")
    del parts
    s1, cand = X.tables("test")
    sp = (d.select(KEY).join(cand.select("cand_id", "source", "suf2"), on=["cand_id", "source"], how="left")
          .join(s1.select("s1_id", "suf1"), on="s1_id", how="left")
          .select(*KEY, pk=pl.col("suf1").fill_null("") + "|" + pl.col("suf2").fill_null(""))
          .join(suf, on="pk", how="left").with_columns(pl.col("pos").fill_null(0), pl.col("neg").fill_null(0)))
    sp = E.log_odds(sp, p0).select(*KEY, suf_pair_lo=pl.col("lo"), suf_pair_n=(pl.col("pos") + pl.col("neg")).cast(pl.Int32))
    d = d.join(sp, on=KEY, how="left").join(X4.ambiguity(d, "test"), on=KEY, how="left")
    return d.with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32))


def score(d):
    cal = pickle.load(open(MD / f"{TAG}_calib.pkl", "rb"))
    missing = [c for c in cal["cols"] if c not in d.columns]
    assert not missing, f"model columns missing on TEST: {missing}"
    boosters = [xgb.Booster(model_file=str(MD / f"{TAG}_fold{i}.json")) for i in range(cal["folds"])]
    p = np.empty(d.height, np.float32)
    for i in range(0, d.height, CH):
        x = d.slice(i, CH).select([pl.col(k).cast(pl.Float32) for k in cal["cols"]]).to_numpy()
        p[i:i + CH] = cal["iso"].predict(np.mean([b.predict(xgb.DMatrix(x, feature_names=cal["cols"])) for b in boosters], axis=0))
    return p, cal["cols"]


def null_report(d, cols, new_keys):
    isnew = d.select(KEY).join(new_keys.with_columns(n=pl.lit(True)), on=KEY, how="left")["n"].fill_null(False)
    num = [c for c in cols if d.schema[c].is_numeric()]
    nn = d.filter(isnew).select([pl.col(c).is_null().mean() for c in num])
    no = d.filter(~isnew).select([pl.col(c).is_null().mean() for c in num])
    bad = [(c, round(nn[c][0], 3), round(no[c][0], 3)) for c in num if abs(nn[c][0] - no[c][0]) > 0.01]
    log(f"model columns whose null share differs new vs old rows by > 1pt (new, old): {bad}")


def parity(d, cols, pred):
    """Train/test parity checks (read-only): schema/dtype vs the training feature matrix, null shares, rank convention, decision rule, per-country candidates / predictions / scores."""
    ref = pl.scan_parquet(EXP / "fwd4_ref" / "du.parquet")            # assembled training matrix of the union model (same schema / feature code)
    rs = ref.collect_schema()
    miss = [c for c in cols if c not in rs.names()]
    dt = [(c, str(rs[c]), str(d.schema[c])) for c in cols if c in rs.names() and str(rs[c]) != str(d.schema[c])]
    print(f"PARITY schema: model columns {len(cols)}; missing in training matrix {miss}; dtype differences train->test {dt}")
    v = ref.filter(pl.col("role") == "val").select([pl.col(c).is_null().mean().alias(c) for c in cols if c in rs.names()]).collect()
    t = d.select([pl.col(c).is_null().mean().alias(c) for c in cols if c in rs.names()])
    bad = [(c, round(v[c][0], 3), round(t[c][0], 3)) for c in v.columns if abs(v[c][0] - t[c][0]) > 0.01]
    print(f"PARITY null share VAL vs TEST differing > 1pt (col, val, test): {bad}")
    print(f"PARITY rank convention: RANK0={RANK0}; TEST 'rank' min/median/max {d['rank'].min()}/{d['rank'].median()}/{d['rank'].max()} (expected min {0 if RANK0 else 1})")
    cfg = json.load(open(CACHE / "decide.json"))
    print(f"PARITY decision rule: cache/decide.json cfg {cfg['cfg']} (same file the validation runs read; apply() = decide.apply)")
    ctry = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    kept = apply(pred, cfg["cfg"]).join(ctry, on="s1_id")
    a = pred.join(ctry, on="s1_id")
    bestp = a.group_by("s1_id", "country").agg(b=pl.col("p").max(), n=pl.len())
    for c in ("France", "India", "US"):
        n = ctry.filter(pl.col("country") == c).height
        b = bestp.filter(pl.col("country") == c)
        k = kept.filter(pl.col("country") == c)
        acc = k["p"].quantile(0.5) if "p" in k.columns else float("nan")
        print(f"PARITY {c:7s} S1 {n:,} cands/S1 {b['n'].sum() / n:.2f} (p90 {b['n'].quantile(.9):.0f}, max {b['n'].max()}) | pred/S1 {k.height / n:.3f} empty {1 - k['s1_id'].n_unique() / n:.2%} | "
              f"best p p10/p25/p50 {b['b'].quantile(.1):.3f}/{b['b'].quantile(.25):.3f}/{b['b'].quantile(.5):.3f}  >=.9 {(b['b'] >= .9).mean():.3f}  <.15 {(b['b'] < .15).mean():.3f} | accepted p median {acc:.3f}")


def density():
    """Train -> test candidate-pool density: S2+S3 records per S1 by country (label-free counts of the normalised source tables)."""
    print("DENSITY (S2+S3 records per S1)")
    for pool in ("train", "test"):
        s1 = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["country"])["country"].value_counts()
        rec = pl.concat([pl.read_parquet(N / f"{pool}_source{k}.parquet", columns=["country"]) for k in (2, 3)])["country"].value_counts()
        a, b = dict(s1.iter_rows()), dict(rec.iter_rows())
        allr = sum(b.values()) / sum(a.values())
        print(f"  {pool:5s} all {allr:.2f} | " + " | ".join(f"{c} {b.get(c, 0) / a[c]:.2f} (S1 {a[c]:,})" for c in sorted(a)))


def diag(d, p, label, ctry):
    """Per-country test/validation diagnostics on one candidate table: d needs KEY, rank, o_rank, o_n5, o_margin; p aligned with d rows."""
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    pred = d.select(KEY).with_columns(p=pl.Series(p) if not isinstance(p, pl.Series) else p)
    _dbg = lambda n, x: print(f"DIAGDBG [{label}] {n}: {x.columns} | {x.height:,} rows", flush=True)   # diagnostic-only: schemas of every table entering the joins
    print(f"DIAGDBG [{label}] KEY={KEY}", flush=True)
    _dbg("d", d); _dbg("pred", pred); _dbg("ctry", ctry)
    ap_ = apply(pred, cfg); _dbg("apply(pred,cfg)", ap_)
    k1_ = ap_.select(*KEY, "p"); _dbg("apply(...).select(*KEY,'p')", k1_)   # FIX: select(KEY, "p") treated the list as a literal column ("literal"), dropping s1_id
    dd_ = d.select(*KEY, "rank", "o_rank", "o_n5", "o_margin"); _dbg("d.select(KEY,rank,o_*)", dd_)
    k2_ = k1_.join(dd_, on=KEY, how="left"); _dbg("k1.join(dd,on=KEY)", k2_)
    _dbg("ctry (right side of the s1_id join)", ctry)
    kept = k2_.join(ctry, on="s1_id")
    a = d.select(*KEY, "rank", "o_rank").with_columns(p=pred["p"]); _dbg("a (before ctry join)", a)
    a = a.join(ctry, on="s1_id")
    r0 = 0 if RANK0 else 1
    print(f"DIAG [{label}]  (rank base {r0}; high-confidence = kept with p>=0.9 / 0.95)")
    for c in ("France", "India", "US"):
        n = ctry.filter(pl.col("country") == c).height
        if n == 0:
            continue
        ac, kc = a.filter(pl.col("country") == c), kept.filter(pl.col("country") == c)
        b = ac.group_by("s1_id").agg(b=pl.col("p").max(), n=pl.len())
        rk = kc["rank"]
        print(f"  {c:7s} S1 {n:>9,} | cands/S1 {ac.height / n:5.2f} | pred/S1 {kc.height / n:.3f} empty {1 - kc['s1_id'].n_unique() / n:6.2%} | all-rows p mean {ac['p'].mean():.4f} >=.5 {(ac['p'] >= .5).mean():.4f} >=.9 {(ac['p'] >= .9).mean():.4f} | "
              f"best p p10/p25/p50 {b['b'].quantile(.1):.3f}/{b['b'].quantile(.25):.3f}/{b['b'].quantile(.5):.3f} <.15 {(b['b'] < .15).mean():.3f} | "
              f"kept p>=.9 {(kc['p'] >= .9).sum() / n:.3f}/S1 ({(kc['p'] >= .9).mean():.3f} of kept), >=.95 {(kc['p'] >= .95).sum() / n:.3f}/S1 | "
              f"kept rank r0/r0+1/r0+2/>= {(rk == r0).mean():.3f}/{(rk == r0 + 1).mean():.3f}/{(rk == r0 + 2).mean():.3f}/{(rk > r0 + 2).mean():.3f} | "
              f"owner: rows o_rank==1 {(ac['o_rank'] == 1).mean():.3f}, kept with o_n5>=2 {(kc['o_n5'] >= 2).mean():.4f}, kept mean o_margin {kc['o_margin'].mean():.3f}")


def val_reference():
    """Same diagnostics on the VAL rows of the E union model (exp/rank2/du.parquet + exp/rank2/pred_CTRL.parquet; same feature definitions, rank 1-based) for comparison."""
    du = pl.scan_parquet(EXP / "fwd4_ref" / "du.parquet").filter(pl.col("role") == "val").select(*KEY, "country", "rank", "o_rank", "o_n5", "o_margin").collect()
    pr = pl.read_parquet(EXP / "union_fwd4_ctl1" / "pred_TRAIN_union_fwd4.parquet", columns=[*KEY, "p"])
    j = du.join(pr, on=KEY, how="left")
    ctry = du.select("s1_id", "country").unique("s1_id")
    global RANK0
    keep, RANK0 = RANK0, False
    diag(j.drop("p"), j["p"], "VAL, frozen FWD4 ctl1 (rank 1-based)", ctry)
    RANK0 = keep


def main():
    import hashlib
    for f in sorted(MD.glob("u_fwd4_*")):
        print("MODEL", f.name, hashlib.sha256(f.read_bytes()).hexdigest()[:16], flush=True)
    print("MODEL DIR", MD, "| UN", UN, flush=True)
    assert not RANK0 and MD.name == "union_fwd4_ctl1"
    DEST.mkdir(parents=True, exist_ok=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    full = OUT.parent / "output_full"
    before = {n: (full / n).read_bytes() for n in ("matching_results.tsv", "candidate_pairs.tsv")}
    if "--from-cache" in sys.argv:
        allf, new_keys = pl.read_parquet(DEST / "allf_union_test.parquet"), pl.read_parquet(DEST / "new_candidates.parquet")
    else:
        allf, new_keys = build_union_test()
    d = assemble_test(allf)
    del allf
    p, cols = score(d)
    oldp = OUTD / "pred_TEST_union.parquet"                       # scores of the earlier (complete) run: must be reproduced exactly by the frozen model
    if oldp.exists():
        o = pl.read_parquet(oldp)
        same = o.height == d.height and all((o[c].to_numpy() == d[c].to_numpy()).all() for c in KEY)
        dd = np.abs(o["p"].to_numpy().astype(np.float64) - p.astype(np.float64)) if same else None
        print(f"REUSE CHECK vs earlier pred_TEST_union.parquet: same rows/order {same} | max|dp| {None if dd is None else dd.max():} | n differing {None if dd is None else int((dd > 0).sum())}", flush=True)
        assert same and dd.max() < 1e-6, "re-scored TEST probabilities differ from the earlier complete run"
    null_report(d, cols, new_keys)
    pred = d.select(KEY).with_columns(p=pl.Series(p))
    pred.write_parquet(OUTD / "pred_TEST_union.parquet")
    parity(d, cols, pred)
    ctry0 = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    density()
    diag(d, pred["p"], "TEST " + TAG, ctry0)
    val_reference()
    del d
    log("scored")

    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    ctry = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    ref = {"submitted": CACHE / "pred_TEST.parquet", "stack2": EXP / "stack2_test" / "pred_TEST_stack2.parquet",
           "stack2fr": EXP / "stack2fr_test" / "pred_TEST_stack2.parquet",
           "union": DEST / "pred_TEST_union.parquet"}
    runs = [(k, pl.read_parquet(f, columns=[*KEY, "p"])) for k, f in ref.items() if f.exists() and not (k == "union" and not FRMAP)]
    runs.append(("union+frmap" if FRMAP else "union (new)", pred))
    for tag, pr in runs:
        k = apply(pr, cfg).join(ctry, on="s1_id")
        for c in ("France", "India", "US"):
            n = ctry.filter(pl.col("country") == c).height
            kc = k.filter(pl.col("country") == c)
            print(f"  {tag:14s} {c:7s} pred/S1 {kc.height / n:.3f}  empty {1 - kc['s1_id'].n_unique() / n:.2%}")
    kept = apply(pred, cfg)
    kn = kept.join(new_keys, on=KEY, how="semi").join(ctry, on="s1_id")
    print("kept rows from NEW reverse-dense candidates per country:", dict(kn["country"].value_counts().iter_rows()),
          f"(VAL: 1,591 of 33,172 S1 = {1591 / 33172:.3f}/S1)")

    ids = {s: pl.read_parquet(CACHE / f"test_source{s}.parquet", columns=["id", "entity_id"]) for s in (1, 2, 3)}
    ent = pl.concat([ids[s].with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    s1 = ids[1].rename({"id": "s1_id", "entity_id": "source1_entity_id"})
    for rows, col, name in ((kept, "matched_entity_ids", "matching_results.tsv"), (pred, "candidate_entity_ids", "candidate_pairs.tsv")):
        lists = (rows.select(KEY).unique().join(ent, on=["cand_id", "source"])
                 .sort("entity_id").group_by("s1_id").agg(pl.col("entity_id").str.join(",").alias(col)))
        out = s1.join(lists, on="s1_id", how="left").with_columns(pl.col(col).fill_null("")).sort("s1_id").select("source1_entity_id", col)
        out.write_csv(OUTD / name, separator="\t", quote_style="never")
        print(f"wrote {OUTD / name}: {len(out):,} rows, non-empty {(out[col] != '').mean():.3f}")
    unchanged = all((full / n).read_bytes() == b for n, b in before.items())
    print("OUTPUT_FULL_UNCHANGED" if unchanged else "OUTPUT_FULL_CHANGED")
    print("DONE")


if __name__ == "__main__":
    main()
