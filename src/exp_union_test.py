"""TEST submission from the UNION model (exp/union: candidates_TEST  UNION  reverse dense k1, stage 1 -> stage-2 groups -> NUM/QUAL/NEW3/NEW4 + dense)
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
UN = EXP / "union"
DEST = EXP / "union_test"
FRMAP = "--fr-map" in sys.argv
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
    k1 = rev.filter(pl.col("rrank") == 0).select(KEY)
    new = k1.join(old.select(KEY), on=KEY, how="anti")
    n_s1 = pl.read_parquet(N / "test_source1.parquet", columns=["id"]).height
    log(f"TEST current {old.height:,} rows ({old.height / n_s1:.2f}/S1) | rev-dense k1 {k1.height:,} | new {new.height:,} "
        f"(+{new.height / n_s1:.2f}/S1; VAL was +0.42)")
    nb = U.blocking_features(new, old, pool="test")
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
    allf = allf.with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("s1_id", "source").cast(pl.Int16))
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
    cal = pickle.load(open(UN / "u_union_calib.pkl", "rb"))
    missing = [c for c in cal["cols"] if c not in d.columns]
    assert not missing, f"model columns missing on TEST: {missing}"
    boosters = [xgb.Booster(model_file=str(UN / f"u_union_fold{i}.json")) for i in range(cal["folds"])]
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


def main():
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
    null_report(d, cols, new_keys)
    pred = d.select(KEY).with_columns(p=pl.Series(p))
    pred.write_parquet(OUTD / "pred_TEST_union.parquet")
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
