"""Experiment: France location-parsing fix -> re-featurise French TEST candidates -> predict -> new submission folder.

Bug (exp_fr_feat): city/state are guessed from component frequency in SOURCE 1 only. French S1 addresses carry the
region ("hauts de france"); French S2/S3 carry the department ("nord", "gironde", ...) or no region, so
"roubaix, nord" parses as state=roubaix, city=nord -> state conflict on 62% of confident French pairs
(US/India true pairs: 1.5%). The model learned "state conflict => not a match", so French pairs are pushed down.

Fix (France rows of TEST S2/S3 only; US/India and all training data untouched):
  regions  = state values of French S1;  city->region from French S1
  department->region learned from S2/S3 records that name a known city next to an unknown component (>=200 votes, >=90%)
  state = region, else mapped department, else None (as for US/India records without a state);  city = known S1 city
Training features are unchanged, so the saved models are reused (retraining would reproduce the same model).
Checks: re-featurising with the ORIGINAL norm must reproduce pred_TEST exactly for French rows.

Usage: python src/exp_fr_v2.py      -> cache/pred_TEST_frv2.parquet, output_france_model_v2/*.tsv
"""
import json
import pickle

import numpy as np
import polars as pl

import features as F
from common import CACHE, OUT, log
from decide import apply
from predict import load_models

N = CACHE / "norm"
DEST = OUT.parent / "output_france_model_v2"
KEY = ["s1_id", "cand_id", "source"]


def parts(df):
    """Non-numeric address components of each record, in order."""
    return (df.select("id", c=pl.col("addr_norm").str.split(", ")).explode("c")
              .filter(pl.col("c").is_not_null() & (pl.col("c") != "") & ~pl.col("c").str.contains(r"\d"))
              .with_row_index("pos"))


def learn_loc(s1fr, candfr):
    regions = set(s1fr["state_guess"].drop_nulls().unique().to_list())
    c2r = (s1fr.drop_nulls(["city_guess", "state_guess"]).group_by("city_guess", "state_guess").len()
               .sort("len", descending=True).group_by("city_guess").first())
    c2r = c2r.filter(pl.col("len") >= 20).select(pl.col("city_guess").alias("c"), region="state_guess")
    p = parts(candfr)
    city = p.join(c2r, on="c").sort("pos").group_by("id").first().select("id", "region")
    other = p.filter(~pl.col("c").is_in(list(regions)) & ~pl.col("c").is_in(c2r["c"].to_list()))
    votes = other.join(city, on="id").group_by("c", "region").len()
    tot = votes.group_by("c").agg(n=pl.col("len").sum())
    dept = (votes.join(tot, on="c").sort("len", descending=True).group_by("c").first()
                 .filter((pl.col("n") >= 200) & (pl.col("len") / pl.col("n") >= 0.9)).select("c", dept_region="region"))
    log(f"France: {len(regions)} regions, {c2r.height} S1 cities, departments learned: {dict(dept.iter_rows())}")
    return regions, c2r, dept


def fix_loc(df, regions, c2r, dept):
    """New city_guess/state_guess for French candidate records."""
    p = parts(df).join(c2r.select("c", is_city=pl.lit(True)), on="c", how="left").join(dept, on="c", how="left")
    p = p.with_columns(is_region=pl.col("c").is_in(list(regions)), is_city=pl.col("is_city").fill_null(False))
    per = p.sort("pos").group_by("id").agg(
        reg=pl.col("c").filter(pl.col("is_region")).first(),
        dreg=pl.col("dept_region").drop_nulls().first(),
        city=pl.col("c").filter(pl.col("is_city")).first(),
        other=pl.col("c").filter(~pl.col("is_region") & ~pl.col("is_city") & pl.col("dept_region").is_null()).first())
    per = per.select("id", state_new=pl.coalesce("reg", "dreg"), city_new=pl.coalesce("city", "other"))
    return df.join(per, on="id", how="left").with_columns(
        state_guess=pl.col("state_new"), city_guess=pl.col("city_new")).drop("state_new", "city_new")


def featurise(c, s1, cand, idf):
    out = []
    for i in range(0, len(c), F.CHUNK):
        d = c.slice(i, F.CHUNK).join(s1, on="s1_id", how="left").join(cand, on=["cand_id", "source"], how="left")
        d = F.pair_features(d, idf)
        out.append(d.drop([x for x in d.columns if x.endswith("_1") or x in F.TXT or x == "name_raw_c"]))
    return pl.concat(out)


def predict(f, cal, models):
    x = f.select([pl.col(k).cast(pl.Float32) for k in cal["cols"]]).to_numpy()
    return f.select(KEY).with_columns(p=pl.Series(cal["iso"].predict(np.mean([m(x) for m in models], axis=0)).astype(np.float32)))


def main():
    ctry = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    fr = ctry.filter(pl.col("country") == "France").select("s1_id")
    s1 = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country", *F.TXT])
    toks = s1.select(pl.col("name_core").str.split(" ").list.unique().alias("tok")).explode("tok")
    idf = toks.group_by("tok").len().with_columns(idf=(len(s1) / pl.col("len")).log().cast(pl.Float32)).select("tok", "idf")
    s1r = s1.rename(lambda x: x if x in ("id", "country") else x + "_1").rename({"id": "s1_id"})
    cand = pl.concat([pl.read_parquet(N / f"test_source{s}.parquet", columns=["id", "country", "name_raw", *F.TXT])
                      .filter(pl.col("country") == "France").with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)])
    # ids are row indices per source -> make them unique across S2/S3 for learning
    regions, c2r, dept = learn_loc(s1.filter(pl.col("country") == "France"),
                                   cand.with_columns(id=pl.col("id").cast(pl.Int64) * 10 + pl.col("source")))
    candfix = pl.concat([fix_loc(cand.filter(pl.col("source") == s), regions, c2r, dept) for s in (2, 3)])
    rn = lambda d: d.drop("country").rename({"id": "cand_id", "name_raw": "name_raw_c"})

    c = F.competition(pl.read_parquet(CACHE / "candidates_TEST.parquet").join(fr, on="s1_id", how="semi"))
    log(f"French candidate pairs: {len(c):,}")
    cal = pickle.load(open(CACHE / "calib.pkl", "rb"))
    models = load_models(cal)
    old = pl.read_parquet(CACHE / "pred_TEST.parquet").join(fr, on="s1_id", how="semi")

    f0 = featurise(c, s1r, rn(cand), idf)                      # original norm -> must reproduce pred_TEST
    p0 = predict(f0, cal, models).join(old, on=KEY, suffix="_old")
    diff = (p0["p"] - p0["p_old"]).abs()
    bad = (diff > 1e-4).mean()
    tie = c.select((pl.col("score").count() - pl.col("score").n_unique()).over("s1_id", "source").alias("t"))["t"] > 0
    log(f"reproduction check on {len(p0):,} French pairs: rows differing {bad:.4%}, max {diff.max():.3f}; "
        f"rows in an (S1,source) group with tied re-ranker scores: {tie.mean():.2%}")
    # differences can only come from arbitrary tie order in competition() (gap_next / owner_rank); OLD vs NEW below
    # share the same competition table, so the comparison is unaffected
    assert bad < 0.01, bad
    f1 = featurise(c, s1r, rn(candfix), idf)                   # fixed location parsing
    p1 = predict(f1, cal, models)

    for tag, f in (("OLD", f0), ("NEW", f1)):
        ff = f.join((p0 if tag == "OLD" else p1).select(*KEY, "p"), on=KEY)
        conf = ff.filter(pl.col("p") >= 0.5)
        print(f"{tag}: France pairs p>=0.5 {conf.height:,} | state_state dist {dict(conf['state_state'].value_counts(normalize=True).sort('state_state').iter_rows())} "
              f"| city_ratio {conf['city_ratio'].mean():.3f} | all-pairs state conflict {(ff['state_state'] == 1).mean():.3f}")

    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    rest = pl.read_parquet(CACHE / "pred_TEST.parquet").join(fr, on="s1_id", how="anti")
    new = pl.concat([rest.select(*KEY, "p"), p1.select(*KEY, "p")])
    new.write_parquet(CACHE / "pred_TEST_frv2.parquet")
    n_fr = fr.height
    for tag, pred in (("SUBMITTED", pl.concat([rest.select(*KEY, "p"), old.select(*KEY, "p")])),
                      ("OLD-recomputed", pl.concat([rest.select(*KEY, "p"), p0.select(*KEY, "p")])), ("NEW", new)):
        k = apply(pred, cfg).join(ctry, on="s1_id")
        for cc in ("France", "India", "US"):
            kc = k.filter(pl.col("country") == cc)
            n = ctry.filter(pl.col("country") == cc).height
            print(f"  {tag} {cc:7s} pred/S1 {kc.height / n:.3f}  empty {1 - kc['s1_id'].n_unique() / n:.2%}")

    # the low-scored French top candidates from exp_france/exp_fr_feat: how many move above 0.5?
    top_old = p0.sort("p_old", descending=True).group_by("s1_id", maintain_order=True).first()
    low = top_old.filter(pl.col("p_old") < 0.15).select(*KEY, "p_old").join(p1, on=KEY)
    print(f"French S1 whose best candidate had p<0.15 before: {low.height:,}; now p>=0.5: {(low['p'] >= 0.5).mean():.1%}, "
          f"p>=0.15: {(low['p'] >= 0.15).mean():.1%}")

    # outputs
    kept = apply(new, cfg)
    ids = {s: pl.read_parquet(CACHE / f"test_source{s}.parquet", columns=["id", "entity_id"]) for s in (1, 2, 3)}
    ent = pl.concat([ids[s].with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    s1ids = ids[1].rename({"id": "s1_id", "entity_id": "source1_entity_id"})
    DEST.mkdir(parents=True, exist_ok=True)
    for rws, col, name in ((kept, "matched_entity_ids", "matching_results.tsv"),
                           (new, "candidate_entity_ids", "candidate_pairs.tsv")):
        lists = (rws.select(KEY).unique().join(ent, on=["cand_id", "source"])
                 .sort("entity_id").group_by("s1_id").agg(pl.col("entity_id").str.join(",").alias(col)))
        out = s1ids.join(lists, on="s1_id", how="left").with_columns(pl.col(col).fill_null("")) \
                   .sort("s1_id").select("source1_entity_id", col)
        out.write_csv(DEST / name, separator="\t", quote_style="never")
        print(f"wrote {DEST / name}: {len(out):,} rows, non-empty {(out[col] != '').mean():.3f}")
    full = OUT.parent / "output_full"
    rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
    a, b = rd(full / "matching_results.tsv"), rd(DEST / "matching_results.tsv")
    d = (a.join(b, on="source1_entity_id", suffix="_new")
          .join(s1ids.join(ctry, on="s1_id").select("source1_entity_id", "country"), on="source1_entity_id")
          .filter(pl.col("matched_entity_ids").fill_null("") != pl.col("matched_entity_ids_new").fill_null("")))
    print("rows changed vs output_full by country:", dict(d.group_by("country").len().iter_rows()))
    print("candidate_pairs identical to output_full:", (full / "candidate_pairs.tsv").read_bytes() == (DEST / "candidate_pairs.tsv").read_bytes())


if __name__ == "__main__":
    main()
