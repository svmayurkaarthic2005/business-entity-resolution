"""Experiment: adaptive top-K on the labelled VAL split, existing model (no retraining).

Needs cache/candidates_VAL.parquet from:  python src/block.py --split VAL --k 30 --floor 0.003
For each variant the VAL S1 candidates are cut to K, merged with the production TRAIN candidates of all other S1
(so owner-competition features see the same competitors as in production), re-featurised, scored with the saved
model, and decided with the production rule.  Variants: K=10/15/20/30 and adaptive K (10 -> 20/30 only for an
(S1, source) whose 10th candidate has re-ranker score >= t).
Usage: python src/exp_k.py      (writes only exp/adaptive_k/)
"""
import json
import pickle
import time

import numpy as np
import polars as pl

import features as F
from common import CACHE, log
from decide import apply
from exp_budget import val_truth
from exp_decide import per_s1
from predict import load_models

N = CACHE / "norm"
OUTD = CACHE.parent / "exp" / "adaptive_k"
KEY = ["s1_id", "cand_id", "source"]


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    val_ids, truth_n, gt = val_truth()
    cv = pl.read_parquet(CACHE / "candidates_VAL.parquet").join(val_ids, on="s1_id", how="semi")
    ctr = pl.read_parquet(CACHE / "candidates_TRAIN.parquet")
    a, b = ctr.join(val_ids, on="s1_id", how="semi").select(KEY), cv.filter(pl.col("rank") < 10).select(KEY)
    both = a.join(b, on=KEY, how="semi").height
    log(f"VAL K30 candidates {cv.height:,}; K10 subset vs production TRAIN rows: {both / a.height:.4%} of production rows reproduced, "
        f"{b.height:,} vs {a.height:,} rows")
    other = ctr.join(val_ids, on="s1_id", how="anti")
    cv = cv.with_columns(s10=pl.col("score").filter(pl.col("rank") == 9).first().over("s1_id", "source").fill_null(0.0))

    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country", *F.TXT]).rename(
        lambda x: x if x in ("id", "country") else x + "_1").rename({"id": "s1_id"})
    cand = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "name_raw", *F.TXT])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id", "name_raw": "name_raw_c"})
    toks = s1.select(pl.col("name_core_1").str.split(" ").list.unique().alias("tok")).explode("tok")
    idf = toks.group_by("tok").len().with_columns(idf=(len(s1) / pl.col("len")).log().cast(pl.Float32)).select("tok", "idf")
    cal = pickle.load(open(CACHE / "calib.pkl", "rb"))
    models = load_models(cal)
    lab = gt.select(KEY).with_columns(is_match=pl.lit(True))
    n_true, n_s1 = gt.height, truth_n.height

    r = pl.col("rank")
    variants = {"K10": r < 10, "K15": r < 15, "K20": r < 20, "K30": r < 30,
                "adapt20 s10>=.05": (r < 10) | ((r < 20) & (pl.col("s10") >= 0.05)),
                "adapt30 s10>=.05": (r < 10) | ((r < 30) & (pl.col("s10") >= 0.05)),
                "adapt20 s10>=.2": (r < 10) | ((r < 20) & (pl.col("s10") >= 0.2))}
    print(f"{'variant':18s} pairRecall S1allFound cands/S1 |  F0.5     P      R    pred/S1 | runtime")
    for name, rule in variants.items():
        t0 = time.time()
        cvk = cv.filter(rule).drop("s10")
        c = F.competition(pl.concat([other, cvk], how="vertical_relaxed"))
        vc = cvk.select("cand_id", "source").unique()
        c = c.join(vc, on=["cand_id", "source"], how="semi")  # VAL rows + every other S1 row on the same records
        out = []
        for i in range(0, len(c), F.CHUNK):
            d = c.slice(i, F.CHUNK).join(s1, on="s1_id", how="left").join(cand, on=["cand_id", "source"], how="left")
            d = F.pair_features(d, idf)
            out.append(d.drop([x for x in d.columns if x.endswith("_1") or x in F.TXT or x == "name_raw_c"]))
        f = pl.concat(out)
        x = f.select([pl.col(k).cast(pl.Float32) for k in cal["cols"]]).to_numpy()
        p = cal["iso"].predict(np.mean([m(x) for m in models], axis=0)).astype(np.float32)
        pred = (f.select(KEY).with_columns(p=pl.Series(p))
                  .with_columns(role=pl.when(pl.col("s1_id").is_in(val_ids["s1_id"])).then(pl.lit("val")).otherwise(pl.lit("comp")))
                  .join(lab, on=KEY, how="left").with_columns(pl.col("is_match").fill_null(False)))
        pred.write_parquet(OUTD / f"pred_{name.replace(' ', '_').replace('>=', 'ge')}.parquet")
        kept = apply(pred, cfg).filter(pl.col("role") == "val")
        fs, ps, rs, ks = per_s1(kept, truth_n)
        hit = cvk.join(lab, on=KEY, how="inner")
        full = truth_n.filter(pl.col("n") > 0).join(hit.group_by("s1_id").len("h"), on="s1_id", how="left").fill_null(0)
        print(f"{name:18s} {hit.height / n_true:.4f}     {(full['h'] == full['n']).mean():.4f}    {cvk.height / n_s1:6.2f}  | "
              f"{fs.mean():.5f} {ps.mean():.4f} {rs.mean():.4f} {ks.mean():.3f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
