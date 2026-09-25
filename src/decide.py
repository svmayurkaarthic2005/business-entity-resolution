"""Phase 6: decision layer -- turn calibrated pair probabilities into match lists.

Usage:  python src/decide.py                 # tune on VAL rows of pred_TRAIN -> cache/decide.json
        python src/decide.py --split TEST    # apply tuned rule -> output/matching_results.tsv + candidate_pairs.tsv
Rule (all parameters tuned for macro F0.5 on VAL):
  1. owner exclusivity: a record's probability is multiplied by `alpha` for every S1 that is not
     its most likely owner (ground truth: each S2/S3 record has exactly one owner)
  2. per S1, candidates sorted by p; keep the prefix that maximises expected F0.5
         E[F](k) = 1.25 * sum(p[:k]) / (0.25 * (sum(p) + miss) + k)
     predict nothing if  gamma * P(no match) = gamma * prod(1 - p)  beats the best prefix
  3. never keep a candidate with p < floor
Also reports the plain global threshold for comparison.
"""
import argparse
import itertools
import json

import numpy as np
import polars as pl

from common import CACHE, OUT, log
from score import score


def owner_adjust(pred, alpha):
    """Down-weight rows where the S1 is not the record's most likely owner."""
    return pred.with_columns(pa=pl.when(pl.col("p") >= pl.col("p").max().over("cand_id", "source"))
                             .then(pl.col("p")).otherwise(pl.col("p") * alpha))


def choose(pred, miss, gamma, floor):
    """Expected-F0.5 prefix selection per S1. Returns rows with a boolean `take`."""
    d = pred.filter(pl.col("pa") >= floor).sort(["s1_id", "pa"], descending=[False, True])
    d = d.with_columns(k=pl.int_range(1, pl.len() + 1).over("s1_id"),
                       cs=pl.col("pa").cum_sum().over("s1_id"),
                       tot=pl.col("pa").sum().over("s1_id"))
    d = d.with_columns(ef=1.25 * pl.col("cs") / (0.25 * (pl.col("tot") + miss) + pl.col("k")))
    p0 = pred.group_by("s1_id").agg(p0=(1 - pl.col("pa").clip(0, 0.999999)).log().sum().exp())
    best = d.group_by("s1_id").agg(kbest=pl.col("k").get(pl.col("ef").arg_max()), efbest=pl.col("ef").max())
    best = best.join(p0, on="s1_id", how="left").with_columns(
        kbest=pl.when(gamma * pl.col("p0") > pl.col("efbest")).then(0).otherwise(pl.col("kbest")))
    return d.join(best.select("s1_id", "kbest"), on="s1_id").with_columns(take=pl.col("k") <= pl.col("kbest"))


def macro_f05(taken, truth_n):
    """Vectorised macro F0.5 over every S1 in truth_n (s1_id, n). taken: rows with is_match of kept preds."""
    agg = taken.group_by("s1_id").agg(k=pl.len(), tp=pl.col("is_match").sum())
    t = truth_n.join(agg, on="s1_id", how="left").fill_null(0)
    f = (pl.when((pl.col("n") == 0) & (pl.col("k") == 0)).then(1.0)
           .when((pl.col("n") == 0) | (pl.col("k") == 0)).then(0.0)
           .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n") + pl.col("k"))))
    return float(t.select(f.mean()).item())


def tune(pred_file="pred_TRAIN.parquet"):
    pred = pl.read_parquet(CACHE / pred_file)
    val_ids = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
    if pred_file == "pred_TRAIN.parquet" and (CACHE / "queried_TRAIN.parquet").exists():  # TRAIN run on a sample
        val_ids = val_ids.join(pl.read_parquet(CACHE / "queried_TRAIN.parquet"), on="s1_id")
    elif pred_file != "pred_TRAIN.parquet":  # partial VAL runs (--frac): score only the S1 that were queried
        val_ids = val_ids.join(pred.filter(pl.col("role") == "val").select("s1_id").unique(), on="s1_id")
    truth_n = val_ids.join(pl.read_parquet(CACHE / "gt_s1.parquet").rename({"n_matches": "n"}), on="s1_id")
    log(f"VAL: {len(truth_n):,} S1, {truth_n['n'].sum():,} true pairs; pred rows {len(pred):,}")
    results = []
    for alpha in (1.0, 0.5, 0.2, 0.0):
        pa = owner_adjust(pred, alpha).filter(pl.col("role") == "val")
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            results.append(({"rule": "thr", "alpha": alpha, "t": t},
                             macro_f05(pa.filter(pl.col("pa") >= t), truth_n)))
        for miss, gamma, floor in itertools.product((0.0, 0.3, 0.6), (0.5, 1.0, 2.0), (0.05, 0.15, 0.3)):
            ch = choose(pa, miss, gamma, floor)
            results.append(({"rule": "ef", "alpha": alpha, "miss": miss, "gamma": gamma, "floor": floor},
                            macro_f05(ch.filter("take"), truth_n)))
    results.sort(key=lambda r: -r[1])
    for cfg, f in results[:8]:
        print(f"{f:.5f}  {cfg}")
    best_thr = max((r for r in results if r[0]["rule"] == "thr"), key=lambda r: r[1])
    print(f"best plain threshold: {best_thr[1]:.5f} {best_thr[0]}")
    cfg, f = results[0]
    # cross-check the vectorised metric against the reference scorer once
    kept = apply(pred, cfg).filter(pl.col("role") == "val")
    pd_ = {s: set(v) for s, v in kept.group_by("s1_id").agg(pl.col("cand_id") * 10 + pl.col("source")).iter_rows()}
    gt = pl.read_parquet(CACHE / "gt_pairs.parquet").join(val_ids, on="s1_id")
    td = {s: set() for s in truth_n["s1_id"].to_list()}
    for s, c, src in gt.iter_rows():
        td[s].add(c * 10 + src)
    ref = score(pd_, td)
    assert abs(ref["f05"] - f) < 1e-9, (ref["f05"], f)
    print(f"VAL macro F0.5 {ref['f05']:.5f} | precision {ref['precision']:.4f} recall {ref['recall']:.4f} "
          f"singleton acc {ref['singleton_acc']:.4f}")
    json.dump({"cfg": cfg, "val_f05": f, "report": ref}, open(CACHE / "decide.json", "w"), indent=1)


def apply(pred, cfg):
    """Kept (s1_id, cand_id, source) rows for a config."""
    pa = owner_adjust(pred, cfg["alpha"])
    if cfg["rule"] == "thr":
        return pa.filter(pl.col("pa") >= cfg["t"])
    return choose(pa, cfg["miss"], cfg["gamma"], cfg["floor"]).filter("take")


def write_outputs(pred_file="pred_TEST.parquet"):
    """Final TEST files: one row per test S1, comma-separated S2-/S3- entity ids."""
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    pred = pl.read_parquet(CACHE / pred_file)
    kept = apply(pred, cfg)
    ids = {s: pl.read_parquet(CACHE / f"test_source{s}.parquet", columns=["id", "entity_id"]) for s in (1, 2, 3)}
    ent = pl.concat([ids[s].with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    s1 = ids[1].rename({"id": "s1_id", "entity_id": "source1_entity_id"})
    OUT.mkdir(parents=True, exist_ok=True)
    for rows, col, name in ((kept, "matched_entity_ids", "matching_results.tsv"),
                            (pred, "candidate_entity_ids", "candidate_pairs.tsv")):
        lists = (rows.select("s1_id", "cand_id", "source").unique().join(ent, on=["cand_id", "source"])
                 .sort("entity_id").group_by("s1_id").agg(pl.col("entity_id").str.join(",").alias(col)))
        out = s1.join(lists, on="s1_id", how="left").with_columns(pl.col(col).fill_null("")) \
                .sort("s1_id").select("source1_entity_id", col)
        out.write_csv(OUT / name, separator="\t", quote_style="never")
        log(f"wrote {name}: {len(out):,} rows, non-empty {(out[col] != '').mean():.3f}, "
            f"avg ids {(out[col].str.count_matches(',') + 1).filter(out[col] != '').mean():.2f}")
    log(f"config {cfg}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="VAL", choices=["VAL", "TEST"])
    ap.add_argument("--pred", default=None, help="prediction parquet in cache/ (default pred_<TRAIN|TEST>)")
    ap.add_argument("--from-candidates", default=None,
                    help="baseline: build the pred file from a candidates parquet, using the re-ranker score as p")
    a = ap.parse_args()
    if a.from_candidates:
        c = pl.read_parquet(CACHE / a.from_candidates)
        gt = pl.read_parquet(CACHE / "gt_pairs.parquet").select(
            "s1_id", cand_id="match_id", source="match_source", is_match=pl.lit(True))
        c = c.join(gt, on=["s1_id", "cand_id", "source"], how="left").with_columns(pl.col("is_match").fill_null(False))
        a.pred = "pred_baseline_" + a.split + ".parquet"
        c.select("s1_id", "cand_id", "source", "is_match", role=pl.lit("val"), p="score").write_parquet(CACHE / a.pred)
    if a.split == "VAL":
        tune(a.pred or "pred_TRAIN.parquet")
    else:
        write_outputs(a.pred or "pred_TEST.parquet")
