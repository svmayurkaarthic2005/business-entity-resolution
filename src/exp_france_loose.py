"""Experiment: France-only looser decision rule on the EXISTING predictions (no retrain / re-block).

Usage:  python src/exp_france_loose.py            # sweep + select + write output_france_loose/
For each config: VAL (US/India, labelled) macro F0.5 / precision / recall / pred-per-S1 with the SAME rule, and
TEST France pred-per-S1, empty %, accepted-score quantiles. Selection (fixed before looking at France):
  keep configs whose VAL F0.5 >= current - 0.0005 (~1.5 s.e.) and VAL precision >= current - 0.002;
  among them take the one whose France pred/S1 is closest to 3.25 (never above 3.30).
Output: output_france_loose/{matching_results,candidate_pairs}.tsv = current rule for US/India + chosen rule for France.
output_full is never touched.
"""
import itertools
import json

import numpy as np
import polars as pl

from common import CACHE, OUT
from decide import choose, owner_adjust
from exp_decide import per_s1

GRID = dict(alpha=(0.3, 0.5, 0.7), miss=(0.7, 0.8, 0.9, 1.0), gamma=(1.0, 1.5, 2.0, 2.5), floor=(0.05, 0.10, 0.15))
DEST = OUT.parent / "output_france_loose"
TARGET, CAP = 3.25, 3.30


def kept_rows(pa_by_alpha, cfg):
    a, miss, gamma, floor = cfg
    return choose(pa_by_alpha[a], miss, gamma, floor).filter("take")


def main():
    cur_cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    cur = (cur_cfg["alpha"], cur_cfg["miss"], cur_cfg["gamma"], cur_cfg["floor"])
    alphas = sorted(set(GRID["alpha"]) | {cur[0]})

    # labelled VAL (US + India), same S1 set as decide.tune
    pfull = pl.read_parquet(CACHE / "pred_TRAIN.parquet")
    val_ids = (pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
               .join(pl.read_parquet(CACHE / "queried_TRAIN.parquet"), on="s1_id"))
    truth_n = val_ids.join(pl.read_parquet(CACHE / "gt_s1.parquet").rename({"n_matches": "n"}), on="s1_id").sort("s1_id")
    pv = {a: owner_adjust(pfull, a).filter(pl.col("role") == "val") for a in alphas}
    del pfull

    # TEST France rows (owner exclusivity computed over all test rows, as in production)
    pt = pl.read_parquet(CACHE / "pred_TEST.parquet")
    ctry = pl.read_parquet(CACHE / "norm" / "test_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    fr_ids = ctry.filter(pl.col("country") == "France").select("s1_id")
    n_fr = fr_ids.height
    pf = {a: owner_adjust(pt, a).join(fr_ids, on="s1_id", how="semi") for a in alphas}

    rows = []
    for cfg in [cur] + [c for c in itertools.product(*GRID.values()) if c != cur]:
        f, p, r, k = per_s1(kept_rows(pv, cfg), truth_n)
        kf = kept_rows(pf, cfg)
        per = kf.group_by("s1_id").len()
        q = kf["pa"].quantile
        rows.append(dict(alpha=cfg[0], miss=cfg[1], gamma=cfg[2], floor=cfg[3],
                         val_f=f.mean(), val_p=p.mean(), val_r=r.mean(), val_k=k.mean(),
                         fr_k=kf.height / n_fr, fr_empty=1 - per.height / n_fr,
                         fr_p10=q(0.10), fr_p25=q(0.25), fr_p50=q(0.5)))
    res = pl.DataFrame(rows)
    base = res.row(0, named=True)
    res = res.with_columns(dF=pl.col("val_f") - base["val_f"], dP=pl.col("val_p") - base["val_p"])
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(4)
    print(f"current {cur}: VAL F {base['val_f']:.5f} P {base['val_p']:.4f} R {base['val_r']:.4f} pred/S1 {base['val_k']:.3f} | "
          f"France pred/S1 {base['fr_k']:.3f} empty {base['fr_empty']:.2%}")
    print("\nall configs sorted by France pred/S1:")
    print(res.sort("fr_k"))

    ok = res.filter((pl.col("dF") >= -0.0005) & (pl.col("dP") >= -0.002) & (pl.col("fr_k") <= CAP))
    print(f"\n{ok.height} configs pass the VAL guard (dF >= -0.0005, dP >= -0.002, France pred/S1 <= {CAP})")
    print(ok.sort("fr_k", descending=True).head(15))
    best = ok.with_columns(gap=(pl.col("fr_k") - TARGET).abs()).sort(["gap", "dF"], descending=[False, True]).row(0, named=True)
    chosen = (best["alpha"], best["miss"], best["gamma"], best["floor"])
    print(f"\nCHOSEN France cfg {chosen}: VAL F {best['val_f']:.5f} (dF {best['dF']:+.5f}) P {best['val_p']:.4f} "
          f"(dP {best['dP']:+.4f}) R {best['val_r']:.4f} | France pred/S1 {best['fr_k']:.3f} empty {best['fr_empty']:.2%} "
          f"accepted p p10/25/50 {best['fr_p10']:.3f}/{best['fr_p25']:.3f}/{best['fr_p50']:.3f}")
    json.dump({"current": cur, "france": chosen, "val": {k: best[k] for k in ("val_f", "val_p", "val_r", "dF", "dP")},
               "france_stats": {k: best[k] for k in ("fr_k", "fr_empty")}},
              open(CACHE / "decide_france_loose.json", "w"), indent=1)

    # outputs: current rule outside France, chosen rule for France
    other = kept_rows({cur[0]: owner_adjust(pt, cur[0]).join(fr_ids, on="s1_id", how="anti")}, cur)
    kept = pl.concat([other.select("s1_id", "cand_id", "source"),
                      kept_rows(pf, chosen).select("s1_id", "cand_id", "source")])
    ids = {s: pl.read_parquet(CACHE / f"test_source{s}.parquet", columns=["id", "entity_id"]) for s in (1, 2, 3)}
    ent = pl.concat([ids[s].with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id"})
    s1 = ids[1].rename({"id": "s1_id", "entity_id": "source1_entity_id"})
    DEST.mkdir(parents=True, exist_ok=True)
    for rws, col, name in ((kept, "matched_entity_ids", "matching_results.tsv"),
                           (pt, "candidate_entity_ids", "candidate_pairs.tsv")):
        lists = (rws.select("s1_id", "cand_id", "source").unique().join(ent, on=["cand_id", "source"])
                 .sort("entity_id").group_by("s1_id").agg(pl.col("entity_id").str.join(",").alias(col)))
        out = s1.join(lists, on="s1_id", how="left").with_columns(pl.col(col).fill_null("")) \
                .sort("s1_id").select("source1_entity_id", col)
        out.write_csv(DEST / name, separator="\t", quote_style="never")
        print(f"wrote {DEST / name}: {len(out):,} rows, non-empty {(out[col] != '').mean():.3f}")
    print(f"kept pairs: total {kept.height:,}; France {kept.height - other.height:,}; others {other.height:,}")

    # sanity: only France rows may differ from output_full; candidate_pairs must be identical
    full = OUT.parent / "output_full"
    rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    a, b = rd(full / "matching_results.tsv"), rd(DEST / "matching_results.tsv")
    d = (a.join(b, on="source1_entity_id", suffix="_new")
          .join(s1.join(ctry, on="s1_id").select("source1_entity_id", "country"), on="source1_entity_id")
          .filter(pl.col("matched_entity_ids") != pl.col("matched_entity_ids_new")))
    print("rows changed vs output_full by country:", dict(d.group_by("country").len().iter_rows()))
    assert d.filter(pl.col("country") != "France").height == 0
    same = (full / "candidate_pairs.tsv").read_bytes() == (DEST / "candidate_pairs.tsv").read_bytes()
    print("candidate_pairs.tsv identical to output_full:", same)


if __name__ == "__main__":
    main()
