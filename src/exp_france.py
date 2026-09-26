"""Experiment (read-only): France vs US/India behaviour of the existing TEST predictions, with the same
diagnostics on labelled VAL (US/India) as the anchor.  Writes only /content/exp_france.log (stdout).

Usage: python src/exp_france.py > /content/exp_france.log
"""
import json

import numpy as np
import polars as pl

from common import CACHE
from decide import apply

N = CACHE / "norm"
CFG = json.load(open(CACHE / "decide.json"))["cfg"]
LEGAL = r"\b(sarl|sas|sasu|sa|sci|eurl|snc|selarl|scp|ei|ltd|llc|inc|corp|pvt|private|limited|llp|co|company|plc|gmbh)\b"
GENERIC = (r"\b(association|amicale|ecole|club|comite|centre|center|syndicat|mairie|commune|eglise|paroisse|"
           r"cooperative|federation|union|societe|society|community|church|school|college|foundation|fondation|"
           r"institute|institut|services|service|group|groupe)\b")
FR_ABBR = r"(?i)\b(r|av|bd|chem|imp|rte|pl|all|sq|fg|lieu[- ]dit|zi|za|zac|bp|cedex|n°|bis|ter)\b\.?"
ACCENT = r"[À-ÖØ-öø-ÿ]"


def s1_table(split):
    pool = "test" if split == "TEST" else "train"
    s1 = pl.read_parquet(CACHE / f"{pool}_source1.parquet", columns=["id", "name", "address"]).rename({"id": "s1_id"})
    nm = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["id", "country", "name_norm", "addr_missing"]).rename({"id": "s1_id"})
    t = s1.join(nm, on="s1_id").with_columns(
        legal=pl.col("name_norm").str.contains(LEGAL),
        accent=(pl.col("name") + " " + pl.col("address")).str.contains(ACCENT),
        generic=pl.col("name_norm").str.contains(GENERIC),
        fr_abbr=pl.col("address").str.contains(FR_ABBR),
        no_addr=pl.col("addr_missing"))
    return t


def q(s, qs=(0.5, 0.9, 0.95, 0.99)):
    s = s.drop_nulls()
    return " ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in [s.quantile(x) for x in qs]) if len(s) else "-"


def per_s1(pred, kept, t):
    c = pred.group_by("s1_id").agg(n_c=pl.len(), pmax=pl.col("p").max(), psum=pl.col("p").sum())
    k = kept.group_by("s1_id").agg(n_k=pl.len())
    return t.join(c, on="s1_id", how="left").join(k, on="s1_id", how="left").with_columns(
        pl.col("n_c").fill_null(0), pl.col("n_k").fill_null(0), pl.col("psum").fill_null(0.0))


def report(label, d, acc):
    if d.height == 0:
        return
    print(f"{label:22s} S1 {d.height:>8,} | cands mean {d['n_c'].mean():5.2f} med/p90/p95/p99 {q(d['n_c'])} | "
          f"pred/S1 {d['n_k'].mean():.2f} empty {(d['n_k'] == 0).mean():6.2%} | sum p/S1 {d['psum'].mean():.2f} | "
          f"max-p p10/25/50/75 {q(d['pmax'], (0.1, 0.25, 0.5, 0.75))} | accepted p p10/25/50 {q(acc, (0.1, 0.25, 0.5))}")


def block(split, pred, t, extra=None, full=None):
    """full: the whole prediction file, so owner exclusivity sees every S1 exactly as in production/tuning."""
    kept = apply(full if full is not None else pred, CFG).join(t.select("s1_id"), on="s1_id", how="semi")
    d = per_s1(pred, kept, t)
    if extra is not None:
        d = d.join(extra, on="s1_id", how="left")
    ka = kept.join(t.select("s1_id", "country", "legal", "accent", "generic", "fr_abbr", "no_addr"), on="s1_id")
    for c in sorted(d["country"].unique().to_list()):
        dc, kc = d.filter(pl.col("country") == c), ka.filter(pl.col("country") == c)
        f = f" trueF {dc['f'].mean():.4f}" if extra is not None else ""
        report(f"{split} {c}", dc, kc["pa"])
        if f:
            print(f"{'':22s}{f}  true matches/S1 {dc['n'].mean():.2f}")
        for flag in ("legal", "accent", "generic", "fr_abbr", "no_addr"):
            for v in (True, False):
                sub = dc.filter(pl.col(flag) == v)
                share = sub.height / max(dc.height, 1)
                if sub.height and share > 0.005:
                    fx = f"  trueF {sub['f'].mean():.4f}" if extra is not None else ""
                    report(f"  {flag}={'Y' if v else 'n'} ({share:.0%})", sub, kc.filter(pl.col(flag) == v)["pa"])
                    if fx:
                        print(f"{'':22s}{fx}")
    return d, kept


def main():
    pl.Config.set_fmt_str_lengths(80)
    # --- labelled anchor: VAL rows of pred_TRAIN (US/India), with the true per-S1 F0.5
    pfull = pl.read_parquet(CACHE / "pred_TRAIN.parquet")
    pv = pfull.filter(pl.col("role") == "val")
    val_ids = (pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
               .join(pl.read_parquet(CACHE / "queried_TRAIN.parquet"), on="s1_id"))
    tv = s1_table("TRAIN").join(val_ids, on="s1_id")
    kv = apply(pfull, CFG).filter(pl.col("role") == "val")
    agg = kv.group_by("s1_id").agg(k=pl.len(), tp=pl.col("is_match").sum())
    tr = (tv.select("s1_id").join(pl.read_parquet(CACHE / "gt_s1.parquet").rename({"n_matches": "n"}), on="s1_id")
          .join(agg, on="s1_id", how="left").fill_null(0)
          .with_columns(f=pl.when((pl.col("n") == 0) & (pl.col("k") == 0)).then(1.0)
                        .when((pl.col("n") == 0) | (pl.col("k") == 0)).then(0.0)
                        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n") + pl.col("k")))).select("s1_id", "n", "f"))
    print(f"decision cfg {CFG}\nVAL overall true macro F0.5 {tr['f'].mean():.5f} (must equal 0.96303)\n\n=== VAL (labelled) ===")
    block("VAL", pv, tv, tr, full=pfull)

    print("\n=== TEST ===")
    pt = pl.read_parquet(CACHE / "pred_TEST.parquet")
    tt = s1_table("TEST")
    d, kept = block("TEST", pt, tt)

    print("\n=== pool size per S1 (S2+S3 records per S1) ===")
    for pool in ("train", "test"):
        s1c = pl.read_parquet(N / f"{pool}_source1.parquet", columns=["country"]).group_by("country").len("s1")
        oc = pl.concat([pl.read_parquet(N / f"{pool}_source{s}.parquet", columns=["country"]) for s in (2, 3)]).group_by("country").len("o")
        for r in s1c.join(oc, on="country").sort("country").iter_rows(named=True):
            print(f"  {pool} {r['country']:8s} S1 {r['s1']:>9,}  S2+S3 {r['o']:>9,}  ratio {r['o'] / r['s1']:.2f}")

    print("\n=== top-candidate share per country on TEST: fraction of S1 whose best cand has p>=0.5 / >=0.9 ===")
    for r in d.group_by("country").agg(ge5=(pl.col("pmax") >= 0.5).mean(), ge9=(pl.col("pmax") >= 0.9).mean()).sort("country").iter_rows(named=True):
        print(f"  {r['country']:8s} {r['ge5']:.3f} {r['ge9']:.3f}")

    # --- manual inspection samples (France)
    raw = {s: pl.read_parquet(CACHE / f"test_source{s}.parquet", columns=["id", "name", "address"]) for s in (1, 2, 3)}
    fr = tt.filter(pl.col("country") == "France").select("s1_id")
    ka = kept.join(fr, on="s1_id")

    def show(s1_id, rows):
        a = raw[1].row(s1_id, named=True)
        print(f"\nS1 {a['name']} | {a['address']}")
        for r in rows.iter_rows(named=True):
            b = raw[r["source"]].row(r["cand_id"], named=True)
            print(f"   {'KEEP' if r.get('keep') else '    '} p={r['p']:.3f} S{r['source']} {b['name']} | {b['address']}")

    allp = pt.join(fr, on="s1_id").join(ka.select("s1_id", "cand_id", "source", keep=pl.lit(True)),
                                         on=["s1_id", "cand_id", "source"], how="left")
    for title, ids in (
            ("20 HIGH-CONFIDENCE matches (accepted, p>=0.95)", ka.filter(pl.col("pa") >= 0.95)["s1_id"].unique().sample(20, seed=1)),
            ("20 BORDERLINE matches (accepted, lowest-p 0.15-0.5)", ka.filter(pl.col("pa") < 0.5)["s1_id"].unique().sample(20, seed=1)),
            ("20 PREDICTED-EMPTY S1 (top-3 candidates shown)", d.filter((pl.col("country") == "France") & (pl.col("n_k") == 0))["s1_id"].sample(20, seed=1))):
        print(f"\n\n##### France: {title}")
        for s in ids.to_list():
            show(s, allp.filter(pl.col("s1_id") == s).sort("p", descending=True).head(4))


if __name__ == "__main__":
    main()
