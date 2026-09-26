"""Targeted error budget of the UNION model (validation, production decision rule).  Usage: python src/exp_errbudget2.py [pred_file]  (default exp/union/pred_TRAIN_union_rev_k1.parquet)

FALSE NEGATIVES (true pair not in the final prediction), exclusive, checked in this order:
  1 never retrieved            the pair is not in the candidate set
  2 outranked                  retrieved, and a FALSE candidate of the same S1 has a higher score
  4 other: owner contested     retrieved, no false candidate of the same S1 scores higher, but ANOTHER S1 scores the same record higher
  3 rejected by decision rule  retrieved, scored above every false candidate of its S1, not contested, still not predicted
FALSE POSITIVES (kept pair that is false), exclusive, checked in this order:
  owner-competition failure    the record truly belongs to a DIFFERENT S1
  sibling business             same core name as the S1 but a different address / house number (record owned by nobody)
  legal-suffix noise           same core name, same/absent house number, different legal suffix
  address conflict             similar name (token-set >= 80) but the house numbers / address numbers conflict
  generic name                the S1 core name is shared by >= 3 S1 records (label-free count)
  other
For every category: errors, share, macro-F0.5 gain if that category alone were perfectly fixed (FN: pairs added with no new FP; FP: pairs removed) -- ALL / US / India.
That one-at-a-time oracle IS both the 'current impact' and the 'maximum theoretical improvement'; it says nothing about what is achievable.
"""
import json
import sys

import polars as pl
from rapidfuzz import fuzz

from common import CACHE
from decide import apply
from exp_budget import val_truth
from exp_errbudget import macro_f

N = CACHE / "norm"
E = CACHE.parent / "exp"
KEY = ["s1_id", "cand_id", "source"]


def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else str(E / "union" / "pred_TRAIN_union_rev_k1.parquet")
    pred = pl.read_parquet(fn, columns=[*KEY, "role", "is_match", "p"])
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    _, truth_n, gt = val_truth()
    val = pred.filter(pl.col("role") == "val")
    kept = apply(pred, cfg).filter(pl.col("role") == "val")
    ctry = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    tn = {"ALL": truth_n, **{c: truth_n.join(ctry.filter(pl.col("country") == c), on="s1_id", how="semi") for c in ("US", "India")}}
    tp0 = kept.filter(pl.col("is_match")).group_by("s1_id").len("tp")
    k0 = kept.group_by("s1_id").len("k")
    base = {c: macro_f(t, tp0, k0) for c, t in tn.items()}
    print(f"model: {fn}\nVAL S1 {truth_n.height:,} (US {tn['US'].height:,}, India {tn['India'].height:,}); true pairs {gt.height:,}; predicted {kept.height:,} "
          f"(TP {kept['is_match'].sum():,}, FP {kept.filter(~pl.col('is_match')).height:,})")
    print("macro F0.5 (production rule): " + "  ".join(f"{c} {v:.5f}" for c, v in base.items()) + "\n")

    def gains(add_tp=None, add_k=None, sub_k=None):
        out = {}
        tp, k = tp0, k0
        if add_tp is not None:
            tp = tp0.join(add_tp, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", tp=pl.col("tp") + pl.col("c"))
        if add_k is not None:
            k = k0.join(add_k, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", k=pl.col("k") + pl.col("c"))
        if sub_k is not None:
            k = k0.join(sub_k, on="s1_id", how="left").fill_null(0).select("s1_id", k=pl.col("k") - pl.col("c"))
        for c, t in tn.items():
            out[c] = macro_f(t, tp, k) - base[c]
        return out

    # ------------------------------------------------------------- false negatives
    miss = gt.join(kept.select(KEY).with_columns(taken=pl.lit(True)), on=KEY, how="left").filter(pl.col("taken").is_null()).drop("taken")
    s1c = ctry.select("s1_id", "country")
    miss = miss.join(val.select(*KEY, "p"), on=KEY, how="left").join(s1c, on="s1_id", how="left")
    fmax = val.filter(~pl.col("is_match")).group_by("s1_id").agg(fmax=pl.col("p").max())
    top = pred.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("cand_id", "source"))
    o1 = top.filter(pl.col("r") == 1).select("cand_id", "source", o1_s1="s1_id", o1_p="p")
    o2 = top.filter(pl.col("r") == 2).select("cand_id", "source", o2_p="p")
    miss = miss.join(fmax, on="s1_id", how="left").join(o1, on=["cand_id", "source"], how="left").join(o2, on=["cand_id", "source"], how="left")
    other_best = pl.when(pl.col("o1_s1") == pl.col("s1_id")).then(pl.col("o2_p").fill_null(0.0)).otherwise(pl.col("o1_p"))
    miss = miss.with_columns(ob=other_best).with_columns(
        cat=pl.when(pl.col("p").is_null()).then(pl.lit("1 never retrieved"))
        .when(pl.col("fmax").fill_null(0.0) > pl.col("p")).then(pl.lit("2 outranked by a false candidate"))
        .when(pl.col("ob") > pl.col("p")).then(pl.lit("4 other: owner contested by another S1"))
        .otherwise(pl.lit("3 rejected by decision rule")))
    n_fn = miss.height
    fp = kept.filter(~pl.col("is_match"))
    n_err = n_fn + fp.height
    print(f"FALSE NEGATIVES: {n_fn:,} of {gt.height:,} true pairs ({n_fn / gt.height:.2%})   |   FALSE POSITIVES: {fp.height:,}   |   total errors {n_err:,}")
    hdr = f"{'category':44s} {'errors':>7s} {'%FN':>6s} {'%all':>6s} | {'US':>6s} {'India':>6s} | dF0.5 if fixed: {'ALL':>8s} {'US':>8s} {'India':>8s}"
    print("\n" + hdr)
    for cat in sorted(miss["cat"].unique().to_list()):
        sub = miss.filter(pl.col("cat") == cat)
        add = sub.group_by("s1_id").len("c")
        g = gains(add_tp=add, add_k=add)
        cnt = sub.group_by("country").len().to_dict(as_series=False)
        d = dict(zip(cnt["country"], cnt["len"]))
        print(f"  {cat:42s} {sub.height:7,} {sub.height / n_fn:6.1%} {sub.height / n_err:6.1%} | {d.get('US', 0):6,} {d.get('India', 0):6,} | {'':15s}{g['ALL']:+8.5f} {g['US']:+8.5f} {g['India']:+8.5f}")
    add = miss.group_by("s1_id").len("c")
    g = gains(add_tp=add, add_k=add)
    print(f"  {'ALL false negatives (no new FP)':42s} {n_fn:7,} {'':6s} {n_fn / n_err:6.1%} | {'':13s} | {'':15s}{g['ALL']:+8.5f} {g['US']:+8.5f} {g['India']:+8.5f}")

    # ------------------------------------------------------------- false positives
    m1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "name_core", "name_suffix", "addr_norm", "house_no", "country"]).rename(
        {"id": "s1_id", "name_core": "n1", "name_suffix": "s1x", "addr_norm": "a1", "house_no": "h1"})
    mc = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "name_core", "name_suffix", "addr_norm", "house_no"])
                    .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename({"id": "cand_id", "name_core": "n2", "name_suffix": "s2x", "addr_norm": "a2", "house_no": "h2"})
    owner = (pl.read_parquet(CACHE / "gt_pairs.parquet").select("s1_id", cand_id="match_id", source="match_source")
             .with_columns(pl.col("source").cast(pl.Int8)).rename({"s1_id": "owner"}))
    gen = m1.group_by("n1").len("n_s1_same")
    fp = (fp.join(m1, on="s1_id", how="left").join(mc, on=["cand_id", "source"], how="left")
            .join(owner, on=["cand_id", "source"], how="left").join(gen, on="n1", how="left"))
    rows = fp.select("n1", "n2", "a1", "a2", "h1", "h2", "s1x", "s2x").to_dicts()
    import re
    nums = lambda a: set(re.findall(r"\d+", a or ""))
    lab = []
    for r, ow, s1i, ng in zip(rows, fp["owner"].to_list(), fp["s1_id"].to_list(), fp["n_s1_same"].to_list()):
        same = (r["n1"] or "") == (r["n2"] or "") and (r["n1"] or "") != ""
        ts = fuzz.token_set_ratio(r["n1"] or "", r["n2"] or "")
        h1, h2 = (r["h1"] or ""), (r["h2"] or "")
        n1, n2 = nums(r["a1"]), nums(r["a2"])
        if ow is not None and ow != s1i:
            lab.append("owner-competition failure (record owned by another S1)")
        elif same and h1 and h2 and h1 != h2:
            lab.append("sibling business (same core name, different address)")
        elif same and (r["s1x"] or "") != (r["s2x"] or ""):
            lab.append("legal-suffix noise (same core name, different suffix)")
        elif same and (r["a1"] or "") != (r["a2"] or ""):
            lab.append("sibling business (same core name, different address)")
        elif ts >= 80 and n1 and n2 and not (n1 & n2):
            lab.append("address conflict (similar name, disjoint numbers)")
        elif (ng or 0) >= 3:
            lab.append("generic name (S1 core name shared by >=3 S1)")
        else:
            lab.append("other")
    fp = fp.with_columns(cat=pl.Series(lab))
    print("\n" + hdr.replace("%FN", "%FP"))
    for cat in sorted(fp["cat"].unique().to_list()):
        sub = fp.filter(pl.col("cat") == cat)
        rm = sub.group_by("s1_id").len("c")
        g = gains(sub_k=rm)
        d = dict(sub.group_by("country").len().iter_rows())
        print(f"  {cat:56s} {sub.height:5,} {sub.height / fp.height:6.1%} {sub.height / n_err:6.1%} | {d.get('US', 0):6,} {d.get('India', 0):6,} | {'':15s}{g['ALL']:+8.5f} {g['US']:+8.5f} {g['India']:+8.5f}")
    rm = fp.group_by("s1_id").len("c")
    g = gains(sub_k=rm)
    print(f"  {'ALL false positives removed':56s} {fp.height:5,} {'':6s} {fp.height / n_err:6.1%} | {'':13s} | {'':15s}{g['ALL']:+8.5f} {g['US']:+8.5f} {g['India']:+8.5f}")
    print("\nNOTE: gains are one-at-a-time oracles (they do not add up exactly and are not claims of achievable improvement).")
    miss.write_parquet(E / "union" / "errbudget_fn.parquet")
    fp.write_parquet(E / "union" / "errbudget_fp.parquet")
    print("DONE")


if __name__ == "__main__":
    main()
