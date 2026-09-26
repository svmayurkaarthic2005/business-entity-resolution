"""Error budget of the BEST configuration on the VAL split (33,172 labelled S1).  Usage: python src/exp_errbudget.py "<config name>" [pred file]

Every VAL true pair that is not in the final prediction is put in exactly ONE category (precedence top to bottom):
  1 not retrieved (missing from the candidate set)
  5 data-quality: retrieved, candidate address missing
  4 contested ownership: retrieved, another S1 scores this record higher
  2 outranked: retrieved, a WRONG candidate of the same S1 scores higher
  3 rejected by the decision rule: retrieved, best-scored for its S1, not contested, still not predicted
  6 other
Also reports, per category, F0.5 points lost (oracle: add the category's pairs to the prediction with no new false positives),
and for category 1 the name patterns (website-style, word order, heavy typo, empty address, generic name, initials, alias, other)
plus how many of them the blocking channels tested in exp_block2 would recover.  Writes exp/errbudget/.
"""
import json
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from common import CACHE
from decide import apply
from exp_budget import val_truth
from exp_stack3 import compact_expr

N = CACHE / "norm"
OUTD = CACHE.parent / "exp" / "errbudget"
KEY = ["s1_id", "cand_id", "source"]
CATS = {1: "not retrieved", 5: "data quality (candidate address missing)", 4: "contested ownership", 2: "outranked by wrong candidate",
        3: "rejected by decision rule", 6: "other"}


def macro_f(truth_n, tp, k):
    t = truth_n.join(tp, on="s1_id", how="left").join(k, on="s1_id", how="left").fill_null(0)
    f = (pl.when((pl.col("n") == 0) & (pl.col("k") == 0)).then(1.0).when((pl.col("n") == 0) | (pl.col("k") == 0)).then(0.0)
           .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n") + pl.col("k"))))
    return float(t.select(f.mean()).item())


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "current"
    fn = sys.argv[2] if len(sys.argv) > 2 else f"pred_TRAIN_{''.join(c if c.isalnum() else '_' for c in name)}.parquet"
    OUTD.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    pred = pl.read_parquet(CACHE.parent / "exp" / "final" / fn, columns=[*KEY, "role", "is_match", "p"])
    _, truth_n, gt = val_truth()
    val = pred.filter(pl.col("role") == "val")
    kept = apply(pred, cfg).filter(pl.col("role") == "val")
    tp_kept = kept.filter(pl.col("is_match")).group_by("s1_id").len("tp")
    k_kept = kept.group_by("s1_id").len("k")
    base_f = macro_f(truth_n, tp_kept, k_kept)
    print(f"config '{name}': macro F0.5 {base_f:.5f}; VAL S1 {truth_n.height:,}; true pairs {gt.height:,}; predicted {kept.height:,} "
          f"(TP {kept['is_match'].sum():,})")

    # ---- context features for every true pair
    top = pred.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("cand_id", "source"))
    o1 = top.filter(pl.col("r") == 1).select("cand_id", "source", o1_s1="s1_id", o1_p="p")
    o2 = top.filter(pl.col("r") == 2).select("cand_id", "source", o2_p="p")
    wrong_best = val.filter(~pl.col("is_match")).group_by("s1_id").agg(wrong_p=pl.col("p").max())
    meta = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "addr_missing", "name_raw", "name_core", "name_alt", "country"])
                      .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename(
        {"id": "cand_id", "name_raw": "n2raw", "name_core": "n2", "name_alt": "alt2"})
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "name_raw", "name_core"]).rename({"id": "s1_id", "name_raw": "n1raw", "name_core": "n1"})
    t = (gt.join(val.select(*KEY, "p"), on=KEY, how="left").join(kept.select(*KEY, taken=pl.lit(True)), on=KEY, how="left")
           .join(o1, on=["cand_id", "source"], how="left").join(o2, on=["cand_id", "source"], how="left")
           .join(wrong_best, on="s1_id", how="left").join(meta, on=["cand_id", "source"], how="left").join(s1, on="s1_id", how="left")
           .with_columns(pl.col("taken").fill_null(False)))
    other_best = pl.when(pl.col("o1_s1") == pl.col("s1_id")).then(pl.col("o2_p").fill_null(0.0)).otherwise(pl.col("o1_p"))
    t = t.with_columns(other_best=other_best)
    miss = t.filter(~pl.col("taken"))
    cat = (pl.when(pl.col("p").is_null()).then(1).when(pl.col("addr_missing")).then(5).when(pl.col("other_best") > pl.col("p")).then(4)
             .when(pl.col("wrong_p").fill_null(0.0) > pl.col("p")).then(2).when(pl.col("p") < 0.5).then(3).otherwise(6))
    miss = miss.with_columns(cat=cat)
    n_fn = miss.height
    print(f"\nfalse negatives: {n_fn:,} of {gt.height:,} true pairs ({n_fn / gt.height:.2%}); true positives {gt.height - n_fn:,}")
    print("\ncategory                                         pairs     share   F0.5 points lost (oracle)")
    for c, label in sorted(CATS.items()):
        sub = miss.filter(pl.col("cat") == c)
        add = sub.group_by("s1_id").len("c")
        tp2 = tp_kept.join(add, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", tp=pl.col("tp") + pl.col("c"))
        k2 = k_kept.join(add, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", k=pl.col("k") + pl.col("c"))
        print(f"  {c} {label:44s} {sub.height:6,}  {sub.height / n_fn:6.1%}   {macro_f(truth_n, tp2, k2) - base_f:+.5f}")
    allfix = miss.group_by("s1_id").len("c")
    tp2 = tp_kept.join(allfix, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", tp=pl.col("tp") + pl.col("c"))
    k2 = k_kept.join(allfix, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", k=pl.col("k") + pl.col("c"))
    print(f"  all false negatives fixed (no new FP): {macro_f(truth_n, tp2, k2):.5f}  (gap to 1.0 also includes {kept.filter(~pl.col('is_match')).height:,} false positives)")
    fp = kept.filter(~pl.col("is_match"))
    print(f"  false positives kept: {fp.height:,}")

    # ---- category 1: unretrieved pairs, name patterns
    u = miss.filter(pl.col("cat") == 1).with_columns(compact_a=compact_expr("n1"), compact_b=compact_expr("n2"))
    ch_dir = CACHE.parent / "exp" / "block2"
    tested = [pl.read_parquet(f).select(KEY) for f in sorted(ch_dir.glob("chan_*.parquet"))]
    tested = pl.concat(tested).unique() if tested else None
    rows = u.select("s1_id", "cand_id", "source", "n1raw", "n2raw", "n1", "n2", "alt2", "addr_missing", "compact_a", "compact_b").to_dicts()
    s1cnt = s1.group_by("n1").len("cnt")
    cnt = dict(zip(s1cnt["n1"].to_list(), s1cnt["cnt"].to_list()))
    web = ("www.", "http", ".com", ".net", ".org", ".in", ".co", ".io", ".biz", ".info")
    out = []
    for r in rows:
        raw2, n1, n2 = r["n2raw"] or "", r["n1"] or "", r["n2"] or ""
        tsr, rr = fuzz.token_sort_ratio(n1, n2), fuzz.ratio(n1, n2)
        cr = fuzz.ratio(r["compact_a"] or "", r["compact_b"] or "")
        if raw2.strip().startswith(("#", "@")) or any(w in raw2.lower() for w in web):
            pat = "website / handle style"
        elif tsr >= 90 and rr < 90:
            pat = "word-order change"
        elif r["addr_missing"]:
            pat = "empty candidate address"
        elif (r["alt2"] or "") != "" or "d/b/a" in raw2.lower() or "doing business" in raw2.lower():
            pat = "alias / DBA"
        elif len(n2.replace(" ", "")) <= 4:
            pat = "initials / very short name"
        elif cnt.get(n1, 0) >= 3:
            pat = "common / generic name"
        elif 55 <= cr < 92:
            pat = "heavy typos"
        elif cr < 55:
            pat = "unrelated-looking name (low similarity)"
        else:
            pat = "other"
        out.append(pat)
    u = u.with_columns(pattern=pl.Series(out))
    if tested is not None:
        u = u.join(tested.with_columns(rec=pl.lit(True)), on=KEY, how="left").with_columns(pl.col("rec").fill_null(False))
    else:
        u = u.with_columns(rec=pl.lit(False))
    print(f"\nunretrieved true pairs: {u.height:,}  (channels tested in exp_block2 would recover {u['rec'].sum():,} = {u['rec'].mean():.1%})")
    tab = u.group_by("pattern").agg(n=pl.len(), recov=pl.col("rec").sum()).sort("n", descending=True)
    print("pattern                                   pairs   share  recoverable by tested channels")
    for r in tab.iter_rows(named=True):
        print(f"  {r['pattern']:40s} {r['n']:5,}  {r['n'] / u.height:6.1%}   {r['recov']:5,} ({r['recov'] / r['n']:.0%})")
    for pat in tab["pattern"].to_list():
        print(f"\n--- examples: {pat}")
        for r in u.filter(pl.col("pattern") == pat).sample(min(6, (u["pattern"] == pat).sum()), seed=3).iter_rows(named=True):
            print(f"  S1 '{r['n1raw']}'  <-  S{r['source']} '{r['n2raw']}'  (cand addr missing: {r['addr_missing']})")
    u.write_parquet(OUTD / "unretrieved.parquet")
    miss.write_parquet(OUTD / "false_negatives.parquet")


if __name__ == "__main__":
    main()
