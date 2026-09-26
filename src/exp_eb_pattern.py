"""Second-level error budget: WHAT KIND of name/address failure sits inside each false-negative category of the union model (validation).
Usage: python src/exp_eb_pattern.py      (needs exp/union/errbudget_fn.parquet written by exp_eb_union.py)
Patterns (exclusive, checked in this order) for every false-negative true pair:
  website/handle   candidate name is a URL/handle (www., .com, #, @)         empty address   candidate address empty
  word order       token-sort ratio >= 90 but plain ratio < 90              alias/DBA       candidate has an alias part / 'dba'
  initials/short   candidate compact name <= 4 characters                    number conflict both addresses have numbers and none is shared (true pair, disagreeing house number)
  typo/leet/digit  compact-name similarity 55..92 (typos, leetspeak, digit substitutions, domain-style concatenation)
  low similarity   compact-name similarity < 55 (unrelated-looking / transliterated / replaced name)
  generic name     S1 core name shared by >= 3 S1 records                    other
Table: pattern x false-negative category (counts) and the one-at-a-time oracle F0.5 gain of fixing the whole pattern (ALL / US / India).
"""
import json
import re

import polars as pl
from rapidfuzz import fuzz

from common import CACHE
from decide import apply
from exp_budget import val_truth
from exp_errbudget import macro_f
from exp_stack3 import compact_expr

N = CACHE / "norm"
E = CACHE.parent / "exp"
KEY = ["s1_id", "cand_id", "source"]
WEB = ("www.", "http", ".com", ".net", ".org", ".in", ".co", ".io", ".biz", ".info")


def main():
    miss = pl.read_parquet(E / "union" / "errbudget_fn.parquet")
    pred = pl.read_parquet(E / "union" / "pred_TRAIN_union_rev_k1.parquet", columns=[*KEY, "role", "is_match", "p"])
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    _, truth_n, gt = val_truth()
    kept = apply(pred, cfg).filter(pl.col("role") == "val")
    tp0 = kept.filter(pl.col("is_match")).group_by("s1_id").len("tp")
    k0 = kept.group_by("s1_id").len("k")
    ctry = pl.read_parquet(N / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    tn = {"ALL": truth_n, **{c: truth_n.join(ctry.filter(pl.col("country") == c), on="s1_id", how="semi") for c in ("US", "India")}}
    base = {c: macro_f(t, tp0, k0) for c, t in tn.items()}
    s1 = pl.read_parquet(N / "train_source1.parquet", columns=["id", "name_raw", "name_core", "addr_norm"]).rename(
        {"id": "s1_id", "name_raw": "n1raw", "name_core": "n1", "addr_norm": "a1"})
    gen = s1.group_by("n1").len("n_same")
    cd = pl.concat([pl.read_parquet(N / f"train_source{s}.parquet", columns=["id", "name_raw", "name_core", "name_alt", "addr_norm", "addr_missing"])
                    .with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)]).rename(
        {"id": "cand_id", "name_raw": "n2raw", "name_core": "n2", "name_alt": "alt2", "addr_norm": "a2"})
    t = (miss.select(*KEY, "cat", "country", "p").join(s1, on="s1_id", how="left").join(cd, on=["cand_id", "source"], how="left").join(gen, on="n1", how="left")
         .with_columns(ca=compact_expr("n1"), cb=compact_expr("n2")))
    rows = t.select("n1raw", "n2raw", "n1", "n2", "alt2", "a1", "a2", "addr_missing", "ca", "cb", "n_same").to_dicts()
    nums = lambda a: set(re.findall(r"\d+", a or ""))
    pat = []
    for r in rows:
        raw2, n1, n2 = r["n2raw"] or "", r["n1"] or "", r["n2"] or ""
        tsr, rr, cr = fuzz.token_sort_ratio(n1, n2), fuzz.ratio(n1, n2), fuzz.ratio(r["ca"] or "", r["cb"] or "")
        x1, x2 = nums(r["a1"]), nums(r["a2"])
        if raw2.strip().startswith(("#", "@")) or any(w in raw2.lower() for w in WEB):
            pat.append("website / handle name")
        elif r["addr_missing"]:
            pat.append("empty candidate address")
        elif tsr >= 90 and rr < 90:
            pat.append("word-order change")
        elif (r["alt2"] or "") != "" or "d/b/a" in raw2.lower() or "dba" in raw2.lower().split():
            pat.append("alias / DBA")
        elif len(n2.replace(" ", "")) <= 4:
            pat.append("initials / very short name")
        elif x1 and x2 and not (x1 & x2):
            pat.append("address-number conflict (true pair)")
        elif 55 <= cr < 92:
            pat.append("typo / leet / digit / domain-style")
        elif cr < 55:
            pat.append("low name similarity (replaced / transliterated)")
        elif (r["n_same"] or 0) >= 3:
            pat.append("generic name")
        else:
            pat.append("other")
    t = t.with_columns(pattern=pl.Series(pat))
    cats = sorted(t["cat"].unique().to_list())
    print("category counts in the parquet:", dict(t.group_by("cat").len().sort("cat").iter_rows()))
    print(f"macro F0.5 {base['ALL']:.5f} (US {base['US']:.5f}, India {base['India']:.5f}); false negatives {t.height:,}\n")
    print(f"{'pattern':48s} {'total':>6s} " + " ".join(f"{c[:1]:>6s}" for c in cats) + " |   US India | dF0.5 if pattern fixed: ALL       US    India")
    print("   (category columns: " + "; ".join(f"{c[:1]} = {c[2:]}" for c in cats) + ")")
    for p in t.group_by("pattern").len().sort("len", descending=True)["pattern"].to_list():
        sub = t.filter(pl.col("pattern") == p)
        add = sub.group_by("s1_id").len("c")
        tp = tp0.join(add, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", tp=pl.col("tp") + pl.col("c"))
        k = k0.join(add, on="s1_id", how="full", coalesce=True).fill_null(0).select("s1_id", k=pl.col("k") + pl.col("c"))
        g = {c: macro_f(tt, tp, k) - base[c] for c, tt in tn.items()}
        cc = dict(zip(*[sub.group_by("cat").len()[c].to_list() for c in ("cat", "len")]))
        d = dict(zip(*[sub.group_by("country").len()[c].to_list() for c in ("country", "len")]))
        print(f"  {p:46s} {sub.height:6,} " + " ".join(f"{cc.get(c, 0):6,}" for c in cats) + f" | {d.get('US', 0):5,} {d.get('India', 0):5,} | {g['ALL']:+8.5f} {g['US']:+8.5f} {g['India']:+8.5f}")
    for cat in cats:
        print(f"\n--- examples, category {cat}")
        for p in t.filter(pl.col("cat") == cat).group_by("pattern").len().sort("len", descending=True).head(3)["pattern"].to_list():
            for r in t.filter((pl.col("cat") == cat) & (pl.col("pattern") == p)).sample(2, seed=1).iter_rows(named=True):
                print(f"  [{p[:22]}] S1 '{r['n1raw']}' | {r['a1']}  <-  '{r['n2raw']}' | {r['a2']}  (p={r['p']})")
    # rejected-by-rule diagnostics: score distribution of the 'rejected' true pairs
    rej = t.filter(pl.col("cat").str.starts_with("3"))
    q = lambda s: "/".join(f"{s.quantile(x):.3f}" for x in (.1, .25, .5, .75, .9))
    print(f"\nrejected-by-rule true pairs: {rej.height:,}; score p quantiles 10/25/50/75/90 = {q(rej['p'])}; share with p>=0.5: {(rej['p'] >= 0.5).mean():.1%}, p>=0.3: {(rej['p'] >= 0.3).mean():.1%}")
    t.write_parquet(E / "union" / "errbudget_fn_pattern.parquet")
    print("DONE")


if __name__ == "__main__":
    main()
