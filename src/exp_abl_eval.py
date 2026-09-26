"""A-E ablation table on the SAME pinned 20 split-halves.  Usage: python src/exp_abl_eval.py
A saved 0.98108 | B queried-universe union, no dense | C full-union competition, no dense | D queried-universe union + dense | E full union + dense (exp_union.py run).
Per config: production-rule F0.5 / P / R / pred per S1 / India / US, blocking recall, candidates/S1, held-out F0.5 mean / worst / best over the 20 split-halves,
the 20 held-out values, and the paired gain vs A (mean / min / max).  Held-out = decision rule re-chosen on one half, scored on the other (exp_stack.evaluate protocol).
"""
import json

import numpy as np
import polars as pl

import exp_stack as S
from common import CACHE
from exp_budget import val_truth

E = CACHE.parent / "exp"
FILES = {"A saved 0.98108": E / "final" / "pred_TRAIN__NEW3_NEW4.parquet",
         "B cand only (queried univ.)": E / "union_abl_B" / "pred_TRAIN_abl_B.parquet",
         "C cand + full-union comp.": E / "union_abl_C" / "pred_TRAIN_abl_C.parquet",
         "D cand + dense (queried)": E / "union_abl_D" / "pred_TRAIN_abl_D.parquet",
         "E full union + dense": E / "union" / "pred_TRAIN_union_rev_k1.parquet"}


def main():
    _, truth_n, gt = val_truth()
    n_true, n_s1 = gt.height, truth_n.height
    ctry = pl.read_parquet(CACHE / "norm" / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    india = truth_n.join(ctry, on="s1_id")["country"].to_numpy() == "India"
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    kc = (cfg["alpha"], cfg["miss"], cfg["gamma"], cfg["floor"])
    res, meta, held = {}, {}, {}
    for name, f in FILES.items():
        if not f.exists():
            print(f"{name}: MISSING {f}")
            continue
        p = pl.read_parquet(f)
        res[name] = S.scores(p, truth_n)
        v = p.filter(pl.col("role") == "val")
        meta[name] = (v.filter(pl.col("is_match")).height / n_true, v.height / n_s1)
        vals = []
        for seed in range(10):
            m = np.random.default_rng(seed).random(len(truth_n)) < 0.5
            for A, B in ((m, ~m), (~m, m)):
                pick = max(res[name], key=lambda c: res[name][c][0][A].mean())
                vals.append(res[name][pick][0][B].mean())
        held[name] = np.array(vals)
    ref = "A saved 0.98108"
    print(f"\n{'config':30s} {'F0.5':>8s} {'P':>7s} {'R':>7s} {'pred/S1':>7s} {'India':>8s} {'US':>8s} | {'block.rec':>9s} {'cand/S1':>7s} | held-out mean  worst   best | gain vs A: mean  min   max")
    for name in res:
        f, pr, r, k = res[name][kc]
        g = held[name] - held[ref]
        print(f"{name:30s} {f.mean():8.5f} {pr.mean():7.4f} {r.mean():7.4f} {k.mean():7.3f} {f[india].mean():8.5f} {f[~india].mean():8.5f} | {meta[name][0]:9.4f} {meta[name][1]:7.2f} | "
              f"{held[name].mean():.5f} {held[name].min():.5f} {held[name].max():.5f} | {g.mean():+.5f} {g.min():+.5f} {g.max():+.5f}   (production-rule gain vs A {f.mean() - res[ref][kc][0].mean():+.5f})")
    print("\nheld-out F0.5 of the 20 split-halves (sorted):")
    for name in res:
        print(f"  {name:30s}", " ".join(f"{x:.5f}" for x in np.sort(held[name])))
    print("\nattribution (production-rule F0.5 differences):")
    fm = {n: res[n][kc][0].mean() for n in res}
    def d(a, b):
        return f"{fm[a] - fm[b]:+.5f}" if a in fm and b in fm else "n/a"
    print(f"  candidate expansion (B - A):                 {d('B cand only (queried univ.)', 'A saved 0.98108')}")
    print(f"  full-union owner competition (C - B):         {d('C cand + full-union comp.', 'B cand only (queried univ.)')}")
    print(f"  dense features, queried universe (D - B):     {d('D cand + dense (queried)', 'B cand only (queried univ.)')}")
    print(f"  dense features, full-union comp. (E - C):     {d('E full union + dense', 'C cand + full-union comp.')}")
    print(f"  full-union competition, with dense (E - D):   {d('E full union + dense', 'D cand + dense (queried)')}")
    print(f"  total (E - A):                                {d('E full union + dense', 'A saved 0.98108')}")
    print("DONE")


if __name__ == "__main__":
    main()
