"""Feature-group ablation of the stage-2 model (VAL, held-out split-halves). Usage: python src/exp_ablate.py  (exp/ablate/)

Reference 'all' = base + stage-2 groups (G,O,SUP,RAR) + number-set (NUM) + name-qualifier (QUAL). Variants drop one group at a time,
plus base+NUM+QUAL WITHOUT any stacking (p1-derived) features, and base+NUM / base+QUAL alone. No is_india flag anywhere.
"""
import pickle

import numpy as np
import polars as pl

import exp_stack as S
import exp_stack2 as E
from common import CACHE, log

OUTA = CACHE.parent / "exp" / "ablate"
GROUPS = {"G": ["g_sum", "g_n5", "g_rank", "g_other_best", "g_gap"],
          "O": ["o_other_best", "o_margin", "o_n5", "o_rank"],
          "SUP": ["sup_name", "sup_both", "sup_x", "n_sup"],
          "RAR": ["cand_name_freq", "s1_name_freq", "grp_dup"],
          "NUM": ["num_jacc", "num_s1_in_c", "num_c_in_s1", "num_conflict", "n_num_s1", "n_num_c"],
          "QUAL": ["x_n", "x_lo_min", "x_lo_sum", "m_n", "m_lo_min", "m_lo_sum"]}


def main():
    OUTA.mkdir(parents=True, exist_ok=True)
    S.OUTD = OUTA
    allf = S.base_features()
    allf, folds, fit_ids = S.stage1(allf)
    d = S.stage2_features(allf)
    fit = allf.filter(pl.col("role") == "fit")
    fold_arr = np.empty(fit.height, np.int8)
    for i, (_, va) in enumerate(folds):
        fold_arr[va] = i
    fold_of = pl.DataFrame({"s1_id": fit["s1_id"], "fold": fold_arr}).unique("s1_id")
    t = E.tok_num(d, "train")
    q, _, _ = E.qualifier_features(t.select(*E.KEY, "X", "M"), d.select(*E.KEY, "is_match"), fold_of)
    d = d.join(t.drop("X", "M"), on=E.KEY, how="left").join(q, on=E.KEY, how="left")
    base_cols = pickle.load(open(CACHE / "calib.pkl", "rb"))["cols"]
    allg = [c for g in GROUPS.values() for c in g]
    variants = {"base (production model)": None, "all": base_cols + allg}
    for g in GROUPS:
        variants[f"all - {g}"] = base_cols + [c for k, v in GROUPS.items() if k != g for c in v]
    variants["base+NUM+QUAL (no stacking)"] = base_cols + GROUPS["NUM"] + GROUPS["QUAL"]
    variants["base+NUM only"] = base_cols + GROUPS["NUM"]
    variants["base+QUAL only"] = base_cols + GROUPS["QUAL"]
    preds = {}
    for name, cols in variants.items():
        if cols is None:
            preds[name] = d.select(*E.KEY, "role", "is_match", p="p1")
        else:
            log(f"training: {name}")
            preds[name] = S.train2(d, cols, folds, fit_ids, "abl_" + "".join(ch if ch.isalnum() else "_" for ch in name))
    S.evaluate(preds, ref="all")


if __name__ == "__main__":
    main()
