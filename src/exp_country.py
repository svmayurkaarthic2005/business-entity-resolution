"""Experiment: country-aware matcher (adds an is_india flag) vs the current model, on the labelled VAL split.

Usage: python src/exp_country.py      (GPU XGBoost, 3-fold GroupKFold; saves only into exp/country/)
Honest comparison: for each model the decision rule is chosen on one random half of the VAL S1 and scored on
the other half (20 repeats), plus the score with the current production rule.
France is unseen: at test time it would get is_india=0 (US-like), so a VAL gain does not transfer to France.
"""
import itertools
import json

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from common import CACHE, log
from decide import choose, owner_adjust
from exp_budget import val_truth
from exp_decide import per_s1
from train import engine, feats, fit_one, matrix

OUTD = CACHE.parent / "exp" / "country"
GRID = dict(alpha=(0.3, 0.5, 0.7), miss=(0.4, 0.6, 0.8), gamma=(1.5, 2.0, 3.0, 4.0), floor=(0.1, 0.15))


def scores(pred, truth_n):
    """per-S1 F for every decision config (owner exclusivity over all rows, as in decide.tune)."""
    out = {}
    pa = {a: owner_adjust(pred, a).filter(pl.col("role") == "val") for a in GRID["alpha"]}
    for cfg in itertools.product(*GRID.values()):
        out[cfg] = per_s1(choose(pa[cfg[0]], *cfg[1:]).filter("take"), truth_n)
    return out


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    f = pl.read_parquet(CACHE / "features_TRAIN.parquet").with_columns(is_india=(pl.col("country") == "India").cast(pl.Float32))
    cols = feats(f)
    assert "is_india" in cols
    fit = f.filter(pl.col("role") == "fit")
    x, y, g = matrix(fit, cols), fit["is_match"].to_numpy(), fit["s1_id"].to_numpy()
    eng = engine()
    log(f"engine {eng}; {len(y):,} rows, {len(cols)} features (+is_india)")
    oof, models = np.zeros(len(y), np.float32), []
    for i, (tr, va) in enumerate(GroupKFold(3).split(x, y, g)):
        m, pred = fit_one(eng, x[tr], y[tr], x[va], y[va], cols)
        oof[va] = pred(x[va])
        models.append(pred)
        m.save_model(str(OUTD / f"fold{i}.{'json' if eng == 'xgb' else 'txt'}"))
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    del x
    xa = matrix(f, cols)
    p = iso.predict(np.mean([m(xa) for m in models], axis=0)).astype(np.float32)
    del xa
    new = f.select("s1_id", "cand_id", "source", "role", "is_match").with_columns(p=pl.Series(p))
    new.write_parquet(OUTD / "pred_TRAIN_country.parquet")
    old = pl.read_parquet(CACHE / "pred_TRAIN.parquet")

    _, truth_n, _ = val_truth()
    cur = json.load(open(CACHE / "decide.json"))["cfg"]
    key_cur = (cur["alpha"], cur["miss"], cur["gamma"], cur["floor"])
    ctry = pl.read_parquet(CACHE / "norm" / "train_source1.parquet", columns=["id", "country"]).rename({"id": "s1_id"})
    india = truth_n.join(ctry, on="s1_id")["country"].to_numpy() == "India"
    res = {}
    for name, pred in (("current model", old), ("country-aware", new)):
        s = scores(pred, truth_n)
        best = max(s, key=lambda k: s[k][0].mean())
        fc, pc, rc, kc = s[key_cur]
        print(f"{name:14s} production rule {key_cur}: F {fc.mean():.5f} P {pc.mean():.4f} R {rc.mean():.4f} pred/S1 {kc.mean():.3f} "
              f"| India {fc[india].mean():.5f} US {fc[~india].mean():.5f} | best rule {best} F {s[best][0].mean():.5f}")
        res[name] = s
    gains = []
    for seed in range(10):
        m = np.random.default_rng(seed).random(len(truth_n)) < 0.5
        for A, B in ((m, ~m), (~m, m)):
            pick = {n: max(res[n], key=lambda k: res[n][k][0][A].mean()) for n in res}
            gains.append(res["country-aware"][pick["country-aware"]][0][B].mean() - res["current model"][pick["current model"]][0][B].mean())
    g = np.array(gains)
    d = res["country-aware"][key_cur][0] - res["current model"][key_cur][0]
    print(f"held-out gain of country-aware model (rule re-chosen on the other half): mean {g.mean():+.5f} "
          f"min {g.min():+.5f} max {g.max():+.5f}; paired diff at production rule {d.mean():+.5f} ± {d.std(ddof=1) / np.sqrt(len(d)):.5f}")


if __name__ == "__main__":
    main()
