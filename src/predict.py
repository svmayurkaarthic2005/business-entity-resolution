"""Apply the saved fold models + isotonic calibrator to features_<split>.parquet.

Usage:  python src/predict.py TEST
Output: cache/pred_<split>.parquet  (s1_id, cand_id, source, [role, is_match], p)
"""
import pickle
import sys

import numpy as np
import polars as pl

from common import CACHE, log

MODELS = CACHE / "models"


def load_models(c):
    """Predict functions of every saved fold model (probabilities are averaged)."""
    if c["engine"] == "xgb":
        import xgboost as xgb
        out = []
        for i in range(c["folds"]):
            m = xgb.Booster(model_file=str(MODELS / f"fold{i}.json"))
            out.append(lambda x, m=m: m.predict(xgb.DMatrix(x, feature_names=c["cols"])))
        return out
    import lightgbm as lgb
    return [lgb.Booster(model_file=str(MODELS / f"fold{i}.txt")).predict for i in range(c["folds"])]


def predict(split, f=None):
    """Calibrated match probability for every candidate row of the split."""
    f = f if f is not None else pl.read_parquet(CACHE / f"features_{split}.parquet")
    c = pickle.load(open(CACHE / "calib.pkl", "rb"))
    x = f.select([pl.col(k).cast(pl.Float32) for k in c["cols"]]).to_numpy()
    raw = np.mean([p(x) for p in load_models(c)], axis=0)
    keep = [k for k in ("s1_id", "cand_id", "source", "role", "is_match") if k in f.columns]
    out = f.select(keep).with_columns(p=pl.Series(c["iso"].predict(raw).astype(np.float32)))
    out.write_parquet(CACHE / f"pred_{split}.parquet")
    log(f"wrote pred_{split}.parquet: {len(out):,} rows, mean p {out['p'].mean():.4f}")


if __name__ == "__main__":
    predict(sys.argv[1] if len(sys.argv) > 1 else "TEST")
