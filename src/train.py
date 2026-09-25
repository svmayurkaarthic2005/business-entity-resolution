"""Phase 5: LightGBM pair classifier + isotonic calibration.

Usage:  python src/train.py                  # train on features_TRAIN role=fit, predict every TRAIN row
        python src/train.py --predict TEST   # apply the saved model to features_TEST
Output: cache/model.txt, cache/calib.pkl, cache/pred_<split>.parquet (s1_id, cand_id, source, role, p)
"""
import argparse
import pickle

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from common import CACHE, log

DROP = {"s1_id", "cand_id", "role", "country", "is_match", "rank"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=42, num_threads=0)


def feats(df):
    """Model input columns: every feature except ids/labels (country deliberately excluded)."""
    return [c for c in df.columns if c not in DROP]


def matrix(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def train():
    f = pl.read_parquet(CACHE / "features_TRAIN.parquet")
    cols = feats(f)
    fit = f.filter(pl.col("role") == "fit")
    ids = fit["s1_id"].unique()
    es_ids = ids.sample(fraction=0.1, seed=42)           # early stopping + calibration holdout
    tr, es = fit.filter(~pl.col("s1_id").is_in(es_ids)), fit.filter(pl.col("s1_id").is_in(es_ids))
    log(f"train {len(tr):,} rows ({tr['is_match'].mean():.2%} pos), holdout {len(es):,}; {len(cols)} features")
    dtr = lgb.Dataset(matrix(tr, cols), tr["is_match"].to_numpy(), feature_name=cols)
    des = lgb.Dataset(matrix(es, cols), es["is_match"].to_numpy(), reference=dtr)
    m = lgb.train(PARAMS, dtr, 3000, valid_sets=[des],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    raw_es = m.predict(matrix(es, cols), num_iteration=m.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_es, es["is_match"].to_numpy())
    m.save_model(str(CACHE / "model.txt"), num_iteration=m.best_iteration)
    pickle.dump({"iso": iso, "cols": cols}, open(CACHE / "calib.pkl", "wb"))
    imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)[:25]
    print("top features by gain:", [(c, int(g)) for g, c in imp])
    val = f.filter(pl.col("role") == "val")
    pv = m.predict(matrix(val, cols), num_iteration=m.best_iteration)
    log(f"holdout AUC {roc_auc_score(es['is_match'], raw_es):.5f} | VAL AUC {roc_auc_score(val['is_match'], pv):.5f}")
    predict("TRAIN", f)


def predict(split, f=None):
    """Calibrated probabilities for every row of features_<split>."""
    f = f if f is not None else pl.read_parquet(CACHE / f"features_{split}.parquet")
    m = lgb.Booster(model_file=str(CACHE / "model.txt"))
    c = pickle.load(open(CACHE / "calib.pkl", "rb"))
    p = c["iso"].predict(m.predict(matrix(f, c["cols"])))
    keep = [x for x in ("s1_id", "cand_id", "source", "role", "is_match") if x in f.columns]
    out = f.select(keep).with_columns(p=pl.Series(p.astype(np.float32)))
    out.write_parquet(CACHE / f"pred_{split}.parquet")
    log(f"wrote pred_{split}.parquet: {len(out):,} rows, mean p {out['p'].mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", default=None, help="split to predict with the saved model")
    a = ap.parse_args()
    predict(a.predict) if a.predict else train()
