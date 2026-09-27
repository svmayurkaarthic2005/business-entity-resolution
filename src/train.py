"""Phase 5: pair classifier (GPU XGBoost, LightGBM fallback) + isotonic calibration on out-of-fold scores.

Usage:  python src/train.py            # 3-fold GroupKFold on role=fit rows -> models + calibrator, pred_TRAIN
        python src/train.py --loco     # robustness: train on US rows only, report AUC on India VAL rows
        python src/predict.py TEST     # apply saved models to features_TEST (see predict.py)
Training rows = every blocked candidate of a 15% sample of FIT S1 (all positives; negatives are the
hard ones blocking already kept). Label = pair in ground truth. Country is never a feature.
Output: cache/models/fold*.{json,txt}, cache/calib.pkl, cache/pred_TRAIN.parquet
"""
import argparse
import pickle

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import CACHE, log

DROP = {"s1_id", "cand_id", "role", "country", "is_match"}
MODELS = CACHE / "models"
FOLDS = 5


def feats(df):
    """Model input columns: every feature except ids/labels (country deliberately excluded)."""
    return [c for c in df.columns if c not in DROP]


def matrix(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def engine():
    """XGBoost on the GPU when CUDA is available, else LightGBM on CPU (both MIT/Apache-2.0)."""
    try:
        import xgboost as xgb
        xgb.train({"device": "cuda", "tree_method": "hist"}, xgb.DMatrix(np.zeros((2, 1)), label=[0, 1]), 1)
        return "xgb"
    except Exception:
        return "lgb"


def fit_one(eng, xtr, ytr, xva, yva, cols):
    """Train one model with early stopping on validation logloss; return (model, predict_fn)."""
    if eng == "xgb":
        import xgboost as xgb
        p = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist",
                 max_depth=12, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=3, reg_lambda=2.0, max_bin=256, seed=42)
        dtr, dva = xgb.DMatrix(xtr, ytr, feature_names=cols), xgb.DMatrix(xva, yva, feature_names=cols)
        m = xgb.train(p, dtr, 10000, evals=[(dva, "val")], early_stopping_rounds=300, verbose_eval=500)
        return m, lambda x: m.predict(xgb.DMatrix(x, feature_names=cols), iteration_range=(0, m.best_iteration + 1))
    import lightgbm as lgb
    p = dict(objective="binary", learning_rate=0.01, num_leaves=255, min_data_in_leaf=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, 
             device="gpu", verbose=-1, seed=42)
    m = lgb.train(p, lgb.Dataset(xtr, ytr, feature_name=cols), 10000,
                  valid_sets=[lgb.Dataset(xva, yva, feature_name=cols)],
                  callbacks=[lgb.early_stopping(300), lgb.log_evaluation(500)])
    return m, lambda x: m.predict(x, num_iteration=m.best_iteration)


def importances(eng, m, cols, k=30):
    if eng == "xgb":
        g = m.get_score(importance_type="gain")
        return sorted(((v, c) for c, v in g.items()), reverse=True)[:k]
    return sorted(zip(m.feature_importance("gain"), cols), reverse=True)[:k]


def save(eng, m, i):
    MODELS.mkdir(parents=True, exist_ok=True)
    if eng == "xgb":
        m.save_model(str(MODELS / f"fold{i}.json"))
    else:
        m.save_model(str(MODELS / f"fold{i}.txt"), num_iteration=m.best_iteration)


def train():
    f = pl.read_parquet(CACHE / "features_TRAIN.parquet")
    cols, eng = feats(f), engine()
    fit = f.filter(pl.col("role") == "fit")
    x, y, g = matrix(fit, cols), fit["is_match"].to_numpy(), fit["s1_id"].to_numpy()
    log(f"engine {eng}; {len(fit):,} training rows ({y.mean():.2%} positive), {len(cols)} features")
    oof, models = np.zeros(len(y), np.float32), []
    for i, (tr, va) in enumerate(GroupKFold(FOLDS).split(x, y, g)):
        m, pred = fit_one(eng, x[tr], y[tr], x[va], y[va], cols)
        oof[va] = pred(x[va])
        models.append(pred)
        save(eng, m, i)
        log(f"fold {i}: AUC {roc_auc_score(y[va], oof[va]):.5f} logloss {log_loss(y[va], oof[va]):.5f}")
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    pickle.dump({"iso": iso, "cols": cols, "engine": eng, "folds": FOLDS}, open(CACHE / "calib.pkl", "wb"))
    print(f"OOF AUC {roc_auc_score(y, oof):.5f} | logloss raw {log_loss(y, oof):.5f} "
          f"calibrated {log_loss(y, iso.predict(oof).clip(1e-6, 1 - 1e-6)):.5f}")
    print("top-30 features by gain:", [(c, round(float(v))) for v, c in importances(eng, m, cols)])
    val = f.filter(pl.col("role") == "val")
    pv = np.mean([p(matrix(val, cols)) for p in models], axis=0)
    log(f"VAL AUC {roc_auc_score(val['is_match'], pv):.5f} logloss {log_loss(val['is_match'], iso.predict(pv).clip(1e-6, 1 - 1e-6)):.5f}")
    from predict import predict
    predict("TRAIN", f)


def loco():
    """Leave-one-country-out: train on US FIT rows, evaluate on India VAL rows (proxy for unseen France)."""
    f = pl.read_parquet(CACHE / "features_TRAIN.parquet")
    cols, eng = feats(f), engine()
    us = f.filter((pl.col("role") == "fit") & (pl.col("country") == "US"))
    va_all = f.filter(pl.col("role") == "val")
    ids = us["s1_id"].unique().sample(fraction=0.1, seed=1)
    tr, es = us.filter(~pl.col("s1_id").is_in(ids)), us.filter(pl.col("s1_id").is_in(ids))
    _, pred = fit_one(eng, matrix(tr, cols), tr["is_match"].to_numpy(), matrix(es, cols), es["is_match"].to_numpy(), cols)
    for c in ("US", "India"):
        v = va_all.filter(pl.col("country") == c)
        print(f"LOCO (trained on US only) -> VAL {c}: AUC {roc_auc_score(v['is_match'], pred(matrix(v, cols))):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--predict", default=None, help="alias for predict.py <split>")
    a = ap.parse_args()
    if a.predict:
        from predict import predict
        predict(a.predict)
    else:
        loco() if a.loco else train()
