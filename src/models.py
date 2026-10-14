import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch

def get_device():
    """Detect whether CUDA GPU is available."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train gradient-boosted decision trees using XGBoost."""
    device = get_device()
    scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos,
        'tree_method': 'hist',
        'device': device,
        'seed': 42
    }
    if params:
        default_params.update(params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)

    print(f"🚀 Training XGBoost on {device.upper()}...")
    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=40,
        verbose_eval=100
    )
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train gradient-boosted decision trees using LightGBM."""
    scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    default_params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'max_depth': 8,
        'learning_rate': 0.05,
        'scale_pos_weight': scale_pos,
        'seed': 42,
        'verbose': -1
    }
    if params:
        default_params.update(params)

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    print("🚀 Training LightGBM...")
    model = lgb.train(
        default_params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    return model

def predict_ensemble(xgb_model, lgb_model, X_df):
    """Predict match probabilities by averaging the XGBoost and LightGBM outputs (equal weights)."""
    dmat = xgb.DMatrix(X_df)
    p_xgb = xgb_model.predict(dmat)
    p_lgb = lgb_model.predict(X_df)
    return 0.5 * p_xgb + 0.5 * p_lgb
