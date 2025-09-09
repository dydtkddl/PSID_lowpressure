# -*- coding: utf-8 -*-
"""
Full modular pipeline (single-trial runner) + NN option:
- Preprocess
- Sampling
- Models: RF / GBM / CatBoost / NN(2-layer MLP)
- Training (holdout only)
"""

from __future__ import annotations
import os, sys, time, json, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Optional CatBoost
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False
# ───────────────────────── PyTorch NN 정의 ─────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, output_dim: int = 1):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_torch_nn(X_tr, y_tr, X_te, y_te,
                   hidden1: int, hidden2: int,
                   lr: float, epochs: int, batch_size: int,
                   seed: int, logger: logging.Logger,
                   patience: int = 20, delta: float = 1e-4, val_ratio: float = 0.1):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleMLP(X_tr.shape[1], hidden1, hidden2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── Train/Val split ──
    n_val = int(len(X_tr) * val_ratio)
    if n_val > 0:
        X_val, y_val = X_tr.iloc[:n_val], y_tr.iloc[:n_val]
        X_train, y_train = X_tr.iloc[n_val:], y_tr.iloc[n_val:]
    else:
        X_val, y_val = None, None
        X_train, y_train = X_tr, y_tr

    train_ds = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                             torch.tensor(y_train.values, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if X_val is not None:
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).to(device)

    X_te_t = torch.tensor(X_te.values, dtype=torch.float32).to(device)
    y_te_t = torch.tensor(y_te.values, dtype=torch.float32).to(device)

    # ── Early stopping variables ──
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    t0 = time.time()
    from tqdm import trange
    for epoch in trange(epochs, desc="Training NN"):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)

        # Validation loss check
        val_loss = None
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            if val_loss < best_loss - delta:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[EarlyStopping] Epoch {epoch}: val_loss did not improve for {patience} rounds. Stop training.")
                break

        # ── 여기서 loss 출력 ──
        if val_loss is not None:
            logger.info(f"[Epoch {epoch}] Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")
        else:
            logger.info(f"[Epoch {epoch}] Train Loss={train_loss:.6f}")

    fit_time = time.time() - t0

    # Restore best state if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test evaluation ──
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()
    pred_time = time.time() - t1

    y_true = y_te.values
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))).mean() * 100.0

    logger.info(f"[Metrics-NN] R2={r2:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f} | MAPE%={mape:.3f}")
    return model, dict(R2=r2, MAE=mae, RMSE=rmse, MAPE_percent=mape,
                       fit_time_sec=fit_time, predict_time_sec=pred_time), y_true, y_pred

# ───────────────────────── Logger ─────────────────────────
def setup_logger(outdir: Optional[Union[str, Path]] = None,
                 name: str = "Trainer",
                 level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handlers = [logging.StreamHandler(sys.stdout)]
    if outdir is not None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        log_path = outdir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt)
        handlers.append(fh)
    for h in handlers:
        h.setFormatter(fmt); logger.addHandler(h)
    return logger

# ───────────────────────── Preprocess / Features ─────────────────────────
def preprocess_has_oms(df: pd.DataFrame, col: str = "Has_OMS") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
    elif not np.issubdtype(df[col].dtype, np.number):
        df[col] = (df[col].astype(str).str.strip().str.lower()
                   .map({"true": 1, "false": 0}).fillna(0).astype(int))
    return df

def select_struct_features(df: pd.DataFrame,
                           features: Optional[List[str]] = None) -> List[str]:
    if features is not None:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        return features
    default = [
        "LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_cm3", "ASA_m2_g",
        "NASA_m2_cm3", "NASA_m2_g", "AV_VF", "AV_cm3_g", "NAV_cm3_g", "Has_OMS"
    ]
    missing = [c for c in default if c not in df.columns]
    if missing:
        raise KeyError(f"Missing structural features: {missing}")
    return default

def dropna_for_training(df: pd.DataFrame, features: List[str], target: str,
                        logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    before = len(df)
    out = df.dropna(subset=features + [target]).copy()
    if logger: logger.info(f"[Preprocess] Drop NaN rows (FEATURES+TARGET): {before} -> {len(out)}")
    return out

def build_X_y_meta(df: pd.DataFrame, features: List[str], target: str,
                   meta_keep: Optional[List[str]] = None
                   ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    X = df[features].copy()
    y = pd.to_numeric(df[target], errors="coerce").astype(float)
    meta = df[meta_keep].copy() if meta_keep else None
    return X, y, meta

# ───────────────────────── Sampling utils ─────────────────────────
def _split_indices_random(all_idx: np.ndarray,
                          train_ratio: float,
                          seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = len(all_idx)
    n_train = int(round(train_ratio * n_total))
    train_idx = rng.choice(all_idx, size=n_train, replace=False)
    test_idx  = np.setdiff1d(all_idx, train_idx, assume_unique=False)
    return train_idx, test_idx

def _make_log10_shift(series: pd.Series, eps: float = 1e-12) -> Tuple[pd.Series, float]:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        raise ValueError("Series contains non-numeric values.")
    min_val = float(s.min())
    shift = -min_val + eps if min_val <= 0 else 0.0
    return np.log10(s + shift), shift

def quantile_weighted_sample(series_for_binning: pd.Series,
                             n_bins: int,
                             gamma: float,
                             n_samples: int,
                             seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = series_for_binning.values.astype(float)
    idx  = series_for_binning.index.values
    n    = len(idx)
    if n == 0 or n_samples <= 0:
        return np.array([], dtype=idx.dtype)

    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        return rng.choice(idx, size=min(n_samples, n), replace=False)

    edges   = np.linspace(vmin, vmax, n_bins + 1)
    bin_ids = np.digitize(vals, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_to_idx = {b: idx[bin_ids == b] for b in range(n_bins)}
    counts     = np.array([len(bin_to_idx[b]) for b in range(n_bins)], dtype=float)

    valid = np.where(counts > 0)[0]
    weights = np.zeros_like(counts)
    weights[valid] = counts[valid] ** gamma
    wsum = weights[valid].sum()
    probs = weights / (wsum if wsum > 0 else 1.0)

    raw   = probs * n_samples
    quota = np.floor(raw).astype(int)
    deficit = int(n_samples - quota.sum())
    if deficit > 0:
        frac = raw - quota
        fsum = frac.sum()
        if fsum > 0:
            add_bins = rng.choice(np.arange(n_bins), size=deficit, replace=True, p=frac / fsum)
        else:
            add_bins = rng.choice(valid, size=deficit, replace=True)
        for b in add_bins:
            quota[b] += 1

    selected = []
    for b in range(n_bins):
        k = int(quota[b])
        if k <= 0: continue
        pool = bin_to_idx.get(b, np.array([], dtype=idx.dtype))
        if len(pool) == 0: continue
        k = min(k, len(pool))
        chosen = rng.choice(pool, size=k, replace=False)
        selected.append(chosen)

    if not selected:
        return np.array([], dtype=idx.dtype)
    sel = np.concatenate(selected)

    if len(sel) > n_samples:
        sel = sel[:n_samples]
    elif len(sel) < n_samples:
        remain = np.setdiff1d(idx, sel, assume_unique=False)
        if len(remain) > 0:
            add = rng.choice(remain, size=min(n_samples - len(sel), len(remain)), replace=False)
            sel = np.concatenate([sel, add])
    return sel

# ───────────────────────── 3 Samplers ─────────────────────────
def sample_random_struct_only(df: pd.DataFrame,
                              train_ratio: float = 0.8,
                              seed: int = 42,
                              logger: Optional[logging.Logger] = None
                              ) -> Dict[str, object]:
    all_idx = df.index.values
    train_idx, test_idx = _split_indices_random(all_idx, train_ratio, seed)
    if logger:
        logger.info(f"[S1 Random-Struct] train={len(train_idx)} / test={len(test_idx)} (ratio={train_ratio:.2f})")
    return {
        "train_idx": train_idx,
        "test_idx":  test_idx,
        "train_qt_idx": None,
        "train_rd_idx": train_idx,
        "info": {"strategy": "random_struct_only", "train_ratio": train_ratio, "seed": seed}
    }

def sample_random_with_input(df: pd.DataFrame,
                             train_ratio: float = 0.8,
                             seed: int = 42,
                             logger: Optional[logging.Logger] = None
                             ) -> Dict[str, object]:
    return sample_random_struct_only(df, train_ratio=train_ratio, seed=seed, logger=logger)

def sample_quantile_then_random_on_input(df: pd.DataFrame,
                                         input_col: str = "Input",
                                         train_ratio: float = 0.8,
                                         qt_frac_total: float = 0.4,
                                         n_bins: int = 10,
                                         gamma: float = 0.5,
                                         seed_qt: int = 123,
                                         seed_rd: int = 456,
                                         logger: Optional[logging.Logger] = None
                                         ) -> Dict[str, object]:
    log_in, shift_in = _make_log10_shift(df[input_col])
    all_idx  = df.index.values
    n_total  = len(all_idx)
    n_qt     = int(round(qt_frac_total * n_total))
    n_train_target = int(round(train_ratio * n_total))
    n_rd     = max(n_train_target - n_qt, 0)
    if n_qt > n_train_target:
        n_qt = n_train_target; n_rd = 0

    qt_idx = quantile_weighted_sample(log_in, n_bins=n_bins, gamma=gamma,
                                      n_samples=n_qt, seed=seed_qt)
    remain = np.setdiff1d(all_idx, qt_idx, assume_unique=False)
    rng = np.random.default_rng(seed_rd)
    rd_idx = rng.choice(remain, size=min(n_rd, len(remain)), replace=False)
    te_idx = np.setdiff1d(remain, rd_idx, assume_unique=False)

    if logger:
        logger.info(f"[S3 QT→RD] train(qt)={len(qt_idx)}, train(rd)={len(rd_idx)}, test={len(te_idx)} "
                    f"(train_ratio={train_ratio:.2f}, qt_frac_total={qt_frac_total:.2f}, shift={shift_in:.3e})")

    train_idx = np.concatenate([qt_idx, rd_idx]) if len(rd_idx) else qt_idx
    return {"train_idx": train_idx, "test_idx": te_idx,
            "train_qt_idx": qt_idx, "train_rd_idx": rd_idx,
            "info": {"strategy": "quantile_then_random_on_input",
                     "train_ratio": train_ratio, "qt_frac_total": qt_frac_total,
                     "n_bins": n_bins, "gamma": gamma,
                     "seed_qt": seed_qt, "seed_rd": seed_rd,
                     "shift_in": shift_in}}

# ───────────────────────── Model factory ─────────────────────────
def build_regressor(kind: str,
                    params: Optional[Dict] = None,
                    logger: Optional[logging.Logger] = None):
    params = dict(params or {}); kind = kind.lower().strip()
    if kind == "rf":
        defaults = dict(n_estimators=800, max_depth=None, n_jobs=-1,
                        random_state=42, oob_score=False)
        defaults.update(params)
        model = RandomForestRegressor(**defaults)
    elif kind == "gbm":
        defaults = dict(n_estimators=500, learning_rate=0.05,
                        max_depth=5, random_state=42)
        defaults.update(params)
        model = GradientBoostingRegressor(**defaults)
    elif kind == "cat":
        if not _HAS_CATBOOST: raise ImportError("CatBoost 설치 필요: pip install catboost")
        defaults = dict(iterations=500, depth=8, learning_rate=0.05,
                        loss_function="RMSE", random_seed=42, verbose=False)
        defaults.update(params)
        model = CatBoostRegressor(**defaults)
    elif kind == "nn":
        # 2-layer NN: hidden_layer_sizes=(64,32)
        defaults = dict(hidden_layer_sizes=(64,32), activation="relu",
                        solver="adam", learning_rate_init=1e-3,
                        max_iter=500, random_state=42, verbose=False)
        defaults.update(params)
        model = MLPRegressor(**defaults)
    else:
        raise ValueError(f"Unsupported model kind='{kind}'. Choose from rf|gbm|cat|nn.")
    if logger:
        if hasattr(model, "get_params"): logger.info(f"[Model] kind={kind} | params={model.get_params()}")
        else: logger.info(f"[Model] kind={kind} | params={params}")
    return model

# ───────────────────────── Importance util ─────────────────────────
def get_feature_importance(model, X_test: pd.DataFrame,
                           y_test: pd.Series,
                           logger: Optional[logging.Logger] = None,
                           use_permutation_if_missing: bool = True,
                           n_repeats: int = 10,
                           random_state: int = 42) -> pd.Series:
    try:
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=X_test.columns, name="importance").sort_values(ascending=False)
        if _HAS_CATBOOST and isinstance(model, CatBoostRegressor):
            vals = model.get_feature_importance(type="PredictionValuesChange")
            return pd.Series(vals, index=X_test.columns, name="importance").sort_values(ascending=False)
        if use_permutation_if_missing:
            if logger: logger.info("[Importance] Using permutation_importance")
            r = permutation_importance(model, X_test, y_test,
                                       n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
            return pd.Series(r.importances_mean, index=X_test.columns, name="importance").sort_values(ascending=False)
    except Exception as e:
        if logger: logger.warning(f"[Importance] fallback failed: {e}")
    return pd.Series(np.zeros(X_test.shape[1]), index=X_test.columns, name="importance")

# ───────────────────────── Training (Holdout) ─────────────────────────
def train_holdout_general(X: pd.DataFrame, y: pd.Series,
                          train_idx: np.ndarray, test_idx: np.ndarray,
                          model_kind: str = "rf",
                          model_params: Optional[Dict] = None,
                          meta: Optional[pd.DataFrame] = None,
                          outdir: Optional[Union[str, Path]] = None,
                          logger: Optional[logging.Logger] = None,
                          save_predictions: bool = True,
                          trial_id: Optional[int] = None) -> Dict[str, object]:
    if logger is None: logger = setup_logger(outdir=outdir, name="Holdout", level=logging.INFO)
    outdir = Path(outdir) if outdir is not None else None
    if outdir is not None: outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_trial{trial_id:03d}" if isinstance(trial_id, int) else ""
    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    meta_te = meta.loc[test_idx].reset_index(drop=True) if meta is not None else None
    if model_kind == "nn":
        # --- 스케일링 추가 ---
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(
            scaler.fit_transform(X_tr),
            index=X_tr.index,
            columns=X_tr.columns
        )
        X_te_scaled = pd.DataFrame(
            scaler.transform(X_te),
            index=X_te.index,
            columns=X_te.columns
        )
        # ---------------------

        hidden1 = model_params.get("hidden1", 64)
        hidden2 = model_params.get("hidden2", 32)
        lr = model_params.get("lr", 1e-3)
        epochs = model_params.get("epochs", 500)
        batch_size = model_params.get("batch_size", 64)
        seed = model_params.get("random_state", 42)

        model, metrics, y_true, y_pred = train_torch_nn(
            X_tr_scaled, y_tr, X_te_scaled, y_te,
            hidden1, hidden2, lr, epochs, batch_size, seed, logger
        )

        # --- 성능 지표 저장 ---
        fit_time = metrics["fit_time_sec"]
        pred_time = metrics["predict_time_sec"]
        r2 = metrics["R2"]
        mae = metrics["MAE"]
        rmse = metrics["RMSE"]
        mape = metrics["MAPE_percent"]
        # ---------------------
        imp = pd.Series(np.zeros(X_tr.shape[1]), index=X_tr.columns, name="importance")

    else:
        model = build_regressor(model_kind, model_params, logger=logger)

        logger.info(f"[Holdout{suffix}] Train={len(X_tr)} | Test={len(X_te)}")
        t0 = time.time(); model.fit(X_tr, y_tr); fit_time = time.time() - t0
        logger.info(f"[Timing{suffix}] fit_time={fit_time:.3f}s")
        t1 = time.time(); y_pred = model.predict(X_te); pred_time = time.time() - t1
        logger.info(f"[Timing{suffix}] predict_time={pred_time:.3f}s")

        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = mean_squared_error(y_te, y_pred)
        mape = (np.abs((y_te - y_pred) / np.clip(np.abs(y_te), 1e-12, None))).mean() * 100.0
        logger.info(f"[Metrics{suffix}] R2={r2:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f} | MAPE%={mape:.3f}")

        imp = get_feature_importance(model, X_te, y_te, logger=logger)

    if outdir is not None:
        if save_predictions:
            pred_df = X_te.reset_index(drop=True).copy()
            pred_df["y_true"] = y_te.values; pred_df["y_pred"] = y_pred
            if meta_te is not None: pred_df = pd.concat([meta_te, pred_df], axis=1)
            pred_df.to_csv(outdir / f"predictions_holdout{suffix}.csv", index=False, encoding="utf-8-sig")
            logger.info(f"[Save] predictions_holdout{suffix}.csv")
        pd.DataFrame([{"Model": model_kind, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_percent": mape,
                       "n_train": len(X_tr), "n_test": len(X_te),
                       "fit_time_sec": fit_time, "predict_time_sec": pred_time}]).to_csv(
                           outdir / f"metrics_holdout{suffix}.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[Save] metrics_holdout{suffix}.csv")
        imp.to_csv(outdir / f"feature_importances_holdout{suffix}.csv", encoding="utf-8-sig")
        logger.info(f"[Save] feature_importances_holdout{suffix}.csv")
        params_path = outdir / f"model_params{suffix}.json"
        try:
            if hasattr(model, "get_params"):
                with open(params_path, "w", encoding="utf-8") as f:
                    json.dump(model.get_params(), f, ensure_ascii=False, indent=2)
            else:
                with open(params_path, "w", encoding="utf-8") as f:
                    json.dump(model_params or {}, f, ensure_ascii=False, indent=2)
            logger.info(f"[Save] model_params{suffix}.json")
        except Exception: pass
    return {"model": model,
            "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_percent": mape,
                        "n_train": len(X_tr), "n_test": len(X_te),
                        "fit_time_sec": fit_time, "predict_time_sec": pred_time},
            "y_true": y_te.values, "y_pred": y_pred,
            "feature_importances_": imp}

# ───────────────────────── Trial seed derivation ─────────────────────────
def _derive_trial_seeds(seed_base: int, trial: int) -> Dict[str, int]:
    return {"seed": seed_base + trial,
            "seed_qt": seed_base + 1000*trial,
            "seed_rd": seed_base + 2000*trial,
            "model_seed": seed_base + 3000*trial}

# ───────────────────────── CLI / Main ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline (single trial)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./RUN_OUT")
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--target", type=str, default="Output")
    p.add_argument("--meta", type=str, nargs="*", default=["filename","name"])
    p.add_argument("--features", type=str, choices=["struct", "struct+input", "custom"], default="struct")
    p.add_argument("--custom-features", type=str, default="")
    p.add_argument("--sampler", type=str,
                   choices=["random_struct", "random_with_input", "qt_then_rd"], default="qt_then_rd")
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--qt-frac", type=float, default=0.40)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--model", type=str, choices=["rf","gbm","cat","nn"], default="rf")
    # NN params
    p.add_argument("--nn-hidden1", type=int, default=64)
    p.add_argument("--nn-hidden2", type=int, default=32)
    p.add_argument("--nn-lr", type=float, default=1e-3)
    p.add_argument("--nn-epochs", type=int, default=500)
    p.add_argument("--nn-batch-size", type=int, default=64)

    return p.parse_args()

def main():
    args = parse_args()
    OUTROOT = Path(args.outdir); OUTROOT.mkdir(parents=True, exist_ok=True)
    trial_dir = OUTROOT / f"trial_{args.trial:03d}"; trial_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir=trial_dir, name=f"Pipeline_trial{args.trial:03d}", level=logging.INFO)
    df = pd.read_csv(args.data)
    logger.info(f"[Data] Loaded: {args.data} | rows={len(df)} | cols={len(df.columns)}")
    df = preprocess_has_oms(df); TARGET = args.target; META_KEEP = args.meta
    if args.features == "struct": FEATURES = select_struct_features(df)
    elif args.features == "struct+input": FEATURES = select_struct_features(df)+["Input"]
    else:
        if not args.custom_features: raise ValueError("Provide --custom-features")
        FEATURES = [c.strip() for c in args.custom_features.split(",") if c.strip()]
        FEATURES = select_struct_features(df, features=FEATURES)
    logger.info(f"[Features] Using {len(FEATURES)} features: {FEATURES}")
    df_train = dropna_for_training(df, FEATURES, TARGET, logger)
    X, y, meta = build_X_y_meta(df_train, FEATURES, TARGET, META_KEEP)
    seeds = _derive_trial_seeds(args.seed_base, args.trial)
    logger.info(f"[Seeds] trial={args.trial} | seeds={seeds}")
    if args.sampler == "random_struct":
        sample = sample_random_struct_only(df_train, train_ratio=args.train_ratio, seed=seeds["seed"], logger=logger)
    elif args.sampler == "random_with_input":
        if "Input" not in df_train.columns: raise KeyError("Input column not found")
        sample = sample_random_with_input(df_train, train_ratio=args.train_ratio, seed=seeds["seed"], logger=logger)
    else:
        if "Input" not in df_train.columns: raise KeyError("Input column not found")
        sample = sample_quantile_then_random_on_input(df_train, input_col="Input",
                    train_ratio=args.train_ratio, qt_frac_total=args.qt_frac,
                    n_bins=args.n_bins, gamma=args.gamma,
                    seed_qt=seeds["seed_qt"], seed_rd=seeds["seed_rd"], logger=logger)
    suffix = f"_trial{args.trial:03d}"
    pd.DataFrame({"train_idx": pd.Series(sample["train_idx"])}).to_csv(trial_dir/f"train_indices{suffix}.csv", index=False)
    pd.DataFrame({"test_idx": pd.Series(sample["test_idx"])}).to_csv(trial_dir/f"test_indices{suffix}.csv", index=False)
    with open(trial_dir/f"sampling_info{suffix}.json","w",encoding="utf-8") as f: json.dump(sample["info"], f, indent=2)

    model_params: Dict[str, Union[int, float]] = {}
    if args.model == "rf": model_params["random_state"] = seeds["model_seed"]
    elif args.model == "gbm": model_params["random_state"] = seeds["model_seed"]
    elif args.model == "cat": model_params["random_seed"] = seeds["model_seed"]
    elif args.model == "nn":
        model_params.update(dict(hidden1=args.nn_hidden1,
                                hidden2=args.nn_hidden2,
                                lr=args.nn_lr,
                                epochs=args.nn_epochs,
                                batch_size=args.nn_batch_size,
                                random_state=seeds["model_seed"]))
    _ = train_holdout_general(X=X, y=y,
        train_idx=sample["train_idx"], test_idx=sample["test_idx"],
        model_kind=args.model, model_params=model_params,
        meta=meta, outdir=trial_dir, logger=logger,
        save_predictions=True, trial_id=args.trial)

    logger.info(f"[Done] trial={args.trial} | outputs at: {trial_dir}")

if __name__ == "__main__":
    main()
