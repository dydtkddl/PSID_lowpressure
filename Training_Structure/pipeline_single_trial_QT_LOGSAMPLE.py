# -*- coding: utf-8 -*-
"""
Full modular pipeline (single-trial runner):
- Preprocess: Has_OMS normalization, feature selection, NA drop
- Sampling (same return interface):
    1) sample_random_struct_only
    2) sample_random_with_input
    3) sample_quantile_then_random_on_input
- Models: RF / GBM / CatBoost (selectable)
- Training (holdout only): train_holdout_general
- Single trial controlled by --trial (int), seeds derived from --seed-base and trial index
- All outputs saved under: outdir/trial_{trial:03d}/... and filenames suffixed with _trial{trial:03d}

Example:
python pipeline_single_trial.py \
  --data ../Data_collect/DataSet/Ar_273K/Ar_273K_Ar_273_0.01_to_Ar_273_15_dataset.csv \
  --outdir ./RUN_OUT \
  --trial 1 --seed-base 52 \
  --features struct+input \
  --sampler qt_then_rd --train-ratio 0.8 --qt-frac 0.4 --n-bins 10 --gamma 0.5 \
  --model rf --rf-n-estimators 800
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional CatBoost
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

import matplotlib.pyplot as plt

def save_sampling_plots(df: pd.DataFrame,
                        input_col: str,
                        output_col: str,
                        train_idx: np.ndarray,
                        qt_idx: Optional[np.ndarray],
                        outdir: Path,
                        trial_id: int,
                        n_bins : int):
    """
    Save scatter and histogram with sampling fractions.
    """
    suffix = f"_trial{trial_id:03d}"
    sel_df = df.loc[train_idx]
    all_df = df

    # ------------------- Scatter plot -------------------
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(all_df[input_col], all_df[output_col],
               s=10, color="lightgray", label="All data")
    ax.scatter(sel_df[input_col], sel_df[output_col],
               s=15, color="red", label="Sampled")
    ax.set_xlabel(input_col)
    ax.set_ylabel(output_col)
    ax.legend()
    ax.set_title(f"Scatter (trial {trial_id})")
    fig.tight_layout()
    fig.savefig(outdir / f"scatter_sampling{suffix}.png", dpi=200)
    plt.close(fig)

    # ------------------- Histogram (bin fraction) -------------------
    if qt_idx is not None and len(qt_idx) > 0:
        vals = np.log10(np.clip(df[input_col].values, a_min=1e-12, a_max=None))
        sel_vals = np.log10(np.clip(df.loc[train_idx, input_col].values, a_min=1e-12, a_max=None))

        counts, edges = np.histogram(vals, bins=n_bins)
        sel_counts, _ = np.histogram(sel_vals, bins=edges)

        frac = np.divide(sel_counts, counts,
                         out=np.zeros_like(sel_counts, dtype=float),
                         where=counts > 0)

        centers = 0.5*(edges[:-1]+edges[1:])

        fig, ax1 = plt.subplots(figsize=(7,5))
        ax1.bar(centers, counts, width=(edges[1]-edges[0]), alpha=0.3, label="Original count")
        ax2 = ax1.twinx()
        ax2.plot(centers, frac*100, "r-o", label="Selected %")
        ax1.set_xlabel(f"log10({input_col})")
        ax1.set_ylabel("Original count")
        ax2.set_ylabel("Selected (%)")
        ax1.set_title(f"Histogram fraction (trial {trial_id})")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(outdir / f"hist_fraction{suffix}.png", dpi=200)
        plt.close(fig)

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
    """
    log10 transform on series_for_binning → uniform bins(min~max in log space)
    → quota ∝ (bin count)^gamma → without-replacement sampling in-bin.

    gamma=1: follow original freq
    gamma→0: closer to equal bins (rarer bins upweighted).
    """
    rng = np.random.default_rng(seed)
    vals = series_for_binning.values.astype(float)
    print(vals)
    idx  = series_for_binning.index.values
    n    = len(idx)
    if n == 0 or n_samples <= 0:
        return np.array([], dtype=idx.dtype)
    print(n_bins)
    print("\n\n\n\n\n")
    # --- log10 변환 ---
    # 0 이하 값이 있으면 log10 불가 → 작은 epsilon 추가
    eps = 1e-12
    #이미 log로변환되어서 오기때문에..
#    log_vals = np.log10(np.clip(vals, a_min=eps, a_max=None))
    log_vals = vals
    print(log_vals)
    vmin, vmax = float(log_vals.min()), float(log_vals.max())
    print(vmin, vmax)
    if vmin == vmax:
        print(1)
        return rng.choice(idx, size=min(n_samples, n), replace=False)

    # --- log space에서 bin 분할 ---
    edges   = np.linspace(vmin, vmax, n_bins + 1)
    bin_ids = np.digitize(log_vals, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    # --- bin별 데이터 할당 ---
    bin_to_idx = {b: idx[bin_ids == b] for b in range(n_bins)}
    counts     = np.array([len(bin_to_idx[b]) for b in range(n_bins)], dtype=float)
    print(len(bin_to_idx))
    print("\n\n\n\n")
    # --- bin별 가중치 ---
    valid = np.where(counts > 0)[0]
    weights = np.zeros_like(counts)
    weights[valid] = counts[valid] ** gamma
    wsum = weights[valid].sum()
    probs = weights / (wsum if wsum > 0 else 1.0)

    # --- quota 계산 ---
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

    # --- bin별 샘플링 ---
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

    # --- 샘플 수 보정 ---
    if len(sel) > n_samples:
        sel = sel[:n_samples]
    elif len(sel) < n_samples:
        remain = np.setdiff1d(idx, sel, assume_unique=False)
        if len(remain) > 0:
            add = rng.choice(remain, size=min(n_samples - len(sel), len(remain)), replace=False)
            sel = np.concatenate([sel, add])

    return sel
# ───────────────────────── 3 Samplers (same interface) ─────────────────────────
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
    # same split; semantics: you intend to include 'Input' in features
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
        n_qt = n_train_target
        n_rd = 0

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
    return {
        "train_idx": train_idx,
        "test_idx":  te_idx,
        "train_qt_idx": qt_idx,
        "train_rd_idx": rd_idx,
        "info": {
            "strategy": "quantile_then_random_on_input",
            "train_ratio": train_ratio,
            "qt_frac_total": qt_frac_total,
            "n_bins": n_bins, "gamma": gamma,
            "seed_qt": seed_qt, "seed_rd": seed_rd,
            "shift_in": shift_in
        }
    }


# ───────────────────────── Model factory ─────────────────────────
def build_regressor(kind: str,
                    params: Optional[Dict] = None,
                    logger: Optional[logging.Logger] = None):
    params = dict(params or {})
    kind = kind.lower().strip()

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
        if not _HAS_CATBOOST:
            raise ImportError("CatBoost가 설치되어 있지 않습니다. pip install catboost")
        defaults = dict(iterations=500, depth=8, learning_rate=0.05,
                        loss_function="RMSE", random_seed=42, verbose=False)
        defaults.update(params)
        model = CatBoostRegressor(**defaults)
    else:
        raise ValueError(f"Unsupported model kind='{kind}'. Choose from rf|gbm|cat.")

    if logger:
        if hasattr(model, "get_params"):
            logger.info(f"[Model] kind={kind} | params={model.get_params()}")
        else:
            logger.info(f"[Model] kind={kind} | params={params}")
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
            imp = pd.Series(model.feature_importances_, index=X_test.columns, name="importance")
            return imp.sort_values(ascending=False)
        if _HAS_CATBOOST and isinstance(model, CatBoostRegressor):
            vals = model.get_feature_importance(type="PredictionValuesChange")
            imp = pd.Series(vals, index=X_test.columns, name="importance")
            return imp.sort_values(ascending=False)
        if use_permutation_if_missing:
            if logger:
                logger.info("[Importance] Using permutation_importance")
            r = permutation_importance(model, X_test, y_test,
                                       n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
            imp = pd.Series(r.importances_mean, index=X_test.columns, name="importance")
            return imp.sort_values(ascending=False)
    except Exception as e:
        if logger:
            logger.warning(f"[Importance] fallback failed: {e}")
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
    if logger is None:
        logger = setup_logger(outdir=outdir, name="Holdout", level=logging.INFO)

    outdir = Path(outdir) if outdir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    suffix = f"_trial{trial_id:03d}" if isinstance(trial_id, int) else ""

    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    meta_te = meta.loc[test_idx].reset_index(drop=True) if meta is not None else None

    model = build_regressor(model_kind, model_params, logger=logger)

    logger.info(f"[Holdout{suffix}] Train={len(X_tr)} | Test={len(X_te)}")
    t0 = time.time(); model.fit(X_tr, y_tr); fit_time = time.time() - t0
    logger.info(f"[Timing{suffix}] fit_time={fit_time:.3f}s")

    t1 = time.time(); y_pred = model.predict(X_te); pred_time = time.time() - t1
    logger.info(f"[Timing{suffix}] predict_time={pred_time:.3f}s")

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = mean_squared_error(y_te, y_pred)
                            #    squared=False)
    mape = (np.abs((y_te - y_pred) / np.clip(np.abs(y_te), 1e-12, None))).mean() * 100.0
    logger.info(f"[Metrics{suffix}] R2={r2:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f} | MAPE%={mape:.3f}")

    imp = get_feature_importance(model, X_te, y_te, logger=logger)

    if outdir is not None:
        if save_predictions:
            pred_df = X_te.reset_index(drop=True).copy()
            pred_df["y_true"] = y_te.values
            pred_df["y_pred"] = y_pred
            if meta_te is not None:
                pred_df = pd.concat([meta_te, pred_df], axis=1)
            pred_df.to_csv(outdir / f"predictions_holdout{suffix}.csv", index=False, encoding="utf-8-sig")
            logger.info(f"[Save] predictions_holdout{suffix}.csv")

        pd.DataFrame([{
            "Model": model_kind, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_percent": mape,
            "n_train": len(X_tr), "n_test": len(X_te),
            "fit_time_sec": fit_time, "predict_time_sec": pred_time
        }]).to_csv(outdir / f"metrics_holdout{suffix}.csv", index=False, encoding="utf-8-sig")
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
        except Exception:
            pass

    return {
        "model": model,
        "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_percent": mape,
                    "n_train": len(X_tr), "n_test": len(X_te),
                    "fit_time_sec": fit_time, "predict_time_sec": pred_time},
        "y_true": y_te.values,
        "y_pred": y_pred,
        "feature_importances_": imp
    }


# ───────────────────────── Trial seed derivation ─────────────────────────
def _derive_trial_seeds(seed_base: int, trial: int) -> Dict[str, int]:
    """trial(정수)마다 일관된 시드 세트를 만들어줌."""
    return {
        "seed":       seed_base + trial,          # random_struct / random_with_input
        "seed_qt":    seed_base + 1000*trial,     # qt sampler
        "seed_rd":    seed_base + 2000*trial,     # random fill after qt
        "model_seed": seed_base + 3000*trial,     # model random_state/random_seed
    }


# ───────────────────────── CLI / Main ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline (single trial): preprocess → sample → model → train")
    p.add_argument("--data", type=str, required=True, help="CSV path")
    p.add_argument("--outdir", type=str, default="./RUN_OUT", help="Output directory root")
    p.add_argument("--trial", type=int, required=True, help="Trial index (e.g., 1, 2, 3 ...)")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed to derive trial-specific seeds")
    p.add_argument("--target", type=str, default="Output")
#    p.add_argument("--meta", type=str, nargs="*", default=["filename","name"], help="Meta columns to keep")
    p.add_argument("--meta", type=str, nargs="*", default=["filename"], help="Meta columns to keep")

    # features
    p.add_argument("--features", type=str, choices=["struct", "struct+input", "custom"], default="struct",
                   help="Which features to train on")
    p.add_argument("--custom-features", type=str, default="",
                   help="Comma-separated list when --features custom")

    # sampler
    p.add_argument("--sampler", type=str,
                   choices=["random_struct", "random_with_input", "qt_then_rd"],
                   default="qt_then_rd")
    p.add_argument("--train-ratio", type=float, default=0.80)

    # qt sampler params
    p.add_argument("--qt-frac", type=float, default=0.40, help="fraction of TOTAL picked by quantile first")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.5)

    # model
    p.add_argument("--model", type=str, choices=["rf","gbm","cat"], default="rf")
    # a few common knobs (optional)
    p.add_argument("--rf-n-estimators", type=int, default=None)
    p.add_argument("--gbm-n-estimators", type=int, default=None)
    p.add_argument("--gbm-learning-rate", type=float, default=None)
    p.add_argument("--gbm-max-depth", type=int, default=None)
    p.add_argument("--cat-iterations", type=int, default=None)
    p.add_argument("--cat-depth", type=int, default=None)
    p.add_argument("--cat-learning-rate", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # ── trial-specific outdir ──
    OUTROOT = Path(args.outdir); OUTROOT.mkdir(parents=True, exist_ok=True)
    trial_dir = OUTROOT / f"trial_{args.trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(outdir=trial_dir, name=f"Pipeline_trial{args.trial:03d}", level=logging.INFO)

    # Load
    
    df = pd.read_csv(args.data)
    mis_lis = ["BIWSEG_ion_b","LETQAE01_ion_b","VEWLAM_clean","ja406030p_si_007_manual","HIHGEM_manual","ja406030p_si_002_manual","POZHUI_ion_b","DONNAW01_SL",]
    if "filename" in df.columns:
        before = len(df)
        df = df[~df["filename"].isin(mis_lis)].reset_index(drop=True)
        print(f"Removed {before - len(df)} rows from mis_lis")
    logger.info(f"[Data] Loaded: {args.data} | rows={len(df)} | cols={len(df.columns)}")

    # Preprocess
    df = preprocess_has_oms(df)
    
    TARGET = args.target
    META_KEEP = args.meta

    # choose features
    if args.features == "struct":
        FEATURES = select_struct_features(df)
    elif args.features == "struct+input":
        FEATURES = select_struct_features(df) + ["Input"]
    else:
        if not args.custom_features:
            raise ValueError("Provide --custom-features when --features custom")
        FEATURES = [c.strip() for c in args.custom_features.split(",") if c.strip()]
        FEATURES = select_struct_features(df, features=FEATURES)  # validates existence
    logger.info(f"[Features] Using {len(FEATURES)} features: {FEATURES}")

    df_train = dropna_for_training(df, FEATURES, TARGET, logger)
    X, y, meta = build_X_y_meta(df_train, FEATURES, TARGET, META_KEEP)

    # derive trial seeds
    seeds = _derive_trial_seeds(args.seed_base, args.trial)
    logger.info(f"[Seeds] trial={args.trial} | seeds={seeds}")

    # Sampling (single trial)
    if args.sampler == "random_struct":
        sample = sample_random_struct_only(
            df_train, train_ratio=args.train_ratio, seed=seeds["seed"], logger=logger
        )
    elif args.sampler == "random_with_input":
        if "Input" not in df_train.columns:
            raise KeyError("Input column not found for random_with_input sampler.")
        sample = sample_random_with_input(
            df_train, train_ratio=args.train_ratio, seed=seeds["seed"], logger=logger
        )
    else:  # qt_then_rd
        if "Input" not in df_train.columns:
            raise KeyError("Input column not found for qt_then_rd sampler.")
        sample = sample_quantile_then_random_on_input(
            df_train, input_col="Input",
            train_ratio=args.train_ratio, qt_frac_total=args.qt_frac,
            n_bins=args.n_bins, gamma=args.gamma,
            seed_qt=seeds["seed_qt"], seed_rd=seeds["seed_rd"], logger=logger
        )

    # Save sampling indices & info (with trial suffix)
    suffix = f"_trial{args.trial:03d}"
    pd.DataFrame({"train_idx": pd.Series(sample["train_idx"])}).to_csv(
        trial_dir / f"train_indices{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"test_idx": pd.Series(sample["test_idx"])}).to_csv(
        trial_dir / f"test_indices{suffix}.csv", index=False, encoding="utf-8-sig")
    if sample["train_qt_idx"] is not None:
        pd.DataFrame({"train_qt_idx": pd.Series(sample["train_qt_idx"])}).to_csv(
            trial_dir / f"train_qt_indices{suffix}.csv", index=False, encoding="utf-8-sig")
    if isinstance(sample["train_rd_idx"], np.ndarray):
        pd.DataFrame({"train_rd_idx": pd.Series(sample["train_rd_idx"])}).to_csv(
            trial_dir / f"train_rd_indices{suffix}.csv", index=False, encoding="utf-8-sig")
    with open(trial_dir / f"sampling_info{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(sample["info"], f, ensure_ascii=False, indent=2)
    if "Input" in df_train.columns and args.target in df_train.columns:
        save_sampling_plots(df_train,
                            input_col="Input",
                            output_col=args.target,
                            train_idx=sample["train_idx"],
                            qt_idx=sample.get("train_qt_idx"),
                            outdir=trial_dir,
                            trial_id=args.trial,
                            n_bins = args.n_bins)
        logger.info(f"[Save] sampling scatter & histogram plots")
    # Build model params (inject trial-specific seed)
    model_params: Dict[str, Union[int, float]] = {}
    if args.model == "rf":
        if args.rf_n_estimators is not None:
            model_params["n_estimators"] = args.rf_n_estimators
        model_params.setdefault("n_jobs", -1)
        model_params["random_state"] = seeds["model_seed"]
    elif args.model == "gbm":
        if args.gbm_n_estimators is not None:
            model_params["n_estimators"] = args.gbm_n_estimators
        if args.gbm_learning_rate is not None:
            model_params["learning_rate"] = args.gbm_learning_rate
        if args.gbm_max_depth is not None:
            model_params["max_depth"] = args.gbm_max_depth
        model_params["random_state"] = seeds["model_seed"]
    else:  # cat
        if args.cat_iterations is not None:
            model_params["iterations"] = args.cat_iterations
        if args.cat_depth is not None:
            model_params["depth"] = args.cat_depth
        if args.cat_learning_rate is not None:
            model_params["learning_rate"] = args.cat_learning_rate
        model_params["random_seed"] = seeds["model_seed"]
        model_params.setdefault("verbose", False)

    # Train (holdout, single trial)
    _ = train_holdout_general(
        X=X, y=y,
        train_idx=sample["train_idx"], test_idx=sample["test_idx"],
        model_kind=args.model, model_params=model_params,
        meta=meta, outdir=trial_dir, logger=logger,
        save_predictions=True, trial_id=args.trial
    )

    logger.info(f"[Done] trial={args.trial} | outputs at: {trial_dir}")


if __name__ == "__main__":
    main()
