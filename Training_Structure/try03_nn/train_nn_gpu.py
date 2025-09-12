import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import logging


# ── Logger 설정 ──
def setup_logger(outdir, name="NN_pipeline", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(Path(outdir) / f"{name}.log", encoding="utf-8")
    fh.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter); ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ── Quantile Weighted Sampler ──
def quantile_weighted_sample(series, n_samples, n_bins=10, gamma=0.5, seed=123):
    rng = np.random.default_rng(seed)
    vals = series.values.astype(float)
    idx = series.index.values
    vmin, vmax = float(vals.min()), float(vals.max())
    edges = np.linspace(vmin, vmax, n_bins + 1)
    bin_ids = np.digitize(vals, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_to_idx = {b: idx[bin_ids == b] for b in range(n_bins)}
    counts = np.array([len(bin_to_idx[b]) for b in range(n_bins)], dtype=float)

    valid = np.where(counts > 0)[0]
    weights = np.zeros_like(counts)
    weights[valid] = counts[valid] ** gamma
    probs = weights / weights.sum()

    quota = (probs * n_samples).astype(int)

    selected = []
    for b in valid:
        pool = bin_to_idx[b]
        k = min(len(pool), quota[b])
        if k > 0:
            chosen = rng.choice(pool, size=k, replace=False)
            selected.append(chosen)

    sel = np.concatenate(selected) if selected else np.array([], dtype=int)
    return sel


# ── NN 모델 ──
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)


# ── Training + Evaluation ──
def train_and_evaluate(X, y, sampler, train_ratio=0.8, qt_frac=0.4,
                       n_bins=10, gamma=0.5, seed=123,
                       hidden1=64, hidden2=32, lr=1e-3,
                       batch_size=64, epochs=100, outdir="./RUN_OUT", trial=1,
                       patience=20, delta=1e-4):

    OUTDIR = Path(outdir) / f"trial_{trial:03d}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUTDIR, name=f"NN_trial{trial:03d}")

    # ✅ Device 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Device] Using {device}")

    # ── log10 변환 ──
    if "Input" in X.columns:
        X["Input"] = np.log10(X["Input"])
    y = np.log10(y)

    rng = np.random.default_rng(seed)
    n_total = len(X)

    if sampler == "random_struct":
        X = X.drop(columns=["Input"], errors="ignore")
        all_idx = X.index.values
        train_idx = rng.choice(all_idx, size=int(n_total * train_ratio), replace=False)
        test_idx = np.setdiff1d(all_idx, train_idx)

        logger.info(f"[Sampler] {sampler} | Train={len(train_idx)} | Test={len(test_idx)}")
    elif sampler == "random_with_input":
        all_idx = X.index.values
        train_idx = rng.choice(all_idx, size=int(n_total * train_ratio), replace=False)
        test_idx = np.setdiff1d(all_idx, train_idx)

        logger.info(f"[Sampler] {sampler} | Train={len(train_idx)} | Test={len(test_idx)}")
    elif sampler == "qt_then_rd":
        n_qt = int(n_total * qt_frac)
        n_rd = int(n_total * (train_ratio - qt_frac))
        qt_idx = quantile_weighted_sample(X["Input"], n_samples=n_qt,
                                          n_bins=n_bins, gamma=gamma, seed=seed)
        remain = np.setdiff1d(X.index.values, qt_idx)
        rd_idx = rng.choice(remain, size=n_rd, replace=False)
        te_idx = np.setdiff1d(remain, rd_idx)
        train_idx = np.concatenate([qt_idx, rd_idx])
        test_idx = te_idx

        logger.info(f"[Sampler] {sampler} | Train={len(train_idx)} | qt={len(qt_idx)} | rd={len(rd_idx)} | Test={len(test_idx)}")
    else:
        raise ValueError("Unsupported sampler.")


    # ── Split ──
    X_train, y_train = X.loc[train_idx].values, y[train_idx]
    X_test, y_test = X.loc[test_idx].values, y[test_idx]

    # ── Scaling ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    # ── Torch Dataset (train/val split) ──
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train_t, y_train_t)
    n_val = max(1, int(0.1 * len(dataset)))  # 10% validation
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ✅ GPU로 올리기
    X_test_t, y_test_t = X_test_t.to(device), y_test_t.to(device)

    # ── 모델 ──
    model = SimpleMLP(input_dim=X_train.shape[1], hidden1=hidden1, hidden2=hidden2).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── Early Stopping 설정 ──
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    # ── 학습 ──
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            logger.info(f"[Epoch {epoch+1}] Train Loss={running_loss/len(train_loader):.6f} | Val Loss={val_loss:.6f}")

        # early stopping check
        if val_loss < best_loss - delta:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[EarlyStopping] Epoch {epoch+1}: no improvement for {patience} epochs.")
                break

    fit_time = time.time() - t0
    logger.info(f"[Timing] fit_time={fit_time:.3f}s")

    # ── Best 모델 복원 ──
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── 평가 ──
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t).squeeze().cpu().numpy()
    pred_time = time.time() - t1
    logger.info(f"[Timing] predict_time={pred_time:.3f}s")

    # inverse transform
    y_pred_inv = y_scaler.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()
    y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()  # ✅ FIX
    #y_pred_inv = y_scaler.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()
#    y_test_inv = y_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).ravel()
    # y_test_inv = y_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).ravel()
    #y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_final = np.power(10, y_pred_inv)
    y_test_final = np.power(10, y_test_inv)

    r2 = r2_score(y_test_final, y_pred_final)
    mae = mean_absolute_error(y_test_final, y_pred_final)
    rmse = mean_squared_error(y_test_final, y_pred_final, squared=False)

    logger.info(f"[Metrics] R2={r2:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f}")

    # ── Save Results ──
    pred_df = pd.DataFrame({"y_true": y_test_final, "y_pred": y_pred_final})
    pred_df.to_csv(OUTDIR / f"predictions_trial{trial:03d}.csv", index=False, encoding="utf-8-sig")
    logger.info(f"[Save] predictions_trial{trial:03d}.csv")

    pd.DataFrame([{
        "R2": r2, "MAE": mae, "RMSE": rmse,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "fit_time_sec": fit_time, "predict_time_sec": pred_time
    }]).to_csv(OUTDIR / f"metrics_trial{trial:03d}.csv", index=False, encoding="utf-8-sig")
    logger.info(f"[Save] metrics_trial{trial:03d}.csv")

    params = {"sampler": sampler, "train_ratio": train_ratio,
              "qt_frac": qt_frac, "n_bins": n_bins, "gamma": gamma,
              "hidden1": hidden1, "hidden2": hidden2,
              "lr": lr, "batch_size": batch_size, "epochs": epochs,
              "seed": seed, "device": str(device),
              "patience": patience, "delta": delta}
    with open(OUTDIR / f"params_trial{trial:03d}.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    logger.info(f"[Save] params_trial{trial:03d}.json")


def main():
    parser = argparse.ArgumentParser(description="NN training pipeline with GPU + EarlyStopping")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./RUN_OUT")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--sampler", type=str,
                        choices=["random_struct", "random_with_input", "qt_then_rd"],
                        default="qt_then_rd")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--qt-frac", type=float, default=0.4)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--delta", type=float, default=1e-4)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    mis_lis = ["BIWSEG_ion_b","LETQAE01_ion_b","VEWLAM_clean",
               "ja406030p_si_007_manual","HIHGEM_manual",
               "ja406030p_si_002_manual","POZHUI_ion_b","DONNAW01_SL"]
    if "filename" in df.columns:
        before = len(df)
        df = df[~df["filename"].isin(mis_lis)].reset_index(drop=True)
        print(f"Removed {before - len(df)} rows from mis_lis")

    X = df[['LCD','PLD','LFPD','cm3_g','ASA_m2_cm3','ASA_m2_g',
            'NASA_m2_cm3','NASA_m2_g','AV_VF','AV_cm3_g',
            'NAV_cm3_g','Has_OMS','Input']].copy()
    y = df["Output"].values

    train_and_evaluate(X, y,
                       sampler=args.sampler, train_ratio=args.train_ratio,
                       qt_frac=args.qt_frac, n_bins=args.n_bins, gamma=args.gamma,
                       hidden1=args.hidden1, hidden2=args.hidden2,
                       lr=args.lr, batch_size=args.batch_size,
                       epochs=args.epochs, outdir=args.outdir, trial=args.trial,
                       patience=args.patience, delta=args.delta)


if __name__ == "__main__":
    main()
