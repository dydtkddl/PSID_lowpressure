import os
import datetime
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Settings ───────────────────────────────────────────
SEED = 52
BATCH_SIZE = 64
HIDDEN_DIM = 64
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-3
EPOCHS = 500
PATIENCE = 30
INITIAL_RATIO = 0.01
SAMPLES_PER_ITER = 10
TARGET_RATIO = 0.30
MCD_N_SIMULATIONS = 20
BIN_WISE_COVERAGE = True
NUM_BINS = 10
RELATIVE = False
X_SCALE = True
QUANTILE_LOW_ADS = True
prefix = "[Ar_298_abs_0.5_1]"

# ── Reproducibility ────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

####### GPU / Backend ##########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# ── Output directory & Logging setup ──────────────────
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir = f"{prefix}AL_{ts}"
os.makedirs(f"{out_dir}/predictions", exist_ok=True)

log_path = os.path.join(out_dir, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting Active Learning run")

# ── Data Load ──────────────────────────────────────────
df = pd.read_csv("./Ar_298K_abs_0.5 bar_1.0 bar.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

X_all = df.iloc[:, 1:-1].values.astype(np.float32)
Y_all = df.iloc[:, -1].values.astype(np.float32)
LOW_all = X_all[:, -1]

idx_all = np.arange(len(X_all))

if X_SCALE:
    scaler_X = StandardScaler().fit(X_all)
    X_scaled = scaler_X.transform(X_all).astype(np.float32)
else:
    X_scaled = X_all

# ── Dataset & Model definitions ────────────────────────
class GCMCDataset(Dataset):
    def __init__(self, X, Y, LOW):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
        self.LOW = torch.tensor(LOW, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.LOW[i]

class GCMCModel(nn.Module):
    def __init__(self, dim, hidden, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# ── Functions ─────────────────────────────────────────
def mc_dropout_predict(model, X, n_simulations):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_simulations):
            preds.append(model(X).cpu().numpy())
    arr = np.array(preds)
    return arr.mean(axis=0), arr.std(axis=0)

def stratified_quantile_sampling(low_values, idx_pool, n_samples):
    quantiles = pd.qcut(low_values[idx_pool], q=NUM_BINS, labels=False, duplicates='drop')
    idx_sampled = []
    per_bin = max(1, n_samples // NUM_BINS)
    for bin_id in np.unique(quantiles):
        bin_idxs = idx_pool[quantiles == bin_id]
        sampled = np.random.choice(bin_idxs, size=min(per_bin, len(bin_idxs)), replace=False)
        idx_sampled.extend(sampled)
    return np.array(idx_sampled)

def train_with_early_stopping(model, optimizer, loss_fn, train_dl, val_dl):
    best_loss = float('inf')
    patience_ctr = 0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb, lb in train_dl:
            feats = scaler_X.transform(xb)
            xb = torch.tensor(feats).float().to(device)
            yb = yb.to(device)
            lb = lb.to(device)
            y_rel = yb / lb if RELATIVE else yb
            preds = model(xb)
            loss = loss_fn(preds, y_rel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb, lb in val_dl:
                feats = scaler_X.transform(xb)
                xb = torch.tensor(feats).float().to(device)
                yb = yb.to(device)
                lb = lb.to(device)
                y_rel = yb / lb if RELATIVE else yb
                total_val += loss_fn(model(xb), y_rel).item()
        avg_val = total_val / len(val_dl.dataset)
        if avg_val < best_loss:
            best_loss = avg_val
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

# ── Initialize pools ──────────────────────────────────
n_samples_init = int(INITIAL_RATIO * len(X_all))
idx_labeled = stratified_quantile_sampling(LOW_all, idx_all, n_samples_init)
idx_unlabeled = np.setdiff1d(idx_all, idx_labeled)

# ── Model, optimizer, loss ────────────────────────────
model = GCMCModel(X_all.shape[1], HIDDEN_DIM, DROPOUT_RATE).to(device)
if device.type == "cuda":
    try:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=True)
    except Exception:
        # PyTorch/드라이버/아키텍처 조합에 따라 fused 미지원일 수 있음
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    # CPU에서는 fused 미지원
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# ── Active Learning loop ───────────────────────────────
metrics = []
n_target = int(TARGET_RATIO * len(X_all))
max_iters = (n_target - len(idx_labeled)) // SAMPLES_PER_ITER + 1

for it in range(max_iters):
    logger.info(f"Iteration {it} - labeled {len(idx_labeled)} / target {n_target}")

    # 1) Train
    d_train = GCMCDataset(X_all[idx_labeled], Y_all[idx_labeled], LOW_all[idx_labeled])
    dl_train = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=False)
    train_with_early_stopping(model, optimizer, loss_fn, dl_train, dl_val)

    # 2) Evaluate
    model.eval()
    X_rem = torch.tensor(scaler_X.transform(X_all[idx_unlabeled])).float().to(device)
    y_true_rem = Y_all[idx_unlabeled]
    preds_mean, _ = mc_dropout_predict(model, X_rem, MCD_N_SIMULATIONS)
    y_pred_rem = preds_mean.flatten() * LOW_all[idx_unlabeled] if RELATIVE else preds_mean.flatten()
    r2 = r2_score(y_true_rem, y_pred_rem)
    mae = mean_absolute_error(y_true_rem, y_pred_rem)
    mse = mean_squared_error(y_true_rem, y_pred_rem)
    logger.info(f"Eval Remaining - R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")

    metrics.append({"iter": it, "n_labeled": len(idx_labeled), "r2": r2, "mae": mae, "mse": mse})
    pd.DataFrame({"y_true": y_true_rem, "y_pred": y_pred_rem})\
        .to_csv(f"{out_dir}/predictions/rem_preds_iter_{it}.csv", index=False)

    if len(idx_labeled) >= n_target:
        break

    # 4) Query selection
    logger.info("[Query] MC Dropout Uncertainty...")
    unl_ds = GCMCDataset(X_all[idx_unlabeled], Y_all[idx_unlabeled], LOW_all[idx_unlabeled])
    unl_dl = DataLoader(unl_ds, batch_size=BATCH_SIZE, shuffle=False)
    uncertainties = []
    for xb, _, _ in unl_dl:
        xb_scaled = torch.tensor(scaler_X.transform(xb)).float().to(device)
        _, stds = mc_dropout_predict(model, xb_scaled, MCD_N_SIMULATIONS)
        uncertainties.extend(stds.flatten())
    uncertainties = np.array(uncertainties)
    half = SAMPLES_PER_ITER // 2
    strat_idx = stratified_quantile_sampling(LOW_all, idx_unlabeled, half)
    uncert_order = np.argsort(uncertainties)[-half:]
    uncert_idx = idx_unlabeled[uncert_order]
    new_idx = np.concatenate([strat_idx, uncert_idx])

    idx_labeled = np.concatenate([idx_labeled, new_idx])
    idx_unlabeled = np.setdiff1d(idx_unlabeled, new_idx)

# Save outputs
pd.DataFrame(metrics).to_csv(f"{out_dir}/active_learning_metrics.csv", index=False)
torch.save(model.state_dict(), f"{out_dir}/final_model.pth")
logger.info("Done. All results saved.")
