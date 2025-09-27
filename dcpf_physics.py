import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.io import loadmat
except Exception as e:
    loadmat = None

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Dataset
# CSV per sample: [Pgen(8), Pload(8), theta(8)] -> 24 columns
# Return both normalized tensors (for model) and raw (for physics loss & metrics)
# ---------------------------
class DCPFDataset(Dataset):
    def __init__(self, csv_path, input_mean=None, input_std=None, target_mean=None, target_std=None, fit_stats=False):
        arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        assert arr.shape[1] == 24, f"Expected 24 columns, got {arr.shape[1]}"
        self.x_raw = arr[:, :16]     # raw [Pgen(8), Pload(8)]
        self.y_raw = arr[:, 16:]     # raw theta(8)

        # compute or use provided normalization stats
        if fit_stats:
            self.input_mean = self.x_raw.mean(axis=0, keepdims=True)
            self.input_std  = self.x_raw.std(axis=0, keepdims=True) + 1e-8
            self.target_mean = self.y_raw.mean(axis=0, keepdims=True)
            self.target_std  = self.y_raw.std(axis=0, keepdims=True) + 1e-8
        else:
            assert input_mean is not None and input_std is not None
            assert target_mean is not None and target_std is not None
            self.input_mean = input_mean; self.input_std = input_std
            self.target_mean = target_mean; self.target_std = target_std

        # standardize
        self.xn = (self.x_raw - self.input_mean) / self.input_std
        self.yn = (self.y_raw - self.target_mean) / self.target_std

    def __len__(self):
        return self.x_raw.shape[0]

    def __getitem__(self, idx):
        return {
            "xn": torch.from_numpy(self.xn[idx]),
            "yn": torch.from_numpy(self.yn[idx]),
            "x_raw": torch.from_numpy(self.x_raw[idx]),
            "y_raw": torch.from_numpy(self.y_raw[idx]),
        }

# ---------------------------
# Model: linear mapping 16 -> 8
# ---------------------------
class LinearDCPF(nn.Module):
    def __init__(self, in_dim=16, out_dim=8):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, x):
        return self.lin(x)

# ---------------------------
# Metrics
# Error (%) = mean over all elements: |y_pred - y_true| / (|y_true| + eps) * 100
# ---------------------------
def mean_relative_error_percent(y_true, y_pred, eps=1e-6):
    with torch.no_grad():
        num = (y_pred - y_true).abs()
        den = y_true.abs() + eps
        return (num / den).mean().item() * 100.0

# ---------------------------
# Physics-informed loss (normalized)
# Loss = alpha * Loss_data + Loss_phys
# where:
#   Loss_data = MSE(θ̂, θ) / (MSE(0, θ) + eps)
#   Loss_phys = MSE(B_pf θ̂, Pgen - Pload) / (MSE(0, Pgen-Pload) + eps)
# Pred θ̂ is de-standardized back to raw scale before computing both terms.
# ---------------------------
def compute_losses(batch, y_pred_n, tgt_mean, tgt_std, B_pf, device, alpha=1.0, eps=1e-12):
    # de-standardize prediction and target to raw scale
    tgt_mean_t = torch.from_numpy(tgt_mean).to(device)
    tgt_std_t  = torch.from_numpy(tgt_std).to(device)
    y_pred = y_pred_n * tgt_std_t + tgt_mean_t
    y_true = batch["y_raw"].to(device)

    # data term (normalized)
    mse_data = ((y_pred - y_true) ** 2).mean()
    denom_data = (y_true ** 2).mean() + eps
    loss_data = mse_data / denom_data

    # physics term (normalized)
    x_raw = batch["x_raw"].to(device)
    pgen = x_raw[:, :8]
    pload = x_raw[:, 8:]
    rhs = pgen - pload
    lhs = torch.matmul(y_pred, torch.from_numpy(B_pf.T).to(device))  # (B_pf @ theta) -> careful with row vs col, use θ row * B^T
    mse_phys = ((lhs - rhs) ** 2).mean()
    denom_phys = (rhs ** 2).mean() + eps
    loss_phys = mse_phys / denom_phys

    total = alpha * loss_data + loss_phys
    return total, loss_data.detach(), loss_phys.detach()

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_one_model(
    train_csv, test_csv_list, B_pf, device="cuda" if torch.cuda.is_available() else "cpu",
    alpha=1.0, epochs=400, batch_size=64, lr=5e-3, weight_decay=1e-4, seed=42, log_prefix=None
):
    set_seed(seed)

    # Fit normalization on training set
    ds_fit = DCPFDataset(train_csv, fit_stats=True)
    in_mean, in_std = ds_fit.input_mean, ds_fit.input_std
    tgt_mean, tgt_std = ds_fit.target_mean, ds_fit.target_std

    train_ds = DCPFDataset(train_csv, input_mean=in_mean, input_std=in_std,
                           target_mean=tgt_mean, target_std=tgt_std, fit_stats=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # Prepare tests with same normalization as training
    test_sets = {}
    for name, path in test_csv_list.items():
        test_sets[name] = DCPFDataset(path, input_mean=in_mean, input_std=in_std,
                                      target_mean=tgt_mean, target_std=tgt_std, fit_stats=False)

    model = LinearDCPF().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "total": [], "data": [], "phys": []}

    # Training loop
    model.train()
    for ep in range(epochs):
        total_epoch = data_epoch = phys_epoch = 0.0
        n_batches = 0
        for batch in train_loader:
            xb = batch["xn"].to(device)
            yb = batch["yn"].to(device)
            y_pred_n = model(xb)

            total_loss, data_loss, phys_loss = compute_losses(
                batch, y_pred_n, tgt_mean, tgt_std, B_pf, device, alpha=alpha
            )

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            total_epoch += total_loss.item()
            data_epoch  += data_loss.item()
            phys_epoch  += phys_loss.item()
            n_batches   += 1

        history["epoch"].append(ep+1)
        history["total"].append(total_epoch / max(1, n_batches))
        history["data"].append(data_epoch / max(1, n_batches))
        history["phys"].append(phys_epoch / max(1, n_batches))

    # Evaluate on train and tests (in raw scale for % error)
    model.eval()
    results = {}
    with torch.no_grad():
        # Train set
        xb = torch.from_numpy(train_ds.xn).to(device)
        y_true_n = torch.from_numpy(train_ds.yn).to(device)
        y_pred_n = model(xb)

        tgt_mean_t = torch.from_numpy(tgt_mean).to(device)
        tgt_std_t  = torch.from_numpy(tgt_std).to(device)
        y_true = y_true_n * tgt_std_t + tgt_mean_t
        y_pred = y_pred_n * tgt_std_t + tgt_mean_t
        results["train"] = mean_relative_error_percent(y_true, y_pred)

        for name, ds in test_sets.items():
            xb = torch.from_numpy(ds.xn).to(device)
            y_true_n = torch.from_numpy(ds.yn).to(device)
            y_pred_n = model(xb)

            y_true = y_true_n * tgt_std_t + tgt_mean_t
            y_pred = y_pred_n * tgt_std_t + tgt_mean_t
            results[name] = mean_relative_error_percent(y_true, y_pred)

    # Optional: save logs for convergence plots
    if log_prefix is not None:
        pd.DataFrame(history).to_csv(f"{log_prefix}_loss_curve.csv", index=False)

    return results, history

def load_Bpf(mat_path="B_pf.mat"):
    if loadmat is None:
        raise RuntimeError("scipy is not available to load .mat; install scipy or provide B as .npy")
    mat = loadmat(mat_path)
    if "B_pf" not in mat:
        raise KeyError("B_pf variable not found in the .mat file")
    B = mat["B_pf"].astype(np.float32)
    # Ensure shape (n_bus, n_bus)
    assert B.ndim == 2 and B.shape[0] == B.shape[1], f"Unexpected B_pf shape: {B.shape}"
    return B

def main():
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--train1", default="train_set1.csv")
    parser.add_argument("--train2", default="train_set2.csv")
    parser.add_argument("--test1",  default="test_set1.csv")
    parser.add_argument("--test2",  default="test_set2.csv")
    parser.add_argument("--Bmat",   default="B_pf.mat")

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # which parts to run
    parser.add_argument("--run_a", action="store_true", help="Run part (a): alpha=0, four combos")
    parser.add_argument("--run_b", action="store_true", help="Run part (b): Train2->Test2, alpha in {0,1,10}")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load B matrix
    B_pf = load_Bpf(args.Bmat)

    # ----------------------
    # (a) alpha=0, four combos
    # ----------------------
    if args.run_a:
        alpha = 0.0
        combos = [
            ("Train1(±20%, N=1000) → Test1(±20%, N=1000)", args.train1, {"Test1": args.test1}),
            ("Train1(±20%, N=1000) → Test2(±40%, N=1000)", args.train1, {"Test2": args.test2}),
            ("Train2(±20%, N=200)  → Test1(±20%, N=1000)", args.train2, {"Test1": args.test1}),
            ("Train2(±20%, N=200)  → Test2(±40%, N=1000)", args.train2, {"Test2": args.test2}),
        ]
        rows = []
        for title, train_csv, test_dict in combos:
            res, _ = train_one_model(
                train_csv=train_csv,
                test_csv_list=test_dict,
                B_pf=B_pf,
                device=device,
                alpha=alpha,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.wd,
                seed=args.seed,
                log_prefix=None
            )
            row = {"Combo": title, "Alpha": alpha, "TrainError(%)": round(res["train"], 6)}
            for k, v in res.items():
                if k == "train": continue
                row[f"{k}Error(%)"] = round(v, 6)
            rows.append(row)
        print("\n=== (a) Physics-only (alpha=0) Results ===")
        df_a = pd.DataFrame(rows)
        ordered = ["Combo", "Alpha", "TrainError(%)", "Test1Error(%)", "Test2Error(%)"]
        print(df_a.reindex(columns=[c for c in ordered if c in df_a.columns]).to_string(index=False))

    # ----------------------
    # (b) Train2->Test2, alpha in {0,1,10}; log curves for convergence speed
    # ----------------------
    if args.run_b:
        alphas = [0.0, 1.0, 10.0]
        rows = []
        for alpha in alphas:
            res, hist = train_one_model(
                train_csv=args.train2,
                test_csv_list={"Test2": args.test2},
                B_pf=B_pf,
                device=device,
                alpha=alpha,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.wd,
                seed=args.seed,
                log_prefix=f"train2_test2_alpha{alpha:g}"
            )
            rows.append({
                "Combo": "Train2(±20%, N=200) → Test2(±40%, N=1000)",
                "Alpha": alpha,
                "TrainError(%)": round(res["train"], 6),
                "Test2Error(%)": round(res["Test2"], 6)
            })
        print("\n=== (b) Convergence & Final Accuracy (Train2→Test2) ===")
        df_b = pd.DataFrame(rows)
        print(df_b.to_string(index=False))
        print("\nSaved loss curves: train2_test2_alpha{0,1,10}_loss_curve.csv (columns: epoch,total,data,phys)")

if __name__ == "__main__":
    main()