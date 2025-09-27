import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Dataset
# CSV layout per sample: [Pgen(8), Pload(8), theta(8)] -> total 24 columns
# ---------------------------
class DCPFDataset(Dataset):
    def __init__(self, csv_path, input_mean=None, input_std=None, target_mean=None, target_std=None, fit_stats=False):
        arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        assert arr.shape[1] == 24, f"Expected 24 columns, got {arr.shape[1]}"
        self.x = arr[:, :16]   # 8 Pgen + 8 Pload
        self.y = arr[:, 16:]   # 8 theta

        # compute or use provided normalization stats
        if fit_stats:
            self.input_mean = self.x.mean(axis=0, keepdims=True)
            self.input_std  = self.x.std(axis=0, keepdims=True) + 1e-8
            self.target_mean = self.y.mean(axis=0, keepdims=True)
            self.target_std  = self.y.std(axis=0, keepdims=True) + 1e-8
        else:
            assert input_mean is not None and input_std is not None
            assert target_mean is not None and target_std is not None
            self.input_mean = input_mean
            self.input_std  = input_std
            self.target_mean = target_mean
            self.target_std  = target_std

        # standardize
        self.xn = (self.x - self.input_mean) / self.input_std
        self.yn = (self.y - self.target_mean) / self.target_std

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.xn[idx]),
            torch.from_numpy(self.yn[idx]),
        )

# ---------------------------
# Model: linear mapping 16 -> 8
# Rationale: DCPF mapping is linear, so a single Linear layer suffices and generalizes better, especially with small data.
# ---------------------------
class LinearDCPF(nn.Module):
    def __init__(self, in_dim=16, out_dim=8):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)  # bias allowed; optimizer can learn near-zero bias

    def forward(self, x):
        return self.lin(x)

# ---------------------------
# Metrics
# Error (%) ~ mean over all elements: |y_pred - y_true| / (|y_true| + eps) * 100
# ---------------------------
def mean_relative_error_percent(y_true, y_pred, eps=1e-6):
    with torch.no_grad():
        num = (y_pred - y_true).abs()
        den = y_true.abs() + eps
        return (num / den).mean().item() * 100.0

# ---------------------------
# Train & Eval
# ---------------------------
def train_one_model(train_csv, test_csv_list, device="cuda" if torch.cuda.is_available() else "cpu",
                    epochs=400, batch_size=64, lr=5e-3, weight_decay=1e-4, seed=42):
    set_seed(seed)

    # Fit normalization on training set
    train_ds_fit = DCPFDataset(train_csv, fit_stats=True)
    in_mean, in_std = train_ds_fit.input_mean, train_ds_fit.input_std
    tgt_mean, tgt_std = train_ds_fit.target_mean, train_ds_fit.target_std

    # Re-instantiate datasets using fixed stats
    train_ds = DCPFDataset(train_csv, input_mean=in_mean, input_std=in_std,
                           target_mean=tgt_mean, target_std=tgt_std, fit_stats=False)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    }

    # Prepare tests with same normalization as training
    test_sets = {}
    for name, path in test_csv_list.items():
        test_sets[name] = DCPFDataset(path, input_mean=in_mean, input_std=in_std,
                                      target_mean=tgt_mean, target_std=tgt_std, fit_stats=False)
    model = LinearDCPF().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Training loop
    model.train()
    for ep in range(epochs):
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate (de-standardize to compute % error on original scale)
    model.eval()
    results = {}
    with torch.no_grad():
        # Train set performance (optional to report)
        xb = torch.from_numpy(train_ds.xn).to(device)
        y_true_n = torch.from_numpy(train_ds.yn).to(device)
        y_pred_n = model(xb)

        # invert standardization
        y_true = y_true_n * torch.from_numpy(tgt_std).to(device) + torch.from_numpy(tgt_mean).to(device)
        y_pred = y_pred_n * torch.from_numpy(tgt_std).to(device) + torch.from_numpy(tgt_mean).to(device)
        results["train"] = mean_relative_error_percent(y_true, y_pred)

        # Test sets
        for name, ds in test_sets.items():
            xb = torch.from_numpy(ds.xn).to(device)
            y_true_n = torch.from_numpy(ds.yn).to(device)
            y_pred_n = model(xb)
            y_true = y_true_n * torch.from_numpy(tgt_std).to(device) + torch.from_numpy(tgt_mean).to(device)
            y_pred = y_pred_n * torch.from_numpy(tgt_std).to(device) + torch.from_numpy(tgt_mean).to(device)
            results[name] = mean_relative_error_percent(y_true, y_pred)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train1", default="train_set1.csv")  # 1000 samples, ±20%
    parser.add_argument("--train2", default="train_set2.csv")  # 200 samples, ±20%
    parser.add_argument("--test1",  default="test_set1.csv")   # 1000 samples, ±20%
    parser.add_argument("--test2",  default="test_set2.csv")   # 1000 samples, ±40%
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Four combinations required by HW:
    # (Train1 -> Test1), (Train1 -> Test2), (Train2 -> Test1), (Train2 -> Test2)
    combos = [
        ("Train1(±20%, N=1000) → Test1(±20%, N=1000)", args.train1, {"Test1": args.test1}),
        ("Train1(±20%, N=1000) → Test2(±40%, N=1000)", args.train1, {"Test2": args.test2}),
        ("Train2(±20%, N=200)  → Test1(±20%, N=1000)", args.train2, {"Test1": args.test1}),
        ("Train2(±20%, N=200)  → Test2(±40%, N=1000)", args.train2, {"Test2": args.test2}),
    ]

    rows = []
    for title, train_csv, test_dict in combos:
        res = train_one_model(
            train_csv=train_csv,
            test_csv_list=test_dict,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.wd,
            seed=args.seed
        )
        row = {"Combo": title, "TrainError(%)": round(res["train"], 6)}
        for k, v in res.items():
            if k == "train":
                continue
            row[f"{k}Error(%)"] = round(v, 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Order columns nicely
    all_cols = ["Combo", "TrainError(%)", "Test1Error(%)", "Test2Error(%)"]
    df = df.reindex(columns=[c for c in all_cols if c in df.columns])
    print("\n=== Data-driven DCPF Results (Linear Model) ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()