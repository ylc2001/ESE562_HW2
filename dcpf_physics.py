import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DCPFDataset(Dataset):
    def __init__(self, csv_path, input_mean=None, input_std=None, target_mean=None, target_std=None, fit_stats=False):
        arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        self.x_raw = arr[:, :16]
        self.y_raw = arr[:, 16:]
        if fit_stats:
            self.input_mean = self.x_raw.mean(axis=0, keepdims=True)
            self.input_std  = self.x_raw.std(axis=0, keepdims=True) + 1e-8
            self.target_mean = self.y_raw.mean(axis=0, keepdims=True)
            self.target_std  = self.y_raw.std(axis=0, keepdims=True) + 1e-8
        else:
            self.input_mean = input_mean; self.input_std = input_std
            self.target_mean = target_mean; self.target_std = target_std
        self.xn = (self.x_raw - self.input_mean) / self.input_std
        self.yn = (self.y_raw - self.target_mean) / self.target_std
    def __len__(self): return self.x_raw.shape[0]
    def __getitem__(self, idx):
        return {
            "xn": torch.from_numpy(self.xn[idx]),
            "yn": torch.from_numpy(self.yn[idx]),
            "x_raw": torch.from_numpy(self.x_raw[idx]),
            "y_raw": torch.from_numpy(self.y_raw[idx]),
        }

class LinearDCPF(nn.Module):
    def __init__(self, in_dim=16, out_dim=8):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, x): return self.lin(x)

def mean_relative_error_percent(y_true, y_pred, eps=1e-6):
    # RMSPE: Root Mean Squared Percentage Error
    with torch.no_grad():
        rel_err_sq = ((y_pred - y_true) / (y_true + eps)) ** 2
        rmspe = torch.sqrt(rel_err_sq.mean()).item() * 100.0
        return rmspe

# ---------------------------
# Physics-informed loss (normalized)
# Loss = alpha * Loss_data + Loss_phys
# where:
#   Loss_data = MSE(θ̂, θ)
#   Loss_phys = MSE(B_pf θ̂ - (Pgen - Pload))
# Pred θ̂ is de-standardized back to raw scale before computing both terms.
# ---------------------------
def compute_losses(batch, y_pred_n, tgt_mean, tgt_std, B_pf, device, alpha=1.0, eps=1e-12):
    tgt_mean_t = torch.from_numpy(tgt_mean).to(device)
    tgt_std_t  = torch.from_numpy(tgt_std).to(device)
    y_pred = y_pred_n * tgt_std_t + tgt_mean_t
    y_true_n = batch["yn"].to(device)

    # Loss_data: MSE(θ̂, θ)
    loss_data = torch.nn.functional.mse_loss(y_pred_n, y_true_n)

    # Loss_phys: MSE(B_pf θ̂ - (Pgen - Pload))
    x_raw = batch["x_raw"].to(device)           # shape: [batch, 16]
    pgen = x_raw[:, :8]
    pload = x_raw[:, 8:]
    rhs = pgen - pload                         # shape: [batch, 8]
    B_pf_t = torch.from_numpy(B_pf.T).to(device)  # shape: [8, 8]
    lhs = torch.matmul(y_pred, B_pf_t)         # shape: [batch, 8]
    loss_phys = torch.nn.functional.mse_loss(lhs, rhs)

    # print(f"Data Loss: {loss_data.item():.6f}, Phys Loss: {loss_phys.item():.6f}")

    total_loss = alpha * loss_data + loss_phys
    return total_loss

# Train one model
def train_one_model(train_csv, test_csv_list, B_pf, alpha=1.0, device="cpu",
                    epochs=400, batch_size=64, lr=5e-3, wd=1e-4, seed=42):
    set_seed(seed)
    ds_fit = DCPFDataset(train_csv, fit_stats=True)
    in_mean, in_std = ds_fit.input_mean, ds_fit.input_std
    tgt_mean, tgt_std = ds_fit.target_mean, ds_fit.target_std
    train_ds = DCPFDataset(train_csv, input_mean=in_mean, input_std=in_std,
                           target_mean=tgt_mean, target_std=tgt_std, fit_stats=False)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_sets = {k:DCPFDataset(v,in_mean,in_std,tgt_mean,tgt_std) for k,v in test_csv_list.items()}

    model = LinearDCPF().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    history = []
    for ep in range(epochs):
        total_ep=0; n=0
        for batch in loader:
            xb, yb = batch["xn"].to(device), batch["yn"].to(device)
            y_pred_n = model(xb)
            loss = compute_losses(batch,y_pred_n,tgt_mean,tgt_std,B_pf,device,alpha)
            opt.zero_grad(); loss.backward(); opt.step()
            total_ep += loss.item(); n+=1
        history.append(total_ep/n)

    # evaluation
    results={}
    for name,ds in {"train":train_ds, **test_sets}.items():
        xb=torch.from_numpy(ds.xn).to(device)
        y_true_n=torch.from_numpy(ds.yn).to(device)
        y_pred_n=model(xb)
        tgt_mean_t=torch.from_numpy(tgt_mean).to(device)
        tgt_std_t=torch.from_numpy(tgt_std).to(device)
        y_true=y_true_n*tgt_std_t+tgt_mean_t
        y_pred=y_pred_n*tgt_std_t+tgt_mean_t
        results[name]=mean_relative_error_percent(y_true,y_pred)
    return results, history

# ---------------------------
# Main
# ---------------------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train1",default="train_set1.csv")
    parser.add_argument("--train2",default="train_set2.csv")
    parser.add_argument("--test1", default="test_set1.csv")
    parser.add_argument("--test2", default="test_set2.csv")
    parser.add_argument("--Bmat",  default="B_pf.mat")
    parser.add_argument("--epochs",type=int,default=400)
    parser.add_argument("--run_a",action="store_true")
    parser.add_argument("--run_b",action="store_true")
    args=parser.parse_args()

    B_pf=loadmat(args.Bmat)["B_pf"].astype(np.float32)
    device="cuda" if torch.cuda.is_available() else "cpu"

    if args.run_a:
        alpha=0.0
        combos=[
            ("Train1→Test1",args.train1,{"Test1":args.test1}),
            ("Train1→Test2",args.train1,{"Test2":args.test2}),
            ("Train2→Test1",args.train2,{"Test1":args.test1}),
            ("Train2→Test2",args.train2,{"Test2":args.test2}),
        ]
        labels=[]; test_errs=[]
        for title,tr,ts in combos:
            res,_=train_one_model(tr,ts,B_pf,alpha,device,epochs=args.epochs)
            labels.append(title); 
            # pick whichever test set exists
            key=[k for k in res.keys() if k.startswith("Test")][0]
            test_errs.append(res[key])

            # print result for this combo
            print(f"{title}: {round(res[key], 6)}")
        plt.figure()
        plt.bar(labels,test_errs)
        plt.ylabel("Test Error (%)")
        plt.title("Part (a): Physics-only α=0")
        plt.savefig("part_a_bar.png",dpi=200)
        plt.close()

    # ----------------------
    # (b) Train2->Test2, alpha in {0,1,10}; log curves for convergence speed
    # ----------------------
    if args.run_b:
        alphas=[0,1,10]; histories=[]; test_errs=[]
        for a in alphas:
            res,hist=train_one_model(args.train2,{"Test2":args.test2},B_pf,a,device,epochs=args.epochs)
            histories.append((a,hist)); test_errs.append((a,res["Test2"]))
        plt.figure()
        for a,hist in histories:
            plt.plot(hist,label=f"α={a}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Part (b): Convergence (Train2→Test2)")
        plt.yscale('log')
        plt.legend()
        plt.savefig("part_b_curve.png",dpi=200)
        plt.close()
        plt.figure()
        xs=[str(a) for a,_ in test_errs]; ys=[v for _,v in test_errs]
        # print these results
        for a,v in test_errs:
            print(f"α={a}: {round(v,6)}")
        plt.bar(xs,ys); plt.ylabel("Test2 Error (%)"); plt.title("Part (b): Final Accuracy")
        plt.savefig("part_b_bar.png",dpi=200); plt.close()

if __name__=="__main__":
    main()