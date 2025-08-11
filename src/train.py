import argparse, os, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from assets import PROJECT_ROOT, DATA_DIR, list_assets
from models.MFCC import MFCC_CNN
from models.LSTM_Attn import MFCC_LSTM_Attn


def spec_augment(x, freq_mask=8, time_mask=16):
    B, C, F, T = x.shape
    x_aug = x.clone()
    for b in range(B):
        if F > freq_mask:
            f0 = random.randint(0, F - freq_mask)
            x_aug[b, :, f0:f0 + freq_mask, :] = 0
        if T > time_mask:
            t0 = random.randint(0, T - time_mask)
            x_aug[b, :, :, t0:t0 + time_mask] = 0
    return x_aug


def build_model(arch, num_classes, n_mfcc):
    if arch == "mfcc_cnn":
        return MFCC_CNN(num_classes=num_classes)
    elif arch == "mfcc_lstm":
        return MFCC_LSTM_Attn(n_mfcc=n_mfcc, num_classes=num_classes, hidden=128, num_layers=1, bidir=True, dropout=0.1)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def ensure_class_map(arg_value: str, results_dir: Path) -> Path:

    if arg_value.lower() == "auto":
        out = results_dir / "classes.auto.json"
    else:
        p = Path(arg_value)
        if p.exists():
            return p
        out = results_dir / "classes.auto.json"

    assets = list_assets() 
    species = sorted(k for k, v in assets.items() if len(v) > 0)
    if not species:
        raise SystemExit("[fatal] No species with audio found under data/animals/. Cannot build class map.")
    class_to_idx = {sp: i for i, sp in enumerate(species)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(class_to_idx, indent=2))
    print(f"[train] Built class map from assets -> {out} : {class_to_idx}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features .npz produced by preprocessing")
    ap.add_argument("--class_map", default="auto", help="Path to classes.json or 'auto' to derive from assets")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ckpt", type=str, default="results/best_model.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", choices=["mfcc_cnn", "mfcc_lstm"], default="mfcc_lstm")
    args = ap.parse_args()

    project_root = PROJECT_ROOT
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    feats_path = Path(args.features)
    if not feats_path.exists():
        raise SystemExit(f"[fatal] Features file not found: {feats_path}")
    data = np.load(feats_path)
    X = torch.tensor(data["X"]).float()   
    y = torch.tensor(data["y"]).long()
    n_mfcc = X.shape[2]

    n = len(X)
    if n < 2:
        raise SystemExit(f"[fatal] Need at least 2 samples, got {n}.")
    idx = np.arange(n); np.random.shuffle(idx)
    tr = max(1, int(0.8 * n))
    tr_idx, va_idx = idx[:tr], idx[tr:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    class_map_path = ensure_class_map(args.class_map, results_dir)
    class_to_idx = json.loads(Path(class_map_path).read_text())
    num_classes = len(class_to_idx)
    if num_classes < 2:
        raise SystemExit(f"[fatal] Only {num_classes} class found in {class_map_path}. "
                         "Add at least one more class (e.g., 'lion', 'dog').")

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = build_model(args.arch, num_classes=num_classes, n_mfcc=n_mfcc).to(device)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=use_cuda)
    val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch_size,
                              num_workers=2, pin_memory=use_cuda)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_f1 = -1.0
    tr_losses, va_losses = [], []

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            xb_aug = spec_augment(xb)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_cuda):
                if args.arch == "mfcc_lstm":
                    logits, _ = model(xb_aug)    
                else:
                    logits = model(xb_aug)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_tr_loss += loss.item() * xb.size(0)

        epoch_tr_loss /= len(train_loader.dataset)
        tr_losses.append(epoch_tr_loss)

        model.eval()
        epoch_va_loss = 0.0
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                if args.arch == "mfcc_lstm":
                    logits, _ = model(xb)
                else:
                    logits = model(xb)
                loss = criterion(logits, yb)
                epoch_va_loss += loss.item() * xb.size(0)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                gts.append(yb.cpu().numpy())

        epoch_va_loss /= len(val_loader.dataset)
        va_losses.append(epoch_va_loss)

        preds = np.concatenate(preds); gts = np.concatenate(gts)
        f1 = f1_score(gts, preds, average="macro")

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={epoch_tr_loss:.4f}  val_loss={epoch_va_loss:.4f}  val_F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)

    print(f"Best macro-F1={best_f1:.3f}. Saved {ckpt_path}")

    loss_csv = results_dir / "loss_curve.csv"
    with loss_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss"])
        for i,(tl,vl) in enumerate(zip(tr_losses, va_losses), start=1):
            w.writerow([i, tl, vl])

    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(tr_losses)+1), tr_losses, label="train")
    plt.plot(range(1, len(va_losses)+1), va_losses, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss curve ({args.arch})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=130)
    plt.close()
    print(f"Saved {results_dir/'loss_curve.png'} and {loss_csv}")

if __name__ == "__main__":
    main()
