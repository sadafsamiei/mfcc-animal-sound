#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT_FS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FS) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_FS))

from assets import PROJECT_ROOT
from src.models.MFCC import MFCC_CNN
from src.models.LSTM_Attn import MFCC_LSTM_Attn

def build_model(arch: str, num_classes: int, n_mfcc: int):
    if arch == "mfcc_cnn":
        return MFCC_CNN(num_classes=num_classes)
    elif arch == "mfcc_lstm":
        return MFCC_LSTM_Attn(n_mfcc=n_mfcc, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained checkpoint .pt")
    ap.add_argument("--features", required=True, help="Path to features .npz")
    ap.add_argument("--class_map", required=True, help="Path to classes.json (use SAME file as training)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--arch", choices=["mfcc_cnn", "mfcc_lstm"], default="mfcc_lstm")
    ap.add_argument("--normalize_cm", action="store_true", help="Row-normalize confusion matrix")
    args = ap.parse_args()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    class_map_path = Path(args.class_map)
    class_to_idx = json.loads(class_map_path.read_text())
    idx2cls = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    feats_path = Path(args.features)
    data = np.load(feats_path)
    X = torch.tensor(data["X"]).float()      
    y = torch.tensor(data["y"]).long()
    n_mfcc = X.shape[2]

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = build_model(args.arch, num_classes=num_classes, n_mfcc=n_mfcc)
    ckpt = torch.load(args.model, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda
    )

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            if args.arch == "mfcc_lstm":
                logits, _ = model(xb)
            else:
                logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            gts.append(yb.numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    target_names = [idx2cls[i] for i in range(num_classes)]

    report = classification_report(gts, preds, target_names=target_names, digits=3)
    print(report)
    (results_dir / "eval_report.txt").write_text(report)

    labels = list(range(num_classes))
    cm = confusion_matrix(gts, preds, labels=labels)

    if args.normalize_cm:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot = (cm / row_sums).astype(float)
        fmt = ".2f"
        title = "Confusion Matrix (row-normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    disp = ConfusionMatrixDisplay(cm_plot, display_labels=[idx2cls[i] for i in labels])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="viridis", colorbar=True, values_format=fmt)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    out_png = results_dir / "confusion_matrix.png"
    plt.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
