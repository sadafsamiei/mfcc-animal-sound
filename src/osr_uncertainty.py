import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from models.MFCC import MFCCDataset, get_dataloaders, collate_fn
from models.LSTM_Attn import LSTM_Attn
from assets import RESULTS_DIR

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

def osr_uncertainty(threshold=0.7, mc_passes=10, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, label2idx, idx2label = get_dataloaders(batch_size=batch_size)

    val_set = MFCCDataset("val", label2idx=label2idx)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx)).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"), map_location=device))
    model.eval()

    print("\nUncertainty Estimation Results")
    print("=" * 90)
    print(f"{'Index':<6}{'True':<12}{'Pred':<12}{'Entropy':<12}{'MC Var':<12}")
    print("-" * 90)

    entropies, variances, labels = [], [], []

    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)
        ent = entropy(probs).item()

        mc_probs = []
        enable_dropout(model)
        for _ in range(mc_passes):
            logits_mc = model(x)
            probs_mc = F.softmax(logits_mc, dim=1).detach().cpu().numpy()
            mc_probs.append(probs_mc)
        mc_probs = np.stack(mc_probs, axis=0)  
        var_mc = np.var(mc_probs, axis=0).mean()

        pred_idx = torch.argmax(probs, dim=1).item()
        true_idx = y.item()

        if true_idx == -1:
            true_label = "OOSR"
            labels.append(1)  
        else:
            true_label = val_set.idx2label[true_idx]
            labels.append(0)  
        pred_label = idx2label[pred_idx]

        print(f"{i:<6}{true_label:<12}{pred_label:<12}{ent:<12.4f}{var_mc:<12.4f}")

        entropies.append(ent)
        variances.append(var_mc)

    print("=" * 90)
    print(f"Avg Entropy: {np.mean(entropies):.4f}")
    print(f"Avg MC Var : {np.mean(variances):.4f}")

    entropies = np.array(entropies)
    variances = np.array(variances)
    labels = np.array(labels)

    fpr_e, tpr_e, _ = roc_curve(labels, entropies)
    fpr_v, tpr_v, _ = roc_curve(labels, variances)

    auc_e = auc(fpr_e, tpr_e)
    auc_v = auc(fpr_v, tpr_v)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_e, tpr_e, label=f"Entropy (AUC={auc_e:.2f})")
    plt.plot(fpr_v, tpr_v, label=f"MC Variance (AUC={auc_v:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for OOD Detection")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "roc_uncertainty.png")
    plt.savefig(out_path)
    plt.close()
    print(f"ROC curves saved to {out_path}")

if __name__ == "__main__":
    osr_uncertainty(threshold=0.7, mc_passes=10)
