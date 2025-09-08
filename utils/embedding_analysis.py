import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from models.MFCC import get_dataloaders, collate_fn
from models.LSTM_Attn import LSTM_Attn
from assets import RESULTS_DIR

def plot_confusion_matrix(batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, label2idx, idx2label = get_dataloaders(batch_size=batch_size)

    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx)).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"), map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(idx2label.keys()))
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f",
                xticklabels=[idx2label[i] for i in idx2label],
                yticklabels=[idx2label[i] for i in idx2label],
                cmap='viridis')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")

if __name__ == "__main__":
    plot_confusion_matrix()
