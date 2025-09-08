import os
import torch
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from models.MFCC import MFCCDataset, get_dataloaders, collate_fn
from models.LSTM_Attn import LSTM_Attn
from assets import RESULTS_DIR, SR

def explain_attention(split="val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, label2idx, idx2label = get_dataloaders()
    val_set = MFCCDataset(split, label2idx=label2idx)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx)).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"), map_location=device))
    model.eval()

    out_dir = os.path.join(RESULTS_DIR, "explainability")
    os.makedirs(out_dir, exist_ok=True)

    class_samples = {}

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            out, _ = model.lstm(x)
            _, attn_weights = model.attn(out)
            attn_weights = attn_weights.squeeze(0).cpu().numpy()

            seq_len = x.size(1)
            duration_sec = seq_len / SR * 512  # hop_length=512
            times = np.linspace(0, duration_sec, seq_len)

            true_idx = y.item()
            class_name = idx2label[true_idx] if true_idx != -1 else "OOSR"

            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append((x.cpu().numpy(), times, attn_weights))

    for cls, samples in class_samples.items():
        n = len(samples)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4), squeeze=False)

        for ax, (spec, times, attn) in zip(axes.flatten(), samples):
            librosa.display.specshow(spec.squeeze(0).T, sr=SR, x_axis="time", cmap="magma", ax=ax)
            ax.plot(times, attn * np.max(spec), color="cyan", label="Attention Weights")
            ax.set_title(f"{cls} sample")
            ax.set_xlabel("Time (s)")
            ax.legend()

        for j in range(len(samples), rows * cols):
            axes.flatten()[j].axis("off")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"class_samples_{cls}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved combined sample-level attention plots for {cls} → {out_path}")

if __name__ == "__main__":
    explain_attention(split="val")
