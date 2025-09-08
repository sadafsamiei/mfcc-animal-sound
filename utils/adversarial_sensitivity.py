import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.MFCC import get_dataloaders, collate_fn
from models.LSTM_Attn import LSTM_Attn
from assets import RESULTS_DIR

def fgsm_attack(model, x, y, eps):
    model.train()  
    x.requires_grad = True
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    grad_sign = x.grad.sign()
    x_adv = x + eps * torch.std(x) * grad_sign
    model.eval()  # back to eval mode
    return x_adv.detach()


def adversarial_sensitivity(noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, label2idx, idx2label = get_dataloaders(batch_size=1)

    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx)).to(device)
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"), map_location=device)
    )
    model.eval()

    acc_gaussian, acc_fgsm = [], []

    with torch.no_grad():
        for eps in noise_levels:
            correct, total = 0, 0
            for x, y in val_loader:
                if y.item() == -1:  
                    continue
                x, y = x.to(device), y.to(device)

                x_std = torch.std(x)
                x_perturbed = x + eps * x_std * torch.randn_like(x)

                logits = model(x_perturbed)
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                if preds.item() == y.item():
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else 0
            acc_gaussian.append(acc)
            print(f"[Gaussian] eps={eps:.3f} → acc={acc:.3f}")

    for eps in noise_levels:
        correct, total = 0, 0
        for x, y in val_loader:
            if y.item() == -1:
                continue
            x, y = x.to(device), y.to(device)
            x_adv = fgsm_attack(model, x.clone(), y, eps)

            logits = model(x_adv)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            if preds.item() == y.item():
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        acc_fgsm.append(acc)
        print(f"[FGSM] eps={eps:.3f} → acc={acc:.3f}")

    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, acc_gaussian, marker="o", label="Gaussian Noise")
    plt.plot(noise_levels, acc_fgsm, marker="s", label="FGSM Attack")
    plt.xlabel("Perturbation size (ε)")
    plt.ylabel("Accuracy")
    plt.title("Adversarial Sensitivity (Gaussian vs FGSM)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(RESULTS_DIR, "adversarial_sensitivity.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved adversarial sensitivity plot to {out_path}")

if __name__ == "__main__":
    adversarial_sensitivity()
