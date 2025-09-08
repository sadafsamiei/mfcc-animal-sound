import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.MFCC import get_dataloaders, collate_fn
from models.LSTM_Attn import LSTM_Attn
from assets import RESULTS_DIR

def pgd_attack(model, x, y, eps=0.05, alpha=0.01, steps=5):
    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    for _ in range(steps):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        delta = torch.clamp(x_adv - x, min=-eps * torch.std(x), max=eps * torch.std(x))
        x_adv = (x + delta).detach()
        x_adv.requires_grad = True

    return x_adv.detach()

def train_model(epochs=20, lr=1e-3, eps=0.05, alpha=0.01, steps=5, adv_weight=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, label2idx, idx2label = get_dataloaders(batch_size=4)

    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits_clean = model(x)
            loss_clean = F.cross_entropy(logits_clean, y)

            x_adv = pgd_attack(model, x.clone(), y, eps=eps, alpha=alpha, steps=steps)
            logits_adv = model(x_adv)
            loss_adv = F.cross_entropy(logits_adv, y)

            loss = adv_weight * loss_clean + (1 - adv_weight) * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                mask = (y != -1)  
                correct += (preds[mask] == y[mask]).sum().item()
                total += mask.sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        acc = correct / total if total > 0 else 0

        print(f"Epoch {epoch}/{epochs} | Train Loss={avg_train_loss:.4f} | "
              f"Val Loss={avg_val_loss:.4f} | Val Acc={acc:.3f}")

    os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"))

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss (PGD Adversarial Training)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    train_model(epochs=20, lr=1e-3, eps=0.05, alpha=0.01, steps=5, adv_weight=0.5)
