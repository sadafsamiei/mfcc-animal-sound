import torch
from models.MFCC import get_dataloaders
from models.LSTM_Attn import LSTM_Attn
from assets import RESULTS_DIR
import os

def evaluate(batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, label2idx, idx2label = get_dataloaders(batch_size=batch_size)

    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx)).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"), map_location=device))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Validation Accuracy: {acc:.2f}")

if __name__ == "__main__":
    evaluate()
