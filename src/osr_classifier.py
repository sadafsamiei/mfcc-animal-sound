import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.MFCC import MFCCDataset, get_dataloaders, collate_fn
from models.LSTM_Attn import LSTM_Attn
from assets import RESULTS_DIR

CHECK = "\u2714"  # ✔
CROSS = "\u2718"  # ✘

def classify_with_osr(threshold=0.7, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, label2idx, idx2label = get_dataloaders(batch_size=batch_size)

    val_set = MFCCDataset("val", label2idx=label2idx)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx)).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"), map_location=device))
    model.eval()

    print("\nValidation Results with OOSR Detection")
    print("=" * 80)
    print(f"{'File':<20}{'True':<12}{'Predicted':<15}{'Correct'}")
    print("-" * 80)

    correct_in, total_in = 0, 0
    correct_oosr, total_oosr = 0, 0

    val_root = os.path.join(RESULTS_DIR, "features", "val")
    class_dirs = sorted(os.listdir(val_root))
    file_list = []
    for cls in class_dirs:
        for f in sorted(os.listdir(os.path.join(val_root, cls))):
            if f.endswith(".npy"):
                file_list.append((cls, f))

    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)

        true_idx = y.item()
        cls_name, fname = file_list[i]
        if true_idx == -1:
            true_label = f"{cls_name}(OOSR)"  # e.g., frog(OOSR)
        else:
            true_label = val_set.idx2label[true_idx]

        if max_prob.item() < threshold:
            pred_label = "OOSR"
        else:
            pred_label = idx2label[pred_idx.item()]

        if true_idx == -1:  # OOSR
            is_correct = (pred_label == "OOSR")
            total_oosr += 1
            correct_oosr += int(is_correct)
        else:  
            is_correct = (pred_label == true_label)
            total_in += 1
            correct_in += int(is_correct)

        mark = CHECK if is_correct else CROSS
        print(f"{fname:<20}{true_label:<12}{pred_label:<15}{mark}")

    print("=" * 80)
    if total_in > 0:
        acc_in = correct_in / total_in
        print(f"In-distribution Accuracy: {acc_in:.2f} ({correct_in}/{total_in})")
    if total_oosr > 0:
        acc_oosr = correct_oosr / total_oosr
        print(f"OOSR Detection Accuracy: {acc_oosr:.2f} ({correct_oosr}/{total_oosr})")

if __name__ == "__main__":
    classify_with_osr()
