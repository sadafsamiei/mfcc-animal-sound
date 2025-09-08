import torch
from models.LSTM_Attn import LSTM_Attn
from models.MFCC import get_dataloaders
from assets import RESULTS_DIR
import os

def export_onnx():
    _, _, label2idx, _ = get_dataloaders()
    model = LSTM_Attn(input_dim=13, hidden_dim=64, num_classes=len(label2idx))
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "models", "lstm_attn.pth"), map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 100, 13) 
    out_path = os.path.join(RESULTS_DIR, "models", "lstm_attn.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["mfcc"],
        output_names=["logits"],
        dynamic_axes={"mfcc": {1: "seq_len"}}, 
        opset_version=13
    )
    print(f"ONNX model saved to {out_path}")

if __name__ == "__main__":
    export_onnx()
