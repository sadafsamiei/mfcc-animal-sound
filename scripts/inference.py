import argparse
import os
import csv
from pathlib import Path
import glob

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import numpy as np
import torchaudio
from assets import PROJECT_ROOT, DATA_DIR, list_assets
from utils.utils_plot import plot_all_species
from utils.utils_audio import mfcc_crop, load_resample
from src.models.MFCC import MFCC_CNN
from src.models.LSTM_Attn import MFCC_LSTM_Attn


def build_model(arch: str, num_classes: int, n_mfcc: int):
    if arch == "mfcc_cnn":
        return MFCC_CNN(num_classes=num_classes)
    elif arch == "mfcc_lstm":
        return MFCC_LSTM_Attn(
            n_mfcc=n_mfcc, num_classes=num_classes,
            hidden=128, num_layers=1, bidir=True, dropout=0.1
        )
    raise ValueError(f"Unknown arch: {arch}")

def prepare_mfcc_tensor(wav_path: str, sr: int, n_mfcc: int, frames: int, device: str):
    y = load_resample(wav_path, sr)
    m = mfcc_crop(y, sr, n_mfcc=n_mfcc, max_frames=frames, device=device)  # (F, T)
    x = m.unsqueeze(0).unsqueeze(0).float()  # (1,1,F,T)
    return x


def main():
    ap = argparse.ArgumentParser(description="Plot audio + (optional) run PyTorch inference on WAVs.")
    ap.add_argument("--do_plot", action="store_true", help="Generate waveform+MFCC figures to results/")
    ap.add_argument("--species", nargs="+", default=["auto"], help="'auto' to discover via assets.py or list: cat lion dog")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--focus_seconds", type=float, default=3.0)
    ap.add_argument("--max_points", type=int, default=24000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--do_predict", action="store_true", help="Run model predictions on WAVs")
    ap.add_argument("--arch", choices=["mfcc_cnn", "mfcc_lstm"], default="mfcc_lstm")
    ap.add_argument("--ckpt", type=str, default="results/best_model.pt")
    ap.add_argument("--class_map", type=str, default="results/classes.json", help="JSON mapping {class_name: index}. Use same file as training/eval.")
    ap.add_argument("--glob", type=str, default=None,  help="Optional glob of WAVs to predict (e.g., 'data/animals/*/*.wav'). " "If omitted, auto-discovers from assets.")
    args = ap.parse_args()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(args.species) == 1 and args.species[0].lower() == "auto":
        assets = list_assets()  # {'cat': [...], 'lion': [...], ...}
        species = sorted(k for k, v in assets.items() if len(v) > 0)
        if not species:
            raise SystemExit("[fatal] No audio files under data/animals/*")
        print(f"[info] Auto species: {species}")
    else:
        species = args.species

    if args.do_plot:
        plot_all_species(
            species=species,
            sr=args.sr,
            n_mfcc=args.n_mfcc,
            frames=args.frames,
            focus_seconds=args.focus_seconds,
            max_points=args.max_points,
            device=args.device,
            out_dir=results_dir,
        )
        print("[plot] Done.")

    if args.do_predict:
        # Load class map
        class_map_path = PROJECT_ROOT / args.class_map if not Path(args.class_map).is_absolute() else Path(args.class_map)
        if not class_map_path.exists():
            raise SystemExit(f"[fatal] class_map not found: {class_map_path}")
        class_to_idx = __import__("json").loads(class_map_path.read_text())
        idx2cls = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)

        use_cuda = (args.device == "cuda") and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = build_model(args.arch, num_classes=num_classes, n_mfcc=args.n_mfcc).to(device)
        ckpt_path = PROJECT_ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        if args.glob:
            wavs = sorted(glob.glob(args.glob))
        else:
            wavs = []
            for sp in species:
                wavs.extend(sorted(glob.glob(str(DATA_DIR / sp / "*.wav"))))
        if not wavs:
            raise SystemExit("[fatal] No WAVs to run prediction on.")

        out_csv = results_dir / "infer_predictions.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wav_path", "pred_class", "pred_index", "probs"])
            with torch.no_grad():
                for wp in wavs:
                    x = prepare_mfcc_tensor(wp, args.sr, args.n_mfcc, args.frames, device=str(device))  # (1,1,F,T)
                    x = x.to(device)
                    if args.arch == "mfcc_lstm":
                        logits, _ = model(x)   # (1, C)
                    else:
                        logits = model(x)      # (1, C)
                    logits = logits[0].detach().cpu().numpy()
                    exps = np.exp(logits - np.max(logits))
                    probs = exps / np.sum(exps)
                    idx = int(np.argmax(probs))
                    pred_cls = idx2cls.get(idx, str(idx))
                    writer.writerow([wp, pred_cls, idx, ";".join(f"{p:.5f}" for p in probs)])
                    print(f"[pred] {wp} -> {pred_cls} (idx={idx})")

        print(f"[predict] wrote {out_csv}")

if __name__ == "__main__":
    main()
