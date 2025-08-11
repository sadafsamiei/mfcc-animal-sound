import argparse
import json
import os
import glob
from pathlib import Path
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from assets import PROJECT_ROOT, DATA_DIR, list_assets
from models.MFCC import MFCC_CNN


def build_class_map_from_assets() -> dict:
    assets = list_assets()  
    species = sorted(k for k, v in assets.items() if len(v) > 0)
    if not species:
        raise SystemExit("[fatal] No species with audio found under data/animals/. Cannot build class map.")
    return {sp: i for i, sp in enumerate(species)}

def ensure_class_map(arg_value: str, results_dir: Path) -> Path:

    if arg_value and arg_value.lower() != "auto":
        p = Path(arg_value)
        if p.exists():
            return p
    out = results_dir / "classes.auto.json"
    class_to_idx = build_class_map_from_assets()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(class_to_idx, indent=2))
    print(f"[onnx] Built class map from assets -> {out} : {class_to_idx}")
    return out

def load_class_map(path_or_auto: str, results_dir: Path) -> dict:
    p = ensure_class_map(path_or_auto or "auto", results_dir)
    return json.loads(Path(p).read_text())


def export_onnx(ckpt, out_onnx, num_classes, n_mfcc=20, frames=256):
    model = MFCC_CNN(num_classes=num_classes)
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 1, n_mfcc, frames)  
    out_onnx = Path(out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_onnx),
        input_names=['input'], output_names=['logits'],
        opset_version=13,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}
    )
    print(f"[export] wrote {out_onnx}")


def preprocess_mfcc(wav_path, sr, n_mfcc, frames, n_fft=1024, hop=256):
    wav, src_sr = torchaudio.load(wav_path)
    wav = wav.mean(0, keepdim=True)  
    if src_sr != sr:
        wav = torchaudio.functional.resample(wav, src_sr, sr)
    tx = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'hop_length': hop, 'center': True}
    )
    mfcc = tx(wav)  # (1, n_mfcc, T)
    T = mfcc.shape[-1]
    if T < frames:
        pad = torch.zeros((1, n_mfcc, frames - T), dtype=mfcc.dtype)
        mfcc = torch.cat([mfcc, pad], dim=-1)
    else:
        mfcc = mfcc[..., :frames]
    x = mfcc.numpy().astype(np.float32)  # (1, n_mfcc, frames)
    return x

def choose_providers():
    prov = []
    try:
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            prov.append('CUDAExecutionProvider')
    except Exception:
        pass
    prov.append('CPUExecutionProvider')
    return prov

def infer_onnx(onnx_path, wav_path, class_map, sr, n_mfcc, frames):
    providers = choose_providers()
    sess = ort.InferenceSession(onnx_path, providers=providers)

    x = preprocess_mfcc(wav_path, sr, n_mfcc, frames)  
    x = x[np.newaxis, ...]  
    logits = sess.run(None, {'input': x})[0][0]  
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)

    idx2cls = {v: k for k, v in class_map.items()} if class_map else {i: f"class_{i}" for i in range(len(probs))}
    for i, p in enumerate(probs):
        print(f"{idx2cls.get(i, str(i))}: {p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', help="Path to trained .pt (CNN)")
    ap.add_argument('--out_onnx', help="Destination ONNX path")
    ap.add_argument('--arch', default='mfcc_cnn', choices=['mfcc_cnn'], help="Export only supports the CNN")
    ap.add_argument('--n_mfcc', type=int, default=20)
    ap.add_argument('--frames', type=int, default=256)
    ap.add_argument('--num_classes', type=int, default=None)
    ap.add_argument('--class_map', type=str, default="auto", help="'auto' to build from assets, or path to classes.json")
    ap.add_argument('--run_infer', action='store_true')
    ap.add_argument('--onnx', type=str, help="Path to ONNX model for inference")
    ap.add_argument('--wav', type=str, help="WAV path for inference")
    ap.add_argument('--sr', type=int, default=16000)
    args = ap.parse_args()

    project_root = PROJECT_ROOT
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.run_infer:
        cls_map = load_class_map(args.class_map, results_dir)

        wav_path = args.wav
        if not wav_path:
            cands = sorted(glob.glob(str(DATA_DIR / "*" / "*.wav")))
            if not cands:
                raise SystemExit("[fatal] No WAV found under data/animals/. Provide --wav.")
            wav_path = cands[0]
            print(f"[infer] Using WAV: {wav_path}")

        if not args.onnx:
            raise SystemExit("Provide --onnx path for inference.")
        infer_onnx(args.onnx, wav_path, cls_map, args.sr, args.n_mfcc, args.frames)
        return

    if not args.ckpt or not args.out_onnx:
        raise SystemExit('For export, provide --ckpt and --out_onnx')

    if args.num_classes is None:
        cls_map = load_class_map(args.class_map, results_dir)
        args.num_classes = len(cls_map)
        if args.num_classes < 2:
            raise SystemExit(f"[fatal] Need >=2 classes for softmax export, got {args.num_classes}.")

    export_onnx(args.ckpt, args.out_onnx, args.num_classes, args.n_mfcc, args.frames)


if __name__ == '__main__':
    main()
