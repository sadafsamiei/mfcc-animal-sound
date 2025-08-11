import argparse
import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path


def list_wavs(root):
    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    files = []
    for c in classes:
        for fp in (root/c).glob("*.wav"):
            files.append((str(fp), c))
    if not files: raise SystemExit(f"No WAVs found under {root}.")
    class_to_idx = {c:i for i,c in enumerate(classes)}
    return files, class_to_idx
def load_wav(path, sr):
    wav, src_sr = torchaudio.load(path)
    wav = wav.mean(0, keepdim=True)
    if src_sr != sr:
        wav = torchaudio.functional.resample(wav, src_sr, sr)
    return wav, sr
def mfcc_tensor(wav, sr, n_mfcc=20, frames=256, n_fft=1024, hop=256):
    mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, melkwargs={"n_fft": n_fft, "hop_length": hop})(wav)
    T = mfcc.shape[-1]
    if T < frames:
        pad = torch.zeros((1, n_mfcc, frames - T), dtype=mfcc.dtype)
        mfcc = torch.cat([mfcc, pad], dim=-1)
    else:
        mfcc = mfcc[..., :frames]
    return mfcc
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--class_map", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--frames", type=int, default=256)
    args = ap.parse_args()
    files, class_to_idx = list_wavs(args.in_dir)
    X_list, y_list = [], []
    for fp, cname in files:
        wav, _ = load_wav(fp, args.sr)
        mfcc = mfcc_tensor(wav, args.sr, n_mfcc=args.n_mfcc, frames=args.frames, n_fft=args.n_fft, hop=args.hop)
        X_list.append(mfcc.numpy())
        y_list.append(class_to_idx[cname])
    X = np.stack(X_list); y = np.array(y_list, dtype=np.int64)
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, X=X, y=y)
    with open(args.class_map, "w") as f: json.dump(class_to_idx, f, indent=2)
    print(f"Saved features {X.shape} to {args.out_npz}")
    print(f"Saved class map to {args.class_map} -> {class_to_idx}")
if __name__ == "__main__":
    main()
