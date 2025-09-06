#!/usr/bin/env python3
"""
Preprocess audio into MFCC features (optionally with Δ and ΔΔ) and save to NPZ.

- Traverses class folders under --in_dir (e.g., data/animals/{cat,dog,...}/*.wav)
- Loads audio (mono + resample), extracts MFCCs with torchaudio
- Optional Δ and ΔΔ features, concatenated along feature axis
- Per-clip CMVN (cepstral mean/variance normalization)
- Pads/trims to a fixed number of frames
- Saves:
    X: (N, 1, F, T) float32
    y: (N,) int64
- Writes class_to_idx JSON

Recommended for OSR & better accuracy:
    --feature_mode mfcc_deltas  --cmvn
    --n_mfcc 40 --frames 384
"""

import argparse
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio


# ------------------------------ helpers ------------------------------ #
def list_wavs(root: Path):
    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    files = []
    for c in classes:
        for fp in sorted((root / c).glob("*.wav")):
            files.append((str(fp), c))
    if not files:
        raise SystemExit(f"[fatal] No WAVs found under {root}. Expect folders like {root}/cat/*.wav")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return files, class_to_idx


def load_wav(path: str, sr: int) -> torch.Tensor:
    y, srr = torchaudio.load(path)        # (C, T)
    y = y.mean(0, keepdim=True)           # mono (1, T)
    if srr != sr:
        y = torchaudio.functional.resample(y, srr, sr)
    return y                              # (1, T)


def pad_trim_time(x: torch.Tensor, frames: int) -> torch.Tensor:
    # x: (1, F, T)
    T = x.shape[-1]
    if T == frames:
        return x
    if T < frames:
        pad = torch.zeros((x.shape[0], x.shape[1], frames - T), dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)
    # T > frames: trim from the start (deterministic). If you prefer center crop:
    # start = max(0, (T - frames) // 2); end = start + frames
    return x[..., :frames]


def compute_features(
    wav: torch.Tensor,
    sr: int,
    n_mfcc: int,
    frames: int,
    n_fft: int,
    hop: int,
    feature_mode: str = "mfcc_deltas",
    cmvn: bool = True,
) -> torch.Tensor:
    """
    Returns (1, F, T), where F = n_mfcc (mfcc) or 3*n_mfcc (mfcc_deltas).
    """
    tx = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft, "hop_length": hop, "center": True},
    )
    mfcc = tx(wav)               # (1, n_mfcc, T')

    if feature_mode == "mfcc":
        feat = mfcc
    elif feature_mode == "mfcc_deltas":
        d1 = torchaudio.functional.compute_deltas(mfcc)
        d2 = torchaudio.functional.compute_deltas(d1)
        feat = torch.cat([mfcc, d1, d2], dim=1)   # (1, 3*n_mfcc, T')
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    if cmvn:
        mu = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True).clamp_min(1e-5)
        feat = (feat - mu) / std

    feat = pad_trim_time(feat, frames)  # (1, F, frames)
    return feat


# ------------------------------ main ------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root folder with class subfolders (e.g., data/animals)")
    ap.add_argument("--out_npz", required=True, help="Output NPZ path (features)")
    ap.add_argument("--class_map", required=True, help="Output JSON path for class_to_idx")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--feature_mode", choices=["mfcc", "mfcc_deltas"], default="mfcc_deltas",
                    help="Use plain MFCCs or MFCC + Δ + ΔΔ")
    ap.add_argument("--no_cmvn", action="store_true", help="Disable cepstral mean/var normalization")
    ap.add_argument("--limit_per_class", type=int, default=None, help="Optional cap per class for quick tests")
    args = ap.parse_args()

    # Keep CPU threads modest for clusters
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)

    files, class_to_idx = list_wavs(args.in_dir)

    # (optional) cap per-class
    if args.limit_per_class is not None:
        by_cls = defaultdict(list)
        for fp, c in files:
            by_cls[c].append(fp)
        files = []
        for c, lst in by_cls.items():
            lst = sorted(lst)[: args.limit_per_class]
            files.extend([(fp, c) for fp in lst])

    # Process
    X_list, y_list = [], []
    counts = defaultdict(int)

    for fp, cname in files:
        wav = load_wav(fp, args.sr)                                     # (1, T)
        feat = compute_features(wav, args.sr, args.n_mfcc, args.frames,
                                args.n_fft, args.hop,
                                feature_mode=args.feature_mode,
                                cmvn=not args.no_cmvn)                   # (1, F, frames)
        # add channel dim to match (1, F, T) → (1, 1, F, T)
        feat = feat.unsqueeze(0)
        X_list.append(feat.cpu().numpy().astype(np.float32))
        y_list.append(class_to_idx[cname])
        counts[cname] += 1

    X = np.concatenate(X_list, axis=0)                 # (N, 1, F, T)
    y = np.array(y_list, dtype=np.int64)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X, y=y)

    class_map_path = Path(args.class_map)
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    class_map_path.write_text(json.dumps(class_to_idx, indent=2))

    F = X.shape[2]
    print(f"[prep] Saved features: {X.shape}  (F={F}) -> {out_npz}")
    print(f"[prep] Saved class map -> {class_map_path} : {class_to_idx}")
    print("[prep] Per-class counts:", dict(counts))


if __name__ == "__main__":
    main()
