from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torchaudio

__all__ = [
    "load_resample",
    "normalize_rms",
    "loudest_window",
    "decimate_blockmean",
    "mfcc_crop",
]

def load_resample(path, sr: int) -> torch.Tensor:
    y, srr = torchaudio.load(str(path))
    y = y.mean(0, keepdim=True)  # mono
    if srr != sr:
        y = torchaudio.functional.resample(y, srr, sr)
    return y.squeeze(0)

def normalize_rms(y: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    rms = torch.sqrt(torch.clamp((y**2).mean(), min=1e-12))
    scale = target_rms / rms
    return (y * scale).clamp(-1.0, 1.0)

def loudest_window(y: torch.Tensor, sr: int, win_sec: float = 3.0):
    L = len(y)
    w = int(sr * win_sec)
    if L <= w or w <= 0:
        return 0, L
    y2 = (y**2).cpu().numpy()
    win = np.ones(w, dtype=np.float32)
    e = np.convolve(y2, win, mode="valid")
    start = int(e.argmax())
    end = start + w
    return start, end

def decimate_blockmean(x: np.ndarray, keep: Optional[int]) -> np.ndarray:
    n = len(x)
    if keep is None or n <= keep:
        return x
    block = int(np.ceil(n / keep))
    cut = (n // block) * block
    x = x[:cut]
    return x.reshape(-1, block).mean(axis=1)

def mfcc_crop(
    y: torch.Tensor,
    sr: int,
    n_mfcc: int = 20,
    max_frames: Optional[int] = None,
    n_fft: int = 1024,
    hop: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    y = y.to(device)
    tx = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft, "hop_length": hop, "center": True},
    ).to(device)
    m = tx(y.unsqueeze(0)).squeeze(0)  # (n_mfcc, T')
    if max_frames is not None and m.shape[-1] > max_frames:
        m = m[:, :max_frames]
    return m.cpu()
