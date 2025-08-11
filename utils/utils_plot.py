from pathlib import Path
from typing import Optional
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import torch

from assets import DATA_DIR, PROJECT_ROOT, list_assets
from utils.utils_audio import (
    load_resample, normalize_rms, loudest_window, decimate_blockmean, mfcc_crop
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]

def plot_species(
    species: str,
    out_dir: Path,
    sr: int,
    n_mfcc: int,
    frames: Optional[int],
    focus_seconds: float,
    max_points: Optional[int],
    device: str,
    equalize_rms: bool = True
):
    sp_dir = DATA_DIR / species
    wavs = sorted(glob.glob(str(sp_dir / f"{species}_*.wav"))) or sorted(glob.glob(str(sp_dir / "*.wav")))
    if not wavs:
        print(f"[WARN] No WAVs for '{species}' in {sp_dir}, skipping.")
        return

    segments, names = [], []
    for p in wavs:
        y = load_resample(p, sr)
        s, e = loudest_window(y, sr, win_sec=focus_seconds)
        y_focus = normalize_rms(y[s:e]) if equalize_rms else y[s:e]
        segments.append(y_focus.numpy()); names.append(Path(p).stem)

    total = sum(len(s) for s in segments)
    keep_total = min(max_points, total) if max_points else total
    keeps = [max(3000, int(len(s)/total * keep_total)) for s in segments]
    overflow = sum(keeps) - keep_total
    for i in range(len(keeps)):
        if overflow <= 0: break
        can_sub = max(0, keeps[i]-1500)
        sub = min(can_sub, overflow)
        keeps[i] -= sub; overflow -= sub

    fig, ax = plt.subplots(figsize=(14, 4))
    start = 0
    for i, (s, k, name) in enumerate(zip(segments, keeps, names)):
        s_ds = decimate_blockmean(np.asarray(s), k)
        end = start + len(s_ds)
        ax.plot(np.arange(start, end), s_ds, linewidth=0.9, color=COLORS[i % len(COLORS)], label=name)
        start = end
    ax.set_title(f"{species.capitalize()} Waveforms (Loudest {focus_seconds:.1f}s, Colored)")
    ax.set_xlabel("Downsampled index"); ax.set_ylabel("Amplitude (RMS-norm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, start); ax.xaxis.set_major_locator(NullLocator()); ax.xaxis.set_major_formatter(NullFormatter())
    ax.set_xticks([]); ax.legend(loc="upper right", frameon=False, fontsize=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wave = out_dir / f"{species}_waveforms_concat_colored.png"
    fig.savefig(out_wave, dpi=110); plt.close(fig)
    print(f"[OK] Saved {out_wave}")

    # MFCC per file
    mfcc_dir = out_dir / "mfcc_plots" / species
    mfcc_dir.mkdir(parents=True, exist_ok=True)
    for p in wavs:
        y = load_resample(p, sr); s, e = loudest_window(y, sr, win_sec=focus_seconds)
        m = mfcc_crop(y[s:e], sr, n_mfcc=n_mfcc, max_frames=frames, device=device)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(m, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"MFCC (loudest {focus_seconds:.1f}s): {Path(p).name}")
        ax.set_xlabel("Frames"); ax.set_ylabel("MFCC Index")
        fig.colorbar(im, ax=ax, label="Coefficient")
        out_m = mfcc_dir / f"mfcc_{Path(p).stem}.png"
        fig.savefig(out_m, dpi=120); plt.close(fig)
        print(f"[OK] Saved {out_m}")

def plot_all_species(
    species,
    sr: int,
    n_mfcc: int,
    frames: Optional[int],
    focus_seconds: float,
    max_points: Optional[int],
    device: str,
    out_dir: Optional[Path] = None
):
    if out_dir is None:
        out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in species:
        plot_species(sp, out_dir, sr, n_mfcc, frames, focus_seconds, max_points, device)
