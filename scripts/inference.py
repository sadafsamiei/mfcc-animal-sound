import argparse
from pathlib import Path
import sys
from typing import Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from assets import PROJECT_ROOT as PRJ
from utils.utils_plot import plot_all_species  
from utils.utils_audio import (
    load_resample, normalize_rms, loudest_window, decimate_blockmean
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd","#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

def discover_species(data_dir: Path) -> List[str]:
    subs = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and any(p.glob("*.wav"))])
    if subs:
        return subs
    names = set()
    for wav in data_dir.glob("*.wav"):
        n = wav.stem.split("_")[0]
        if n:
            names.add(n)
    return sorted(names)

def plot_combined_species(
    species_list: List[str],
    data_dir: Path,
    out_dir: Path,
    sr: int,
    focus_seconds: float,
    max_points: Optional[int] = 24000,
    equalize_rms: bool = True,
) -> None:

    wave_dir = out_dir / "waveforms_plots"
    wave_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    start = 0

    if max_points is None:
        per_sp_budget = None
    else:
        per_sp_budget = max(3000, max_points // max(1, len(species_list)))

    for i, sp in enumerate(species_list):
        sp_dir = data_dir / sp
        wavs = sorted(list(sp_dir.glob(f"{sp}_*.wav"))) or sorted(list(sp_dir.glob("*.wav")))
        if not wavs:
            print(f"[WARN] No WAVs for '{sp}' in {sp_dir}, skipping.")
            continue

        segments = []
        for p in wavs:
            y = load_resample(p, sr)
            s, e = loudest_window(y, sr, win_sec=focus_seconds)
            seg = y[s:e]
            if equalize_rms:
                seg = normalize_rms(seg)
            segments.append(seg.numpy())

        combined = np.concatenate(segments) if len(segments) > 1 else segments[0]
        combined_ds = decimate_blockmean(combined, per_sp_budget)

        end = start + len(combined_ds)
        ax.plot(
            np.arange(start, end),
            combined_ds,
            linewidth=0.9,
            color=COLORS[i % len(COLORS)],
            label=sp,
        )
        start = end

    ax.set_title("Combined Waveforms by Species (loudest segments per file)")
    ax.set_xlabel("Downsampled index")
    ax.set_ylabel("Amplitude (RMS-norm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, start)
    ax.xaxis.set_major_locator(NullLocator())
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.set_xticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    out_png = wave_dir / "combined_animals_waveforms.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PRJ))
    ap.add_argument("--do_plot", action="store_true", help="Create per-species plots")
    ap.add_argument("--species", nargs="+", default=["auto"],help="List species or 'auto' to discover from data/animals")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--focus_seconds", type=float, default=3.0)
    ap.add_argument("--max_points", type=int, default=24000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot_combined", action="store_true",help="Also save results/waveforms_plots/combined_animals_waveforms.png")
    args = ap.parse_args()

    root = Path(args.root)
    data_dir = root / "data" / "animals"
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.species) == 1 and args.species[0].lower() == "auto":
        species_list = discover_species(data_dir)
    else:
        species_list = args.species

    if args.do_plot:
        plot_all_species(
            species=species_list,
            sr=args.sr,
            n_mfcc=args.n_mfcc,
            frames=args.frames,
            focus_seconds=args.focus_seconds,
            max_points=args.max_points,
            device=args.device,
            out_dir=out_dir,
        )

    if args.plot_combined and len(species_list) >= 2:
        plot_combined_species(
            species_list=species_list,
            data_dir=data_dir,
            out_dir=out_dir,
            sr=args.sr,
            focus_seconds=args.focus_seconds,
            max_points=args.max_points,
            equalize_rms=True,
        )

if __name__ == "__main__":
    main()
