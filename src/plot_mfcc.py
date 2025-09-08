import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from assets import DATA_DIR, RESULTS_DIR, SR, N_MFCC

def plot_mfcc_and_waveforms(split="train"):
    split_dir = os.path.join(DATA_DIR, split)
    out_dir = os.path.join(RESULTS_DIR, "plots", split)
    os.makedirs(out_dir, exist_ok=True)

    for label in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for wav in os.listdir(class_dir):
            if not wav.endswith(".wav"):
                continue
            wav_path = os.path.join(class_dir, wav)
            y, sr = librosa.load(wav_path, sr=SR)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr//2)
            log_S = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(1, 2, figsize=(16, 5))

            img1 = librosa.display.specshow(
                mfcc, x_axis="time", sr=sr, cmap="viridis", ax=ax[0]
            )
            fig.colorbar(img1, ax=ax[0], format="%+2.0f", label="MFCC Value")
            ax[0].set_title(f"MFCC - {label} - {wav}")
            ax[0].set_ylabel("MFCC Coefficients")
            ax[0].set_xlabel("Time (s)")

            img2 = librosa.display.specshow(
                log_S, x_axis="time", y_axis="mel", sr=sr, cmap="viridis", ax=ax[1]
            )
            fig.colorbar(img2, ax=ax[1], format="%+2.0f dB", label="Log Power")
            ax[1].set_title(f"Log-Mel Spectrogram - {label} - {wav}")
            ax[1].set_ylabel("Mel Frequency (Hz)")
            ax[1].set_xlabel("Time (s)")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{label}_{wav.replace('.wav','_features.png')}"))
            plt.close()

        plt.figure(figsize=(12, 8))

        wav_files = [w for w in os.listdir(class_dir) if w.endswith(".wav")]
        cmap = plt.get_cmap("viridis", len(wav_files))  

        offset = 0
        for idx, wav in enumerate(wav_files):
            wav_path = os.path.join(class_dir, wav)
            y, sr = librosa.load(wav_path, sr=SR)

            y = y / np.max(np.abs(y))

            plt.plot(y + offset, color=cmap(idx), label=wav)
            offset += 2  

        plt.title(f"Waveforms - {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{label}_waveforms.png"))
        plt.close()

def main():
    plot_mfcc_and_waveforms("train")

if __name__ == "__main__":
    main()
