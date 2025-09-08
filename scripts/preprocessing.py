import os
import librosa
import numpy as np
from tqdm import tqdm
from assets import DATA_DIR, RESULTS_DIR, SR, N_MFCC

def extract_and_save_mfcc(split):
    split_dir = os.path.join(DATA_DIR, split)
    out_dir = os.path.join(RESULTS_DIR, "features", split)
    os.makedirs(out_dir, exist_ok=True)

    for label in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir):
            continue

        out_class_dir = os.path.join(out_dir, label)
        os.makedirs(out_class_dir, exist_ok=True)

        for wav in tqdm(os.listdir(class_dir), desc=f"{split}-{label}"):
            if not wav.endswith(".wav"):
                continue

            wav_path = os.path.join(class_dir, wav)
            y, sr = librosa.load(wav_path, sr=SR)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            np.save(os.path.join(out_class_dir, wav.replace(".wav", ".npy")), mfcc)

def main():
    for split in ["train", "val"]:
        extract_and_save_mfcc(split)

if __name__ == "__main__":
    main()
