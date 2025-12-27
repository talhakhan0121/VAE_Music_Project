import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa

# -----------------------------
# Configuration
# -----------------------------
SR = 22050
DURATION = 10.0          # seconds per clip (truncate/pad)
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048
MAX_FILES_PER_LANG = 5000   # 5000 en + 5000 bn = 10k total

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "audio" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EN_ROOT = PROJECT_ROOT / "data" / "audio" / "raw" / "english_audio" / "LibriSpeech" / "train-clean-100"
BN_ROOT = PROJECT_ROOT / "data" / "audio" / "raw" / "bangla_audio"

# -----------------------------
def find_audio_files(root: Path, exts=(".flac", ".wav", ".mp3", ".au")):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)

def load_clip(path: Path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    target_len = int(SR * DURATION)

    # pad or truncate
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def mfcc_feature(y):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    # Take mean and std over time -> fixed length
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)  # (2*N_MFCC,)
    return feat.astype(np.float32)

# -----------------------------
def process_language(lang_name: str, root: Path, max_files: int):
    files = find_audio_files(root)
    if len(files) == 0:
        raise RuntimeError(f"No audio files found under {root}")

    files = files[:max_files]
    X = []
    meta_rows = []

    for fp in tqdm(files, desc=f"Processing {lang_name}", total=len(files)):
        try:
            y = load_clip(fp)
            feat = mfcc_feature(y)
            X.append(feat)
            meta_rows.append({"path": str(fp), "lang": lang_name})
        except Exception as e:
            # skip problematic files
            continue

    X = np.stack(X, axis=0)
    meta = pd.DataFrame(meta_rows)
    return X, meta

def main():
    print("EN root:", EN_ROOT)
    print("BN root:", BN_ROOT)

    X_en, meta_en = process_language("en", EN_ROOT, MAX_FILES_PER_LANG)
    X_bn, meta_bn = process_language("bn", BN_ROOT, MAX_FILES_PER_LANG)

    X = np.concatenate([X_en, X_bn], axis=0)
    meta = pd.concat([meta_en, meta_bn], axis=0).reset_index(drop=True)

    # labels: en=0, bn=1
    y = (meta["lang"] == "bn").astype(int).to_numpy()

    print("Final shapes:", X.shape, y.shape, meta.shape)

    np.save(OUT_DIR / "mfcc_features_10k.npy", X)
    np.save(OUT_DIR / "mfcc_labels_10k.npy", y)
    meta.to_csv(OUT_DIR / "mfcc_meta_10k.csv", index=False)

    print("Saved:")
    print(" -", OUT_DIR / "mfcc_features_10k.npy")
    print(" -", OUT_DIR / "mfcc_labels_10k.npy")
    print(" -", OUT_DIR / "mfcc_meta_10k.csv")

if __name__ == "__main__":
    main()
