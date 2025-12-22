from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm

from utils import ensure_dirs, set_seed


def extract_mfcc(
    audio_root: Path,
    n_mfcc: int = 40,
    sr: int = 22050,
    max_files: int | None = None,
):
    """
    Extract MFCC mean+std features from all audio files in GTZAN directory.
    Returns:
      X: (N, 2*n_mfcc)
      y: genre labels
      paths: file paths
    """
    X, y, paths = [], [], []

    genre_dirs = sorted([d for d in audio_root.iterdir() if d.is_dir()])

    for genre_dir in genre_dirs:
        genre = genre_dir.name
        audio_files = sorted(list(genre_dir.glob("*.au")))

        if max_files:
            audio_files = audio_files[:max_files]

        for f in tqdm(audio_files, desc=f"Processing {genre}"):
            try:
                audio, _ = librosa.load(f, sr=sr, mono=True)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean = mfcc.mean(axis=1)
                mfcc_std = mfcc.std(axis=1)
                feat = np.concatenate([mfcc_mean, mfcc_std])
                X.append(feat)
                y.append(genre)
                paths.append(str(f))
            except Exception as e:
                print(f"Skipping {f}: {e}")

    return np.array(X, dtype=np.float32), np.array(y), paths


def main():
    set_seed(42)
    p = ensure_dirs()

    audio_root = p.audio_raw / "gtzan"
    assert audio_root.exists(), f"Missing audio directory: {audio_root}"

    print("Extracting MFCC features from:", audio_root)

    X, y, paths = extract_mfcc(audio_root)

    out_feat = p.audio_processed / "mfcc_features.npy"
    out_labels = p.audio_processed / "mfcc_labels.npy"
    out_paths = p.audio_processed / "mfcc_paths.txt"

    np.save(out_feat, X)
    np.save(out_labels, y)

    with open(out_paths, "w", encoding="utf-8") as f:
        for pth in paths:
            f.write(pth + "\n")

    print("Saved MFCC features:", out_feat)
    print("Feature shape:", X.shape)
    print("Saved labels:", out_labels)


if __name__ == "__main__":
    main()
