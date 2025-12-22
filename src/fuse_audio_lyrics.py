from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import ensure_dirs, set_seed


def main(N: int = 800):
    set_seed(42)
    p = ensure_dirs()

    # Load lyrics embeddings + meta
    L = np.load(p.lyrics_processed / "lyrics_embeddings.npy")  # (1228,384)
    meta = pd.read_csv(p.lyrics_processed / "lyrics_meta.csv")
    lang = meta["language"].astype(str).values
    y_lang = (lang == "bn").astype(int)

    # Balanced lyrics selection
    idx_bn = np.where(y_lang == 1)[0]
    idx_en = np.where(y_lang == 0)[0]
    n_half = N // 2
    pick_lyr = np.concatenate([
        np.random.choice(idx_bn, size=n_half, replace=False),
        np.random.choice(idx_en, size=n_half, replace=False),
    ])
    np.random.shuffle(pick_lyr)

    Ls = L[pick_lyr]
    y_lang_s = y_lang[pick_lyr]

    # Load audio MFCC
    A = np.load(p.audio_processed / "mfcc_features.npy")  # (1000,80)
    y_genre = np.load(p.audio_processed / "mfcc_labels.npy", allow_pickle=True)

    # sample N audio items
    pick_aud = np.random.choice(np.arange(len(A)), size=N, replace=False)
    As = A[pick_aud]
    y_genre_s = y_genre[pick_aud]

    # Standardize each modality separately then fuse
    Lz = StandardScaler().fit_transform(Ls)
    Az = StandardScaler().fit_transform(As)

    fused = np.concatenate([Az, Lz], axis=1).astype(np.float32)  # (N, 464)

    out_fused = p.data / "fused_features.npy"
    out_meta = p.data / "fused_meta.csv"

    fused_meta = pd.DataFrame({
        "lang_label": y_lang_s,
        "genre_label": y_genre_s.astype(str),
        "lyric_id": meta.loc[pick_lyr, "id"].values,
        "audio_index": pick_aud
    })
    np.save(out_fused, fused)
    fused_meta.to_csv(out_meta, index=False)

    print("Saved fused features:", out_fused)
    print("Fused shape:", fused.shape)
    print("Saved fused meta:", out_meta)


if __name__ == "__main__":
    main()
