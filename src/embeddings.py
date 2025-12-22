from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from utils import ensure_dirs, set_seed


def build_lyrics_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    max_rows: int | None = None,
) -> tuple[Path, Path]:
    """
    Loads processed lyrics CSV, encodes lyrics into dense vectors, and saves:
      - lyrics_embeddings.npy  (float32, shape [N, D])
      - lyrics_meta.csv        (id, language, title, artist, genre)
    """
    set_seed(42)
    p = ensure_dirs()

    csv_path = p.lyrics_processed / "lyrics_processed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}. Run `python src/dataset.py` first.")

    df = pd.read_csv(csv_path)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    texts = df["lyrics"].astype(str).tolist()

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts)} lyrics (batch_size={batch_size}) ...")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    out_npy = p.lyrics_processed / "lyrics_embeddings.npy"
    np.save(out_npy, emb)

    meta = df[["id", "language", "title", "artist", "genre"]].copy()
    out_meta = p.lyrics_processed / "lyrics_meta.csv"
    meta.to_csv(out_meta, index=False)

    print("Saved:", out_npy)
    print("Saved:", out_meta)
    print("Embeddings shape:", emb.shape)

    return out_npy, out_meta


if __name__ == "__main__":
    # Safe default for your CPU-only setup
    build_lyrics_embeddings(batch_size=32)
