from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import re

import pandas as pd
from tqdm import tqdm

from utils import ensure_dirs


TEXT_COL_CANDIDATES = [
    "lyrics", "lyric", "text", "content", "Lyric", "Lyrics", "Text", "song_lyrics"
]
TITLE_COL_CANDIDATES = [
    "title", "song", "song_name", "track_name", "name", "Title", "Song"
]
ARTIST_COL_CANDIDATES = [
    "artist", "singer", "Artist", "Singer"
]
GENRE_COL_CANDIDATES = [
    "genre", "Genre", "category"
]


def _guess_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # try case-insensitive match
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    # remove very long repeated punctuation, keep Bangla script
    t = re.sub(r"[•\t\r\n]+", " ", t)
    return t


def load_lyrics_folder(folder: Path, language: str) -> pd.DataFrame:
    """
    Loads all CSV files from a folder, guesses columns, and returns a standardized dataframe:
    id, title, artist, genre, language, lyrics
    """
    rows = []
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        text_col = _guess_column(df, TEXT_COL_CANDIDATES)
        if text_col is None:
            raise ValueError(f"Could not find lyrics/text column in {csv_path.name}. "
                             f"Columns: {list(df.columns)}")

        title_col = _guess_column(df, TITLE_COL_CANDIDATES)
        artist_col = _guess_column(df, ARTIST_COL_CANDIDATES)
        genre_col = _guess_column(df, GENRE_COL_CANDIDATES)

        for i, r in df.iterrows():
            lyrics = clean_lyrics(r.get(text_col, ""))
            if len(lyrics) < 20:
                continue  # skip too short
            title = str(r.get(title_col, "")) if title_col else ""
            artist = str(r.get(artist_col, "")) if artist_col else ""
            genre = str(r.get(genre_col, "")) if genre_col else ""

            rows.append({
                "id": f"{language}_{csv_path.stem}_{i}",
                "title": title,
                "artist": artist,
                "genre": genre,
                "language": language,
                "lyrics": lyrics
            })

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["lyrics"]).reset_index(drop=True)
    return out


def build_lyrics_dataset() -> Path:
    p = ensure_dirs()

    bangla_dir = p.lyrics_raw / "bangla"
    english_dir = p.lyrics_raw / "english"

    df_bn = load_lyrics_folder(bangla_dir, language="bn")
    df_en = load_lyrics_folder(english_dir, language="en")

    # balance for clustering fairness (optional but recommended)
    n = min(len(df_bn), len(df_en))
    df_bn = df_bn.sample(n=n, random_state=42).reset_index(drop=True)
    df_en = df_en.sample(n=n, random_state=42).reset_index(drop=True)

    df = pd.concat([df_bn, df_en], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_path = p.lyrics_processed / "lyrics_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved processed lyrics dataset: {out_path}")
    print("Rows:", len(df), "| bn:", (df["language"] == "bn").sum(), "| en:", (df["language"] == "en").sum())
    return out_path


if __name__ == "__main__":
    build_lyrics_dataset()
