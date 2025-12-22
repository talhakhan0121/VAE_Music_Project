from __future__ import annotations
import os
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


@dataclass(frozen=True)
class Paths:
    root: Path = Path(r"D:\VAE_Music_Project")

    data: Path = root / "data"
    results: Path = root / "results"
    src: Path = root / "src"

    lyrics_raw: Path = data / "lyrics" / "raw"
    lyrics_processed: Path = data / "lyrics" / "processed"

    audio_raw: Path = data / "audio" / "raw"
    audio_processed: Path = data / "audio" / "processed"

    latent_vis: Path = results / "latent_visualization"


def ensure_dirs() -> Paths:
    p = Paths()
    dirs = [
        p.results, p.latent_vis,
        p.lyrics_processed,
        p.audio_processed,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # harmless if no cuda
    os.environ["PYTHONHASHSEED"] = str(seed)
