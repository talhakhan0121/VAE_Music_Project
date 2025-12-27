from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (prevents Tkinter crash)

import matplotlib.pyplot as plt
import umap


def umap_plot(X: np.ndarray, labels: np.ndarray, out_path: Path, title: str):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X2 = reducer.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
