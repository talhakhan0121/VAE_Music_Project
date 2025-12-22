from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def _n_clusters_effective(labels: np.ndarray) -> int:
    """Count clusters excluding DBSCAN noise (-1)."""
    labels = np.asarray(labels)
    uniq = set(np.unique(labels).tolist())
    uniq.discard(-1)
    return len(uniq)


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total = 0
    for c in np.unique(y_pred):
        if c == -1:
            continue  # ignore noise for purity
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        true_labels = y_true[idx]
        counts = np.bincount(true_labels.astype(int))
        total += counts.max() if len(counts) > 0 else 0

    denom = np.sum(y_pred != -1)
    if denom == 0:
        return float("nan")
    return float(total / denom)


def eval_clustering(X: np.ndarray, labels: np.ndarray, y_true: np.ndarray | None = None) -> dict:
    X = np.asarray(X)
    labels = np.asarray(labels)

    out = {}
    k_eff = _n_clusters_effective(labels)
    n = len(labels)

    # For metrics that require >=2 clusters and < n clusters
    valid_partition = (k_eff >= 2) and (k_eff < n)

    # Silhouette: if DBSCAN noise exists, sklearn can still compute, but it's often misleading.
    # We'll compute only if valid_partition and at least 2 non-noise clusters.
    out["silhouette"] = float(silhouette_score(X, labels)) if valid_partition else float("nan")
    out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels)) if valid_partition else float("nan")
    out["davies_bouldin"] = float(davies_bouldin_score(X, labels)) if valid_partition else float("nan")

    if y_true is not None:
        out["ARI"] = float(adjusted_rand_score(y_true, labels))
        out["NMI"] = float(normalized_mutual_info_score(y_true, labels))
        out["purity"] = float(cluster_purity(y_true, labels))
    else:
        out["ARI"] = float("nan")
        out["NMI"] = float("nan")
        out["purity"] = float("nan")

    out["clusters_effective"] = int(k_eff)
    out["noise_frac"] = float(np.mean(labels == -1)) if np.any(labels == -1) else 0.0

    return out
