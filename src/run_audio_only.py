from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from utils import ensure_dirs, set_seed
from evaluation import eval_clustering
from visualize import umap_plot
from vae import MLPVAE, vae_loss

import torch
from torch.utils.data import DataLoader, TensorDataset


def train_vae_latents(X: np.ndarray, latent_dim=16, epochs=60, batch_size=64, lr=1e-3, beta=1.0):
    device = torch.device("cpu")
    x_tensor = torch.tensor(X, dtype=torch.float32)
    dl = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)

    model = MLPVAE(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            x_hat, mu, logvar, z = model(xb)
            loss, recon, kl = vae_loss(xb, x_hat, mu, logvar, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)

        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | loss={total/len(x_tensor):.4f}")

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_tensor.to(device))
        Z = mu.cpu().numpy()
    return Z


def run_clusterers(feature_name: str, F: np.ndarray, y_true: np.ndarray, p):
    rows = []

    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = km.fit_predict(F)
    rows.append({"feature": feature_name, "clusterer": "KMeans(k=10)", **eval_clustering(F, labels, y_true=y_true)})
    umap_plot(F, labels, p.latent_vis / f"umap_audio_{feature_name}_kmeans.png", f"UMAP: audio {feature_name} + KMeans(k=10)")

    agg = AgglomerativeClustering(n_clusters=10)
    labels = agg.fit_predict(F)
    rows.append({"feature": feature_name, "clusterer": "Agglomerative(k=10)", **eval_clustering(F, labels, y_true=y_true)})
    umap_plot(F, labels, p.latent_vis / f"umap_audio_{feature_name}_agg.png", f"UMAP: audio {feature_name} + Agglomerative(k=10)")

    for eps in [0.3, 0.5, 0.8, 1.0]:
        db = DBSCAN(eps=eps, min_samples=8)
        labels = db.fit_predict(F)
        rows.append({"feature": feature_name, "clusterer": f"DBSCAN(eps={eps},ms=8)", **eval_clustering(F, labels, y_true=y_true)})
        umap_plot(F, labels, p.latent_vis / f"umap_audio_{feature_name}_dbscan_eps{str(eps).replace('.','_')}.png",
                  f"UMAP: audio {feature_name} + DBSCAN(eps={eps})")

    return rows


def main():
    set_seed(42)
    p = ensure_dirs()

    X = np.load(p.audio_processed / "mfcc_features.npy")  # (1000, 80)
    y = np.load(p.audio_processed / "mfcc_labels.npy", allow_pickle=True)  # genres as strings

    # encode genres as integers
    genres = sorted(np.unique(y).tolist())
    g2i = {g: i for i, g in enumerate(genres)}
    y_true = np.array([g2i[g] for g in y], dtype=int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_pca = PCA(n_components=16, random_state=42).fit_transform(Xs)
    Z = train_vae_latents(Xs, latent_dim=16, epochs=60, batch_size=64, beta=1.0)

    rows = []
    rows += run_clusterers("pca16", X_pca, y_true, p)
    rows += run_clusterers("vae16", Z, y_true, p)

    out = pd.DataFrame(rows)
    out_path = p.results / "clustering_metrics_audio_only.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
