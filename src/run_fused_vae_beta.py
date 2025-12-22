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


def train_vae_latents(X: np.ndarray, latent_dim=16, epochs=80, batch_size=64, lr=1e-3, beta=1.0):
    device = torch.device("cpu")
    x_tensor = torch.tensor(X, dtype=torch.float32)
    dl = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)

    model = MLPVAE(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total, r_total, k_total = 0.0, 0.0, 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            x_hat, mu, logvar, z = model(xb)
            loss, recon, kl = vae_loss(xb, x_hat, mu, logvar, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            r_total += float(recon.item()) * len(xb)
            k_total += float(kl.item()) * len(xb)

        if ep % 20 == 0 or ep == 1:
            n = len(x_tensor)
            print(f"[beta={beta}] Epoch {ep:03d} | loss={total/n:.4f} recon={r_total/n:.4f} kl={k_total/n:.4f}")

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_tensor.to(device))
        Z = mu.cpu().numpy()
    return Z


def run_clusterers(tag: str, F: np.ndarray, y_true: np.ndarray, p):
    rows = []

    # We cluster into 2 groups for language separation
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(F)
    rows.append({"rep": tag, "clusterer": "KMeans(k=2)", **eval_clustering(F, labels, y_true=y_true)})
    umap_plot(F, labels, p.latent_vis / f"umap_fused_{tag}_kmeans.png", f"UMAP: fused {tag} + KMeans(k=2)")

    agg = AgglomerativeClustering(n_clusters=2)
    labels = agg.fit_predict(F)
    rows.append({"rep": tag, "clusterer": "Agglomerative(k=2)", **eval_clustering(F, labels, y_true=y_true)})
    umap_plot(F, labels, p.latent_vis / f"umap_fused_{tag}_agg.png", f"UMAP: fused {tag} + Agglomerative(k=2)")

    for eps in [0.4, 0.6, 0.8]:
        db = DBSCAN(eps=eps, min_samples=8)
        labels = db.fit_predict(F)
        rows.append({"rep": tag, "clusterer": f"DBSCAN(eps={eps},ms=8)", **eval_clustering(F, labels, y_true=y_true)})
        umap_plot(F, labels, p.latent_vis / f"umap_fused_{tag}_dbscan_eps{str(eps).replace('.','_')}.png",
                  f"UMAP: fused {tag} + DBSCAN(eps={eps})")

    return rows


def main():
    set_seed(42)
    p = ensure_dirs()

    fused = np.load(p.data / "fused_features.npy")  # (800, 464)
    meta = pd.read_csv(p.data / "fused_meta.csv")

    # Language labels for evaluation (bn=1, en=0)
    y_lang = meta["lang_label"].astype(int).values

    # Standardize fused
    Xs = StandardScaler().fit_transform(fused)

    rows = []

    # Baseline PCA
    X_pca = PCA(n_components=16, random_state=42).fit_transform(Xs)
    rows += run_clusterers("pca16", X_pca, y_lang, p)

    # Standard VAE (beta=1)
    Z_vae = train_vae_latents(Xs, latent_dim=16, epochs=80, batch_size=64, beta=1.0)
    rows += run_clusterers("vae16_beta1", Z_vae, y_lang, p)

    # Beta-VAE (Hard task) beta=4 and beta=10
    Z_b4 = train_vae_latents(Xs, latent_dim=16, epochs=80, batch_size=64, beta=4.0)
    rows += run_clusterers("vae16_beta4", Z_b4, y_lang, p)

    Z_b10 = train_vae_latents(Xs, latent_dim=16, epochs=80, batch_size=64, beta=10.0)
    rows += run_clusterers("vae16_beta10", Z_b10, y_lang, p)

    out = pd.DataFrame(rows)
    out_path = p.results / "clustering_metrics_fused_betaVAE.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
