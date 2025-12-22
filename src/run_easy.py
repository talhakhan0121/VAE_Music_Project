from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import ensure_dirs, set_seed
from vae import MLPVAE, vae_loss
from evaluation import eval_clustering
from visualize import umap_plot


def train_vae(X: np.ndarray, latent_dim=16, epochs=60, batch_size=64, lr=1e-3, beta=1.0):
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

        if ep % 10 == 0 or ep == 1:
            n = len(x_tensor)
            print(f"Epoch {ep:03d} | loss={total/n:.4f} recon={r_total/n:.4f} kl={k_total/n:.4f}")

    # Extract latent means (mu) as features for clustering (more stable than sampled z)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x_tensor.to(device))
        Z = mu.cpu().numpy()
    return model, Z


def main():
    set_seed(42)
    p = ensure_dirs()

    emb_path = p.lyrics_processed / "lyrics_embeddings.npy"
    meta_path = p.lyrics_processed / "lyrics_meta.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run `python src/embeddings.py` first to create embeddings and meta files.")

    X = np.load(emb_path)  # (N, 384)
    meta = pd.read_csv(meta_path)

    # Standardize for PCA/VAE stability
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results_rows = []

    # ----- Baseline: PCA + KMeans -----
    pca = PCA(n_components=16, random_state=42)
    X_pca = pca.fit_transform(Xs)

    kmeans_pca = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_pca = kmeans_pca.fit_predict(X_pca)

    m_pca = eval_clustering(X_pca, labels_pca)
    results_rows.append({
        "method": "PCA(16)+KMeans(k=2)",
        **m_pca
    })
    umap_plot(X_pca, labels_pca, p.latent_vis / "umap_pca_kmeans.png", "UMAP: PCA(16) + KMeans(k=2)")

    # ----- VAE + KMeans -----
    _, Z = train_vae(Xs, latent_dim=16, epochs=60, batch_size=64, lr=1e-3, beta=1.0)

    kmeans_vae = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_vae = kmeans_vae.fit_predict(Z)

    m_vae = eval_clustering(Z, labels_vae)
    results_rows.append({
        "method": "VAE(latent=16)+KMeans(k=2)",
        **m_vae
    })
    umap_plot(Z, labels_vae, p.latent_vis / "umap_vae_kmeans.png", "UMAP: VAE latent(16) + KMeans(k=2)")

    # Save metrics
    out_csv = p.results / "clustering_metrics.csv"
    pd.DataFrame(results_rows).to_csv(out_csv, index=False)
    print("Saved metrics:", out_csv)

    # Save clustering outputs for inspection
    out_assign = p.results / "cluster_assignments_easy.csv"
    out_df = meta.copy()
    out_df["cluster_pca"] = labels_pca
    out_df["cluster_vae"] = labels_vae
    out_df.to_csv(out_assign, index=False)
    print("Saved assignments:", out_assign)


if __name__ == "__main__":
    main()
