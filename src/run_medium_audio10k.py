from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

from vae import MLPVAE, vae_loss
from evaluation import eval_clustering
from visualize import umap_plot
from utils import set_seed, ensure_dirs


def train_vae_return_z(X: np.ndarray, latent_dim=16, epochs=40, batch_size=128, lr=1e-3, beta=1.0):
    device = torch.device("cpu")
    x_tensor = torch.tensor(X, dtype=torch.float32)
    dl = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)

    model = MLPVAE(input_dim=X.shape[1], latent_dim=latent_dim, hidden_dims=(256, 128)).to(device)
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

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            n = len(x_tensor)
            print(f"Epoch {ep:03d} | loss={total/n:.4f} recon={r_total/n:.4f} kl={k_total/n:.4f}")

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x_tensor.to(device))
        Z = mu.cpu().numpy()
    return model, Z


def run_clusterers(rep_name: str, F: np.ndarray, y_true: np.ndarray, vis_prefix: str, p):
    rows = []

    # KMeans
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(F)
    rows.append({"rep": rep_name, "clusterer": "KMeans(k=2)", **eval_clustering(F, labels, y_true=y_true)})
    umap_plot(F, labels, p.latent_vis / f"umap_{vis_prefix}_{rep_name}_kmeans.png",
              f"UMAP: {vis_prefix} {rep_name} + KMeans")

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=2)
    labels = agg.fit_predict(F)
    rows.append({"rep": rep_name, "clusterer": "Agglomerative(k=2)", **eval_clustering(F, labels, y_true=y_true)})
    umap_plot(F, labels, p.latent_vis / f"umap_{vis_prefix}_{rep_name}_agg.png",
              f"UMAP: {vis_prefix} {rep_name} + Agglomerative")

    # DBSCAN: try a few eps values
    for eps in [0.4, 0.6, 0.8, 1.0]:
        db = DBSCAN(eps=eps, min_samples=8)
        labels = db.fit_predict(F)
        # eval_clustering already handles edge cases; still plot
        rows.append({"rep": rep_name, "clusterer": f"DBSCAN(eps={eps},ms=8)", **eval_clustering(F, labels, y_true=y_true)})
        umap_plot(F, labels, p.latent_vis / f"umap_{vis_prefix}_{rep_name}_dbscan_eps{str(eps).replace('.','_')}.png",
                  f"UMAP: {vis_prefix} {rep_name} + DBSCAN(eps={eps})")

    return rows


def main():
    set_seed(42)
    p = ensure_dirs()

    X = np.load(p.audio_processed / "mfcc_features_10k.npy")
    y_true = np.load(p.audio_processed / "mfcc_labels_10k.npy")

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=16, random_state=42)
    X_pca = pca.fit_transform(Xs)

    # VAE latent
    _, Z = train_vae_return_z(Xs, latent_dim=16, epochs=40, batch_size=128, lr=1e-3, beta=1.0)

    rows = []
    rows += run_clusterers("pca16", X_pca, y_true, vis_prefix="audio10k", p=p)
    rows += run_clusterers("vae16", Z, y_true, vis_prefix="audio10k", p=p)

    out_csv = p.results / "clustering_metrics_audio10k_medium.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved metrics:", out_csv)


if __name__ == "__main__":
    main()
