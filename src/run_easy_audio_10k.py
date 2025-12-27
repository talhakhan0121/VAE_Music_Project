from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from vae import MLPVAE, vae_loss
from evaluation import eval_clustering
from visualize import umap_plot
from utils import set_seed, ensure_dirs

def train_vae(X: np.ndarray, latent_dim=16, epochs=40, batch_size=128, lr=1e-3, beta=1.0):
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

        if ep % 10 == 0 or ep == 1:
            n = len(x_tensor)
            print(f"Epoch {ep:03d} | loss={total/n:.4f} recon={r_total/n:.4f} kl={k_total/n:.4f}")

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x_tensor.to(device))
        Z = mu.cpu().numpy()
    return model, Z



def main():
    set_seed(42)
    p = ensure_dirs()

    X = np.load(p.audio_processed / "mfcc_features_10k.npy")
    y_true = np.load(p.audio_processed / "mfcc_labels_10k.npy")

    # Standardize for PCA/VAE stability
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results_rows = []

    # ----- Baseline: PCA + KMeans -----
    pca = PCA(n_components=16, random_state=42)
    X_pca = pca.fit_transform(Xs)

    kmeans_pca = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_pca = kmeans_pca.fit_predict(X_pca)

    m_pca = eval_clustering(X_pca, labels_pca, y_true=y_true)
    results_rows.append({"method": "Audio10k PCA(16)+KMeans(k=2)", **m_pca})
    umap_plot(X_pca, labels_pca, p.latent_vis / "umap_audio10k_pca16_kmeans.png",
              "UMAP: Audio10k PCA(16)+KMeans(k=2)")

    # ----- VAE + KMeans -----
    _, Z = train_vae(Xs, latent_dim=16, epochs=40, batch_size=128, lr=1e-3, beta=1.0)

    kmeans_vae = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_vae = kmeans_vae.fit_predict(Z)

    m_vae = eval_clustering(Z, labels_vae, y_true=y_true)
    results_rows.append({"method": "Audio10k VAE(latent=16)+KMeans(k=2)", **m_vae})
    umap_plot(Z, labels_vae, p.latent_vis / "umap_audio10k_vae16_kmeans.png",
              "UMAP: Audio10k VAE(16)+KMeans(k=2)")

    out_csv = p.results / "clustering_metrics_audio10k_easy.csv"
    pd.DataFrame(results_rows).to_csv(out_csv, index=False)
    print("Saved metrics:", out_csv)


if __name__ == "__main__":
    main()
