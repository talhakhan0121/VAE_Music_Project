# VAE-Based Unsupervised Clustering for Hybrid Language Music

Course Project for **Neural Networks (CSE425)** — Unsupervised Learning (VAE / β-VAE).  
Author: **Talha Islam Khan**, BRAC University (CSE)

This project builds an unsupervised pipeline inspired by **Variational Autoencoders (VAE)** to learn latent representations for clustering:
- **Lyrics** (English + Bangla)
- **Audio** (MFCC features)
- **Fused multi-modal** (audio + lyrics), including **β-VAE**

---

## Repository Structure (as required)

project/
data/ # gitignored (datasets not uploaded)
audio/
lyrics/
notebooks/
exploratory.ipynb
src/
vae.py
dataset.py
clustering.py
evaluation.py
results/
latent_visualization/
clustering_metrics.csv
README.md
requirements.txt


---

## What’s Implemented

### Easy Task (Lyrics-only)
- VAE for feature extraction
- KMeans on latent space
- PCA + KMeans baseline
- UMAP visualizations
- Metrics: Silhouette, Calinski–Harabasz

### Medium Task
- Audio MFCC extraction + audio-only clustering
- Multi-modal fused representation (audio + lyrics)
- Clusterers: KMeans, Agglomerative, DBSCAN
- Metrics: Silhouette, CH, Davies–Bouldin, ARI, NMI, Purity (where labels exist)

### Hard Task
- β-VAE experiments (β = 1, 4, 10)
- Multi-modal clustering (audio + lyrics)
- Full quantitative comparison + UMAP visualizations

---

## Key Results (Summary)

**Lyrics-only (KMeans k=2):**
- PCA: Silhouette **0.5105**, CH **1490.54**
- VAE: Silhouette **0.6880**, CH **3171.62**

**Audio-only (KMeans k=10):**
- PCA: Silhouette **0.1209**, DB **1.9206**
- VAE: Silhouette **0.2719**, DB **1.1618**

**Fused (KMeans k=2):**
- PCA: Silhouette **0.4358**, DB **0.9903**
- VAE (β=1): Silhouette **0.7540**, DB **0.3849**
- β-VAE (β=4): CH **4531.28**, DB **0.3693**, ARI **0.9850**, NMI **0.9647**

---

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
