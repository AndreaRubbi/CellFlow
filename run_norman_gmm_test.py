"""Quick GMM-CellFlow test on Norman 2019 (subset) with gge-eval evaluation.

Usage
-----
    python run_norman_gmm_test.py

Produces:
    - evaluation metrics via gge-eval
    - plots in ./norman_gmm_results/
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import matplotlib
matplotlib.use("Agg")       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

# ── 1. LOAD NORMAN 2019 ──────────────────────────────────────────────────────
print("Loading Norman 2019 …")
DATA_PATH = os.path.expanduser("~/.cache/tcfm/norman_2019_scperturb.h5ad")
if os.path.exists(DATA_PATH):
    adata = ad.read_h5ad(DATA_PATH)
else:
    import pertpy
    adata = pertpy.data.norman_2019()

print(f"  raw: {adata.shape}")

# ── 2. PREPROCESS ────────────────────────────────────────────────────────────
# Basic filtering & normalisation
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=50)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.pca(adata, n_comps=50)
print(f"  after preprocessing (2000 HVGs for training): {adata.shape}")

# Keep a subset of top DEGs for gge-eval (distributional metrics are O(n_genes))
N_EVAL_GENES = 500
sc.pp.highly_variable_genes(adata, n_top_genes=N_EVAL_GENES, flavor="seurat_v3", layer="counts")
eval_gene_mask = adata.var["highly_variable"].copy()
print(f"  will evaluate on top {N_EVAL_GENES} HVGs")

# ── 3. SUBSET: keep control + top-N single-gene perturbations ───────────────
N_PERTS = 15  # fast test
pert_col = "perturbation"
adata.obs["is_control"] = adata.obs[pert_col] == "control"

# pick top single-gene perts by cell count (nperts==1)
single = adata.obs.loc[adata.obs["nperts"] == 1, pert_col]
top_perts = single.value_counts().head(N_PERTS).index.tolist()
keep_mask = adata.obs["is_control"] | adata.obs[pert_col].isin(top_perts)
adata = adata[keep_mask.values].copy()
adata.obs_names_make_unique()
adata.obs[pert_col] = adata.obs[pert_col].cat.remove_unused_categories()
print(f"  subset ({N_PERTS} perts + control): {adata.shape}")
print(f"  perturbations: {adata.obs[pert_col].value_counts().to_dict()}")

# ── 4. PREPARE FOR CELLFLOW ─────────────────────────────────────────────────
# CellFlow needs: categorical perturbation column, a control bool, embeddings
# We'll use one-hot for perturbation embeddings (no external reps)
# Create one-hot embeddings stored in adata.uns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
all_perts = adata.obs[pert_col].cat.categories.tolist()
le.fit(all_perts)
emb_dim = len(all_perts)
pert_emb = {}
for p in all_perts:
    vec = np.zeros((emb_dim, 1), dtype=np.float32)
    vec[le.transform([p])[0], 0] = 1.0
    pert_emb[p] = vec
adata.uns["pert_emb"] = pert_emb

# Make sure dtypes are right
adata.obs[pert_col] = adata.obs[pert_col].astype("category")
adata.obs["is_control"] = adata.obs["is_control"].astype(bool)

print("  Data ready for CellFlow.")

# ── 5. TRAIN CELLFLOW WITH GMM SOURCE ───────────────────────────────────────
import cellflow
import optax

print("\nPreparing CellFlow (GMM source) …")
cf = cellflow.model.CellFlow(adata, solver="otfm")
cf.prepare_data(
    sample_rep="X_pca",
    control_key="is_control",
    perturbation_covariates={"pert": [pert_col]},
    perturbation_covariate_reps={"pert": "pert_emb"},
)

cf.prepare_model(
    condition_embedding_dim=64,
    hidden_dims=(256, 256),
    decoder_dims=(256, 256),
    time_encoder_dims=(256, 256),
    source_type="gmm",
    gmm_kwargs={
        "num_modes": 16,
        "variance": 0.01,
        "temperature": 0.5,
        "means_hidden_dims": (128,),
        "logits_hidden_dims": (128,),
    },
    geodesic_loss_weight=1.0,
    optimizer=optax.adam(1e-4),
)

NUM_ITERS = 5000     # longer run
BATCH_SIZE = 256
print(f"Training for {NUM_ITERS} iterations …")
cf.train(num_iterations=NUM_ITERS, batch_size=BATCH_SIZE)
print("Training done.")

# ── 6. PREDICT FROM GMM FOR EACH PERTURBATION ───────────────────────────────
print("\nGenerating predictions …")
N_SAMPLES = 200   # cells per condition

# Build covariate df for all non-control perturbations
perts_to_predict = [p for p in top_perts]
cov_rows = []
for p in perts_to_predict:
    cov_rows.append({pert_col: p, "is_control": False})
cov_df = pd.DataFrame(cov_rows)
cov_df[pert_col] = pd.Categorical(cov_df[pert_col], categories=all_perts)

preds = cf.predict_from_gmm(cov_df, n_samples=N_SAMPLES)
print(f"  Generated {len(preds)} conditions, {N_SAMPLES} cells each.")

# ── 7. BUILD GENERATED ANNDATA FOR EVALUATION ───────────────────────────────
gen_arrays = []
gen_perts = []
for key, arr in preds.items():
    gen_arrays.append(arr)
    if isinstance(key, tuple):
        pname = key[0]
    else:
        pname = key
    gen_perts.extend([pname] * arr.shape[0])

gen_X_pca = np.vstack(gen_arrays)
# Reconstruct to gene space for gge-eval (project back from PCA)
pca_loadings = adata.varm["PCs"]  # (n_genes, n_pcs)
pca_mean = np.asarray(adata.X.mean(axis=0)).ravel() if hasattr(adata.X, "toarray") else adata.X.mean(axis=0)
gen_X = gen_X_pca @ pca_loadings.T + pca_mean

gen_obs = pd.DataFrame({pert_col: pd.Categorical(gen_perts, categories=all_perts)})
gen_adata = ad.AnnData(X=gen_X.astype(np.float32), obs=gen_obs)
gen_adata.var_names = adata.var_names.copy()
gen_adata.obsm["X_pca"] = gen_X_pca.astype(np.float32)

# Build real (perturbed only) adata for comparison
real_mask = (~adata.obs["is_control"]) & (adata.obs[pert_col].isin(perts_to_predict))
real_idx = np.where(real_mask.values)[0]
real_adata = adata[real_idx].copy()
real_adata.X = np.asarray(real_adata.X.todense()) if hasattr(real_adata.X, "todense") else np.asarray(real_adata.X)
real_adata.X = real_adata.X.astype(np.float32)

print(f"  Real (perturbed): {real_adata.shape}")
print(f"  Generated:        {gen_adata.shape}")

# Subset to top eval genes for gge-eval (distributional metrics are expensive per-gene)
real_adata_eval = real_adata[:, eval_gene_mask].copy()
gen_adata_eval = gen_adata[:, eval_gene_mask].copy()
print(f"  Eval subset (top {N_EVAL_GENES} genes): real={real_adata_eval.shape}, gen={gen_adata_eval.shape}")

# ── 8. GGE-EVAL ─────────────────────────────────────────────────────────────
print("\nRunning gge-eval …")
import gge
import time

OUT_DIR = "norman_gmm_results"
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
results = gge.evaluate(
    real_data=real_adata_eval,
    generated_data=gen_adata_eval,
    condition_columns=[pert_col],
    metrics=["mse", "pearson", "spearman", "r_squared",
             "wasserstein_1", "wasserstein_2", "mmd", "energy"],
    include_multivariate=True,
    verbose=True,
)
print(f"  gge-eval completed in {time.time()-t0:.1f}s")

print("\n── Evaluation Summary ──")
print(results.summary())

# ── 9. PLOTS ─────────────────────────────────────────────────────────────────
print("\nGenerating plots …")

# 9a. Use gge-eval visualizer
data_loader = gge.GeneExpressionDataLoader(
    real_data=real_adata_eval,
    generated_data=gen_adata_eval,
    condition_columns=[pert_col],
)
data_loader.load()
vis = gge.EvaluationVisualizer(results)

# Boxplot of metrics per condition
fig = vis.boxplot_metrics(figsize=(14, 5))
fig.savefig(os.path.join(OUT_DIR, "boxplot_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# Scatter grid: real vs generated per condition (top conditions)
try:
    fig = vis.scatter_grid(max_conditions=min(12, N_PERTS), ncols=4, figsize_per_panel=(4, 4))
    fig.savefig(os.path.join(OUT_DIR, "scatter_grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"  scatter_grid failed: {e}")

# Embedding plot (PCA, real vs generated colored by type)
try:
    fig = vis.embedding_by_condition(data_loader, method="pca", max_samples=3000, figsize=(14, 10))
    fig.savefig(os.path.join(OUT_DIR, "embedding_by_condition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"  embedding_by_condition failed: {e}")

# Per-condition scatter: real vs generated for each perturbation
for p in perts_to_predict:
    try:
        fig = vis.scatter_real_vs_generated(condition=p, figsize=(8, 6), alpha=0.4)
        fig.savefig(os.path.join(OUT_DIR, f"scatter_{p}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  scatter for {p} failed: {e}")

# 9b. Custom UMAP: real vs generated per condition
print("  Computing UMAP for real vs generated …")
combined = ad.concat(
    [real_adata, gen_adata],
    keys=["real", "generated"],
    label="source",
)
sc.pp.pca(combined, n_comps=30)
sc.pp.neighbors(combined, n_pcs=30)
sc.tl.umap(combined)

fig, axes = plt.subplots(3, 5, figsize=(25, 15))
axes = axes.ravel()

for i, p in enumerate(perts_to_predict[:15]):
    ax = axes[i]
    # Background: all cells in grey
    umap_all = combined.obsm["X_umap"]
    ax.scatter(umap_all[:, 0], umap_all[:, 1], c="lightgrey", s=1, alpha=0.1, rasterized=True)
    # Real cells for this condition
    real_mask_p = (combined.obs["source"] == "real") & (combined.obs[pert_col] == p)
    gen_mask_p = (combined.obs["source"] == "generated") & (combined.obs[pert_col] == p)
    umap_real = combined[real_mask_p].obsm["X_umap"]
    umap_gen = combined[gen_mask_p].obsm["X_umap"]
    ax.scatter(umap_real[:, 0], umap_real[:, 1], c="tab:blue", s=6, alpha=0.6, label="real")
    ax.scatter(umap_gen[:, 0], umap_gen[:, 1], c="tab:red", s=6, alpha=0.6, label="generated")
    ax.set_title(p, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=8, loc="upper left")

plt.suptitle("UMAP: Real (blue) vs GMM-CellFlow Generated (red) per perturbation", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "umap_per_condition.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll results saved to {OUT_DIR}/")
print("Done!")
