"""Compare Vanilla CellFlow vs GMM-CellFlow on Norman 2019.

Proper held-out evaluation:
  - 105 single-gene perturbations (>=50 cells each)
  - 30/20/50 train/dev/test split on perturbations
  - Models trained on control + train perturbations only
  - ALL metrics computed on top-20 DEGs per test condition
  - DEG correlation (LFC-based) on test set
  - Losses and gradient norms logged to Weights & Biases

Data loaded from scverse (same as TopologicalFlowMathing repo).

Usage
-----
    python run_vanilla_vs_gmm.py
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import warnings
warnings.filterwarnings("ignore")

import time
import json
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import argparse

parser = argparse.ArgumentParser(description="Vanilla vs GMM CellFlow on Norman 2019")
parser.add_argument("--split", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Split index (0-3) for different train/test splits")
args = parser.parse_args()
SPLIT = args.split

# ===============================================================================
# CONFIGURATION
# ===============================================================================
PERT_COL = "perturbation_name"
CONTROL_VALUE = "control"
MIN_CELLS_PER_PERT = 50
TRAIN_FRACTION = 0.30
DEV_FRACTION = 0.20
# TEST_FRACTION = 1 - TRAIN_FRACTION - DEV_FRACTION = 0.50
N_TOP_DEGS = 20
NUM_ITERS = 35000
BATCH_SIZE = 256
N_SAMPLES = 2000
LR = 5e-4
SEED = 42 + SPLIT
GMM_WARMUP = 5000    # VF-only warmup before GMM training starts
GMM_COOLDOWN = 3000  # VF-only cooldown after GMM training ends
GMM_LR = 1e-3
GEODESIC_WEIGHT = 0.005
SOURCE_MATCH_WEIGHT = 2.0
OUT_DIR = f"comparison_results/split_{SPLIT}"
WANDB_PROJECT = "cellflow-norman"

print("=" * 70)
print(f"VANILLA vs GMM CELLFLOW - Norman 2019 (Split {SPLIT}, seed={SEED})")
print("=" * 70)

# ===============================================================================
# 1. LOAD DATA
# ===============================================================================
print("\n[1] Loading Norman 2019 from scverse ...")
CACHE_DIR = os.path.expanduser("~/.cache/tcfm/data")
DATA_PATH = os.path.join(CACHE_DIR, "norman_2019.h5ad")

if os.path.exists(DATA_PATH):
    adata_raw = ad.read_h5ad(DATA_PATH)
    print(f"  Loaded from cache: {adata_raw.shape}")
else:
    import sys
    sys.path.insert(0, os.path.expanduser("~/Desktop/TopologicalFlowMathing"))
    from tcfm.data.perturbation import norman_2019
    adata_raw = norman_2019()
    print(f"  Downloaded: {adata_raw.shape}")

# ===============================================================================
# 2. PREPROCESS
# ===============================================================================
print("\n[2] Preprocessing ...")
adata = adata_raw.copy()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=50)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.pca(adata, n_comps=50)
print(f"  After preprocessing: {adata.shape}")

# ===============================================================================
# 3. SELECT PERTURBATIONS & TRAIN/TEST SPLIT
# ===============================================================================
print("\n[3] Selecting perturbations & splitting ...")
adata.obs["is_control"] = adata.obs[PERT_COL] == CONTROL_VALUE

single_mask = (
    ~adata.obs[PERT_COL].astype(str).str.contains(r"\+", regex=True)
    & (adata.obs[PERT_COL] != CONTROL_VALUE)
)
pert_counts = adata.obs.loc[single_mask, PERT_COL].value_counts()
eligible_perts = pert_counts[pert_counts >= MIN_CELLS_PER_PERT].index.tolist()
print(f"  Eligible single-gene perturbations (>={MIN_CELLS_PER_PERT} cells): {len(eligible_perts)}")

rng = np.random.default_rng(SEED)
shuffled = rng.permutation(eligible_perts).tolist()
n_train = max(1, int(len(shuffled) * TRAIN_FRACTION))
n_dev = max(1, int(len(shuffled) * DEV_FRACTION))
train_perts = sorted(shuffled[:n_train])
dev_perts = sorted(shuffled[n_train:n_train + n_dev])
test_perts = sorted(shuffled[n_train + n_dev:])
print(f"  Train perturbations: {len(train_perts)}")
print(f"  Dev perturbations:   {len(dev_perts)}")
print(f"  Test perturbations:  {len(test_perts)}")
print(f"  Dev perts:  {dev_perts}")
print(f"  Test perts: {test_perts}")

all_perts_list = sorted(eligible_perts)
keep = adata.obs["is_control"] | adata.obs[PERT_COL].isin(all_perts_list)
adata = adata[keep.values].copy()
adata.obs_names_make_unique()
adata.obs[PERT_COL] = adata.obs[PERT_COL].astype("category").cat.remove_unused_categories()
print(f"  Full dataset: {adata.shape}")

train_keep = adata.obs["is_control"] | adata.obs[PERT_COL].isin(train_perts)
adata_train = adata[train_keep.values].copy()
adata_train.obs_names_make_unique()
adata_train.obs[PERT_COL] = adata_train.obs[PERT_COL].astype("category").cat.remove_unused_categories()
print(f"  Train adata: {adata_train.shape}")

# ===============================================================================
# 4. COMPUTE TOP-20 DEGs ON TEST SET
# ===============================================================================
print(f"\n[4] Computing top-{N_TOP_DEGS} DEGs for {len(test_perts)} test perturbations ...")

ctrl_mask = adata.obs["is_control"].values.astype(bool)
control_X = np.asarray(adata.X[ctrl_mask].todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X[ctrl_mask])
control_mean = control_X.mean(axis=0)

deg_genes_per_condition = {}
all_deg_genes = set()

for pert in test_perts:
    pert_mask = (adata.obs[PERT_COL] == pert).values
    pert_X = np.asarray(adata.X[pert_mask].todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X[pert_mask])
    pert_mean = pert_X.mean(axis=0)
    fc = pert_mean - control_mean
    top_idx = np.argsort(np.abs(fc))[-N_TOP_DEGS:]
    deg_genes = adata.var_names[top_idx].tolist()
    deg_genes_per_condition[pert] = deg_genes
    all_deg_genes.update(deg_genes)

all_deg_genes = sorted(all_deg_genes)
n_deg = len(all_deg_genes)
print(f"  Unique DEGs across test conditions: {n_deg}")

# ===============================================================================
# 5. PREPARE CELLFLOW INPUTS  (GenePT 1536-dim gene embeddings)
# ===============================================================================
print("\n[5] Preparing CellFlow inputs (GenePT embeddings) ...")

GENEPT_PATH = os.path.expanduser("~/Desktop/SP-FM/data/GenePT_gene_embedding.pickle")
with open(GENEPT_PATH, "rb") as fh:
    genept_dict = pickle.load(fh)
print(f"  Loaded GenePT embeddings: {len(genept_dict)} genes, dim={len(next(iter(genept_dict.values())))}")

all_train_cats = adata_train.obs[PERT_COL].cat.categories.tolist()
pert_emb = {}
for p in all_train_cats:
    if p in genept_dict:
        pert_emb[p] = np.array(genept_dict[p], dtype=np.float32)
    else:
        print(f"  WARNING: train pert '{p}' not in GenePT dict, using zeros")
        pert_emb[p] = np.zeros(1536, dtype=np.float32)

for p in test_perts:
    if p not in pert_emb:
        if p in genept_dict:
            pert_emb[p] = np.array(genept_dict[p], dtype=np.float32)
        else:
            print(f"  WARNING: test pert '{p}' not in GenePT dict, using zeros")
            pert_emb[p] = np.zeros(1536, dtype=np.float32)

# Control gets a zero vector of same dim
pert_emb[CONTROL_VALUE] = np.zeros(1536, dtype=np.float32)

adata_train.uns["pert_emb"] = pert_emb
print(f"  Built pert_emb dict: {len(pert_emb)} entries, dim={pert_emb[all_train_cats[0]].shape}")
adata_train.obs[PERT_COL] = adata_train.obs[PERT_COL].astype("category")
adata_train.obs["is_control"] = adata_train.obs["is_control"].astype(bool)

pca_loadings = adata.varm["PCs"]
pca_mean = np.asarray(adata.X.mean(axis=0)).ravel() if hasattr(adata.X, "toarray") else adata.X.mean(axis=0)

all_cats_for_pred = all_train_cats + [p for p in test_perts if p not in all_train_cats]
cov_rows = [{"perturbation_name": p, "is_control": False} for p in test_perts]
cov_df = pd.DataFrame(cov_rows)
cov_df[PERT_COL] = pd.Categorical(cov_df[PERT_COL], categories=all_cats_for_pred)

real_test_mask = (~adata.obs["is_control"]) & (adata.obs[PERT_COL].isin(test_perts))
real_test_adata = adata[real_test_mask.values].copy()
real_test_adata.X = np.asarray(real_test_adata.X.todense()).astype(np.float32) if hasattr(real_test_adata.X, "todense") else np.asarray(real_test_adata.X).astype(np.float32)
print(f"  Train categories: {len(all_train_cats)} ({len(train_perts)} perts + control)")
print(f"  Test perturbations: {len(test_perts)}")
print(f"  Real test cells: {real_test_adata.shape}")

# ===============================================================================
# 6. TRAINING & PREDICTION
# ===============================================================================
import cellflow
import optax


from cellflow.training._callbacks import WandbLogger


def train_and_predict(model_name, source_type, gmm_kwargs=None):
    print(f"\n{'=' * 70}")
    print(f"  TRAINING: {model_name}")
    print(f"  (training on {len(train_perts)} perts + control, predicting {len(test_perts)} held-out)")
    print(f"{'=' * 70}")

    cf = cellflow.model.CellFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"pert": [PERT_COL]},
        perturbation_covariate_reps={"pert": "pert_emb"},
        null_value=0.0,
    )

    # Project 1536-dim GenePT embeddings before and after pooling
    layers_before_pool = {
        "pert": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
    }
    layers_after_pool = {
        "layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0,
    }

    model_kwargs = dict(
        condition_embedding_dim=256,
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        hidden_dims=(512, 512, 512),
        decoder_dims=(512, 512, 512),
        time_encoder_dims=(512, 512, 512),
        conditioning="film",
        cond_output_dropout=0.1,
        pooling="attention_token",
        optimizer=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR)),
    )
    if source_type == "gmm":
        model_kwargs["source_type"] = "gmm"
        model_kwargs["gmm_kwargs"] = gmm_kwargs
        model_kwargs["gmm_optimizer"] = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(GMM_LR))
        model_kwargs["geodesic_loss_weight"] = GEODESIC_WEIGHT
        model_kwargs["solver_kwargs"] = {
            "gmm_warmup_iters": GMM_WARMUP,
            "gmm_cooldown_iters": GMM_COOLDOWN,
            "source_matching_weight": SOURCE_MATCH_WEIGHT,
        }

    cf.prepare_model(**model_kwargs)

    # W&B callback — logs loss & gradient norms every iteration
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        out_dir=OUT_DIR,
        config={
            "model_name": model_name,
            "source_type": source_type,
            "num_iters": NUM_ITERS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "n_train_perts": len(train_perts),
            "n_dev_perts": len(dev_perts),
            "n_test_perts": len(test_perts),
            "condition_embedding_dim": 256,
            "hidden_dims": "512x512x512",
            "conditioning": "film",
            "gmm_warmup": GMM_WARMUP if source_type == "gmm" else 0,
            "gmm_cooldown": GMM_COOLDOWN if source_type == "gmm" else 0,
            "source_matching_weight": SOURCE_MATCH_WEIGHT if source_type == "gmm" else 0,
            "geodesic_weight": GEODESIC_WEIGHT if source_type == "gmm" else 0,
            "gmm_lr": GMM_LR if source_type == "gmm" else 0,
            **(gmm_kwargs or {}),
        },
        name=f"{source_type}_{NUM_ITERS}it_film512_v9_s{SPLIT}",
    )

    t0 = time.time()
    cf.train(
        num_iterations=NUM_ITERS,
        batch_size=BATCH_SIZE,
        callbacks=[wandb_logger],
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Finish this W&B run
    try:
        wandb_logger.wandb.finish()
    except Exception:
        pass

    t0 = time.time()
    if source_type == "gmm":
        preds = cf.predict_from_gmm(cov_df, n_samples=N_SAMPLES)
    else:
        ctrl_idx = np.where(adata_train.obs["is_control"].values.astype(bool))[0]
        rng_np = np.random.default_rng(SEED)
        sub_idx = rng_np.choice(ctrl_idx, size=N_SAMPLES, replace=False)
        ctrl_adata = adata_train[sub_idx].copy()
        ctrl_adata.obs["is_control"] = True
        preds = cf.predict(ctrl_adata, cov_df)
    pred_time = time.time() - t0
    print(f"  Prediction time: {pred_time:.1f}s")

    gen_arrays, gen_perts_list = [], []
    for key, arr in preds.items():
        gen_arrays.append(np.asarray(arr))
        pname = key[0] if isinstance(key, tuple) else key
        gen_perts_list.extend([pname] * arr.shape[0])

    gen_X_pca = np.vstack(gen_arrays)
    gen_X = gen_X_pca @ pca_loadings.T + pca_mean

    gen_obs = pd.DataFrame({PERT_COL: pd.Categorical(gen_perts_list, categories=all_cats_for_pred)})
    gen_adata = ad.AnnData(X=gen_X.astype(np.float32), obs=gen_obs)
    gen_adata.var_names = adata.var_names.copy()
    gen_adata.obsm["X_pca"] = gen_X_pca.astype(np.float32)

    return gen_adata, train_time, pred_time


gen_vanilla, t_train_v, t_pred_v = train_and_predict(
    "Vanilla CellFlow (Gaussian source)", source_type="gaussian",
)

gen_gmm, t_train_g, t_pred_g = train_and_predict(
    "GMM CellFlow (Learned GMM source)", source_type="gmm",
    gmm_kwargs={
        "num_modes": 16,
        "variance": 0.2,
        "temperature": 0.5,
        "hard_gumbel": True,
        "means_hidden_dims": (512, 256),
        "logits_hidden_dims": (256, 128),
    },
)

# ===============================================================================
# 7. EVALUATE ON TOP DEGs (TEST SET ONLY)
# ===============================================================================
print(f"\n{'=' * 70}")
print(f"  EVALUATION - {len(test_perts)} held-out perturbations, top-{N_TOP_DEGS} DEGs ({n_deg} genes)")
print(f"{'=' * 70}")

import gge
os.makedirs(OUT_DIR, exist_ok=True)

deg_mask = real_test_adata.var_names.isin(all_deg_genes)
real_eval = real_test_adata[:, deg_mask].copy()

GGE_METRICS = ["mse", "pearson", "spearman", "r_squared",
               "wasserstein_1", "wasserstein_2", "mmd", "energy"]

results_dict = {}
for name, gen_adata in [("Vanilla", gen_vanilla), ("GMM", gen_gmm)]:
    print(f"\n  Evaluating {name} on DEGs ...")
    gen_eval = gen_adata[:, deg_mask].copy()
    print(f"    real={real_eval.shape}, gen={gen_eval.shape}")

    t0 = time.time()
    results = gge.evaluate(
        real_data=real_eval,
        generated_data=gen_eval,
        condition_columns=[PERT_COL],
        metrics=GGE_METRICS,
        include_multivariate=True,
        verbose=True,
    )
    eval_time = time.time() - t0
    print(f"    Time: {eval_time:.1f}s")
    results_dict[name] = results

# ===============================================================================
# 8. DEG CORRELATION (top-20 LFC per condition)
# ===============================================================================
print(f"\n{'=' * 70}")
print(f"  DEG CORRELATION (top-{N_TOP_DEGS} per condition, test set)")
print(f"{'=' * 70}")

deg_corr_results = {}
for name, gen_adata in [("Vanilla", gen_vanilla), ("GMM", gen_gmm)]:
    corrs = []
    for pert in test_perts:
        real_cells = real_test_adata[real_test_adata.obs[PERT_COL] == pert]
        real_mean = np.asarray(real_cells.X).mean(axis=0)
        gen_cells = gen_adata[gen_adata.obs[PERT_COL] == pert]
        gen_mean = np.asarray(gen_cells.X).mean(axis=0)

        fc_real = real_mean - control_mean
        fc_gen = gen_mean - control_mean

        top_idx = np.argsort(np.abs(fc_real))[-N_TOP_DEGS:]

        if np.std(fc_real[top_idx]) < 1e-10 or np.std(fc_gen[top_idx]) < 1e-10:
            r = 0.0
        else:
            r, _ = pearsonr(fc_real[top_idx], fc_gen[top_idx])
        corrs.append(r)

    deg_corr_results[name] = corrs
    mean_r = np.mean(corrs)
    std_r = np.std(corrs)
    print(f"  {name:>10s}: DEG corr = {mean_r:.4f} +/- {std_r:.4f}  (over {len(test_perts)} test perts)")

# ===============================================================================
# 9. SUMMARY TABLE
# ===============================================================================
print(f"\n{'=' * 70}")
print("  COMPARISON SUMMARY (test set only)")
print(f"{'=' * 70}")

higher_better = {"pearson", "spearman", "r_squared"}

all_metrics = GGE_METRICS + ["multivariate_wasserstein", "multivariate_mmd"]
header = f"  {'Metric':<30s} | {'Vanilla':>20s} | {'GMM':>20s} | {'Better':>8s}"
print(f"\n  All metrics on top-{N_TOP_DEGS} DEGs ({n_deg} genes), {len(test_perts)} held-out perts")
print(header)
print("  " + "-" * (len(header) - 2))

summary_rows = []
v_agg = results_dict["Vanilla"].summary()["splits"]["all"]["aggregates"]
g_agg = results_dict["GMM"].summary()["splits"]["all"]["aggregates"]

for metric in all_metrics:
    v_mean = v_agg.get(f"{metric}_mean", float("nan"))
    v_std = v_agg.get(f"{metric}_std", float("nan"))
    g_mean = g_agg.get(f"{metric}_mean", float("nan"))
    g_std = g_agg.get(f"{metric}_std", float("nan"))

    if metric in higher_better:
        better = "GMM" if g_mean > v_mean else "Vanilla"
    else:
        better = "GMM" if g_mean < v_mean else "Vanilla"

    v_str = f"{v_mean:.4f} +/- {v_std:.4f}"
    g_str = f"{g_mean:.4f} +/- {g_std:.4f}"
    print(f"  {metric:<30s} | {v_str:>20s} | {g_str:>20s} | {better:>8s}")
    summary_rows.append({"metric": metric, "scope": "DEG", "vanilla_mean": v_mean,
                          "vanilla_std": v_std, "gmm_mean": g_mean, "gmm_std": g_std,
                          "better": better})

v_deg = np.mean(deg_corr_results["Vanilla"])
v_deg_std = np.std(deg_corr_results["Vanilla"])
g_deg = np.mean(deg_corr_results["GMM"])
g_deg_std = np.std(deg_corr_results["GMM"])
better_deg = "GMM" if g_deg > v_deg else "Vanilla"
print(f"  {'deg_corr (top20 LFC)':<30s} | {v_deg:>8.4f} +/- {v_deg_std:.4f}  | {g_deg:>8.4f} +/- {g_deg_std:.4f}  | {better_deg:>8s}")
summary_rows.append({"metric": "deg_correlation_top20", "scope": "DEG",
                      "vanilla_mean": v_deg, "vanilla_std": v_deg_std,
                      "gmm_mean": g_deg, "gmm_std": g_deg_std, "better": better_deg})

print(f"\n  {'Timing':<30s} | {'Vanilla':>20s} | {'GMM':>20s}")
print(f"  {'  Train (s)':<30s} | {t_train_v:>20.1f} | {t_train_g:>20.1f}")
print(f"  {'  Predict (s)':<30s} | {t_pred_v:>20.1f} | {t_pred_g:>20.1f}")

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, "comparison_summary.csv"), index=False)

split_info = {
    "n_train_perts": len(train_perts), "n_dev_perts": len(dev_perts),
    "n_test_perts": len(test_perts),
    "train_perts": train_perts, "dev_perts": dev_perts, "test_perts": test_perts,
    "n_deg_genes": n_deg, "n_top_degs_per_condition": N_TOP_DEGS,
    "num_iters": NUM_ITERS, "seed": SEED, "split": SPLIT,
}
with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)
print(f"\n  Saved to {OUT_DIR}/comparison_summary.csv and split_info.json")

# ===============================================================================
# 10. PLOTS
# ===============================================================================
print(f"\n{'=' * 70}")
print("  GENERATING PLOTS")
print(f"{'=' * 70}")

# 10a. Bar chart
fig, axes = plt.subplots(2, 5, figsize=(28, 10))
bar_metrics = all_metrics
bar_labels = [
    "MSE (lower)", "Pearson (higher)", "Spearman (higher)", "R2 (higher)",
    "W-1 (lower)", "W-2 (lower)", "MMD (lower)", "Energy (lower)",
    "MV-Wass (lower)", "MV-MMD (lower)",
]
for ax, metric, label in zip(axes.ravel(), bar_metrics, bar_labels):
    vals = [v_agg[f"{metric}_mean"], g_agg[f"{metric}_mean"]]
    errs = [v_agg[f"{metric}_std"], g_agg[f"{metric}_std"]]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(["Vanilla", "GMM"], vals, yerr=errs, color=colors, capsize=5,
                  edgecolor="black", linewidth=0.5)
    ax.set_title(label, fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

plt.suptitle(
    f"Vanilla vs GMM CellFlow - Norman 2019 - top-{N_TOP_DEGS} DEGs, "
    f"{len(test_perts)} held-out perts", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "metric_comparison_bars.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 10b. DEG correlation per test condition
fig, ax = plt.subplots(figsize=(max(12, len(test_perts) * 0.45), 6))
x = np.arange(len(test_perts))
w = 0.35
ax.bar(x - w / 2, deg_corr_results["Vanilla"], w, label="Vanilla", color="#4C72B0",
       edgecolor="black", linewidth=0.5)
ax.bar(x + w / 2, deg_corr_results["GMM"], w, label="GMM", color="#DD8452",
       edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(test_perts, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("DEG Correlation (Pearson)")
ax.set_title(f"DEG Correlation (top-{N_TOP_DEGS} LFC) - Held-out Test Perturbations", fontweight="bold")
ax.legend()
ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "deg_correlation_per_condition.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 10c. Per-condition scatter (capped at 16)
n_scatter = min(16, len(test_perts))
ncols = 4
nrows = (n_scatter + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 7, nrows * 3.5))
axes = np.atleast_2d(axes)

for i, pert in enumerate(test_perts[:n_scatter]):
    row = i // ncols
    col_base = (i % ncols) * 2

    real_cells = real_test_adata[real_test_adata.obs[PERT_COL] == pert]
    real_mean = np.asarray(real_cells.X).mean(axis=0)

    for j, (name, gen_adata, color) in enumerate([
        ("Vanilla", gen_vanilla, "#4C72B0"), ("GMM", gen_gmm, "#DD8452")
    ]):
        ax = axes[row, col_base + j]
        gen_cells = gen_adata[gen_adata.obs[PERT_COL] == pert]
        gen_mean = np.asarray(gen_cells.X).mean(axis=0)
        r, _ = pearsonr(real_mean, gen_mean)
        ax.scatter(real_mean, gen_mean, alpha=0.3, s=3, c=color)
        mn = min(real_mean.min(), gen_mean.min())
        mx = max(real_mean.max(), gen_mean.max())
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, linewidth=0.5)
        ax.set_title(f"{name} | {pert} | r={r:.3f}", fontsize=8)
        ax.tick_params(labelsize=6)

for i in range(n_scatter, nrows * ncols):
    row = i // ncols
    col_base = (i % ncols) * 2
    axes[row, col_base].set_visible(False)
    axes[row, col_base + 1].set_visible(False)

plt.suptitle("Real vs Generated (mean expr) - Test Perturbations", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scatter_per_condition.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 10d. UMAP
print("  Computing UMAP ...")
n_umap_perts = min(15, len(test_perts))
umap_perts = test_perts[:n_umap_perts]

real_umap = real_test_adata[real_test_adata.obs[PERT_COL].isin(umap_perts)].copy()
gen_v_umap = gen_vanilla[gen_vanilla.obs[PERT_COL].isin(umap_perts)].copy()
gen_g_umap = gen_gmm[gen_gmm.obs[PERT_COL].isin(umap_perts)].copy()

combined = ad.concat(
    [real_umap, gen_v_umap, gen_g_umap],
    keys=["Real", "Vanilla", "GMM"],
    label="source",
)
sc.pp.pca(combined, n_comps=30)
sc.pp.neighbors(combined, n_pcs=30)
sc.tl.umap(combined)

ncols_u = 5
nrows_u = (n_umap_perts + ncols_u - 1) // ncols_u
fig, axes = plt.subplots(nrows_u, ncols_u, figsize=(ncols_u * 5, nrows_u * 5))
axes_flat = np.atleast_1d(axes).ravel()

for i, pert in enumerate(umap_perts):
    ax = axes_flat[i]
    umap_all = combined.obsm["X_umap"]
    ax.scatter(umap_all[:, 0], umap_all[:, 1], c="lightgrey", s=1, alpha=0.05, rasterized=True)
    for src, color in [("Real", "tab:blue"), ("Vanilla", "tab:green"), ("GMM", "tab:red")]:
        mask = (combined.obs["source"] == src) & (combined.obs[PERT_COL] == pert)
        umap_sub = combined[mask].obsm["X_umap"]
        ax.scatter(umap_sub[:, 0], umap_sub[:, 1], c=color, s=6, alpha=0.6, label=src)
    ax.set_title(pert, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")

for i in range(n_umap_perts, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.suptitle("UMAP: Real (blue) vs Vanilla (green) vs GMM (red) - Test Perts",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "umap_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 10e. Boxplot of per-condition metrics
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
for ax, metric in zip(axes.ravel(), GGE_METRICS):
    box_data, box_colors = [], []
    for name, color in [("Vanilla", "#4C72B0"), ("GMM", "#DD8452")]:
        try:
            df = results_dict[name].to_dataframe()
            vals = df.loc[df["metric"] == metric, "value"].values
            if len(vals) == 0:
                raise ValueError
        except Exception:
            agg = results_dict[name].summary()["splits"]["all"]["aggregates"]
            vals = np.array([agg[f"{metric}_mean"]])
        box_data.append(vals)
        box_colors.append(color)
    bp = ax.boxplot(box_data, labels=["Vanilla", "GMM"], patch_artist=True, widths=0.6,
                    medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(metric, fontweight="bold")

plt.suptitle(f"Per-condition Metric Distributions (top-{N_TOP_DEGS} DEGs, test set)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "boxplot_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done!")
