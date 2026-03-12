"""Compare Vanilla CellFlow vs GMM-CellFlow on ComboSciPlex.

Proper held-out evaluation:
  - Drug combination perturbations on A549 cell line
  - Fixed train/test split (7 held-out combinations from CellFlow manuscript)
  - Models trained on control + train perturbations only
  - ALL metrics computed on top-20 DEGs per test condition
  - DEG correlation (LFC-based) on test set
  - Losses and gradient norms logged to Weights & Biases

Uses molecular fingerprints as drug condition representations.

Usage
-----
    python run_combosciplex_vanilla_vs_gmm.py
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import warnings
warnings.filterwarnings("ignore")

import functools
import time
import json
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
import argparse

parser = argparse.ArgumentParser(description="Vanilla vs GMM CellFlow on ComboSciPlex")
parser.add_argument("--split", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Split index (0-3) for different train/test splits")
args = parser.parse_args()
SPLIT = args.split

# ===============================================================================
# CONFIGURATION
# ===============================================================================
CONDITION_COL = "condition"
DRUG1_COL = "Drug1"
DRUG2_COL = "Drug2"
CONTROL_VALUE = "control"
N_TOP_DEGS = 20
NUM_ITERS = 35000
BATCH_SIZE = 256
N_SAMPLES = 2000
LR = 5e-4
SEED = 42 + SPLIT
GMM_WARMUP = 5000
GMM_COOLDOWN = 3000
GMM_LR = 1e-3
GEODESIC_WEIGHT = 0.005
SOURCE_MATCH_WEIGHT = 2.0
OUT_DIR = f"combosciplex_results/split_{SPLIT}"
WANDB_PROJECT = "cellflow-combosciplex"
N_TOP_GENES = 2000
N_PCA_COMPS = 50

# Test conditions: split 0 = CellFlow manuscript, splits 1-3 = random 7 conditions
_MANUSCRIPT_TEST_CONDITIONS = [
    "Panobinostat+Crizotinib",
    "Panobinostat+Curcumin",
    "Panobinostat+SRT1720",
    "Panobinostat+Sorafenib",
    "SRT2104+Alvespimycin",
    "control+Alvespimycin",
    "control+Dacinostat",
]

_ALL_COMBO_CONDITIONS = [
    "Alvespimycin+Pirarubicin", "Cediranib+PCI-34051", "Dacinostat+Danusertib",
    "Dacinostat+Dasatinib", "Dacinostat+PCI-34051", "Givinostat+Carmofur",
    "Givinostat+Cediranib", "Givinostat+Crizotinib", "Givinostat+Curcumin",
    "Givinostat+Dasatinib", "Givinostat+SRT1720", "Givinostat+SRT2104",
    "Givinostat+Sorafenib", "Givinostat+Tanespimycin", "Panobinostat+Alvespimycin",
    "Panobinostat+Crizotinib", "Panobinostat+Curcumin", "Panobinostat+Dasatinib",
    "Panobinostat+PCI-34051", "Panobinostat+SRT1720", "Panobinostat+SRT2104",
    "Panobinostat+SRT3025", "Panobinostat+Sorafenib", "SRT2104+Alvespimycin",
    "SRT3025+Cediranib", "control+Alvespimycin", "control+Dacinostat",
    "control+Dasatinib", "control+Givinostat", "control+Panobinostat",
    "control+SRT2104",
]

if SPLIT == 0:
    TEST_CONDITIONS = _MANUSCRIPT_TEST_CONDITIONS
else:
    _rng_split = np.random.default_rng(SEED)
    _shuffled = _rng_split.permutation(_ALL_COMBO_CONDITIONS).tolist()
    TEST_CONDITIONS = sorted(_shuffled[:7])

print("=" * 70)
print(f"VANILLA vs GMM CELLFLOW - ComboSciPlex (Split {SPLIT}, seed={SEED})")
print("=" * 70)

# ===============================================================================
# 1. LOAD DATA
# ===============================================================================
print("\n[1] Loading ComboSciPlex ...")
DATA_PATH = "/tmp/combosciplex.h5ad"
if not os.path.exists(DATA_PATH):
    print("  Downloading via pertpy ...")
    import pertpy
    adata_raw = pertpy.data.combosciplex()
    print(f"  Downloaded: {adata_raw.shape}")
else:
    adata_raw = ad.read_h5ad(DATA_PATH)
    print(f"  Loaded from cache: {adata_raw.shape}")

# ===============================================================================
# 2. PREPROCESS
# ===============================================================================
print("\n[2] Preprocessing ...")
adata = adata_raw.copy()

# Fix condition column: control+control -> control
adata.obs[CONDITION_COL] = adata.obs.apply(
    lambda x: CONTROL_VALUE if x[CONDITION_COL] == "control+control" else x[CONDITION_COL],
    axis=1,
)
adata.obs["is_control"] = (adata.obs[CONDITION_COL] == CONTROL_VALUE)

print(f"  Total cells: {adata.shape[0]}")
print(f"  Unique conditions: {adata.obs[CONDITION_COL].nunique()}")
print(f"  Control cells: {adata.obs['is_control'].sum()}")

# Standard preprocessing
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=N_TOP_GENES)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.pca(adata, n_comps=N_PCA_COMPS)
print(f"  After preprocessing: {adata.shape}")

# ===============================================================================
# 3. COMPUTE MOLECULAR FINGERPRINTS
# ===============================================================================
print("\n[3] Computing molecular fingerprints ...")
import pickle as _pkl

FP_CACHE = os.path.join(OUT_DIR, "fingerprints_cache.pkl")
if os.path.exists(FP_CACHE):
    print("  Loading fingerprints from cache ...")
    with open(FP_CACHE, "rb") as _f:
        adata.uns["fingerprints"] = _pkl.load(_f)
else:
    from cellflow.preprocessing import annotate_compounds, get_molecular_fingerprints
    annotate_compounds(adata, compound_keys=[DRUG1_COL, DRUG2_COL])
    get_molecular_fingerprints(adata, compound_keys=[DRUG1_COL, DRUG2_COL])
    # Add zero vector for control
    adata.uns["fingerprints"][CONTROL_VALUE] = np.zeros_like(
        next(iter(adata.uns["fingerprints"].values()))
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(FP_CACHE, "wb") as _f:
        _pkl.dump(adata.uns["fingerprints"], _f)
    print("  Saved fingerprints to cache.")

# Ensure control has zero vector (whether loaded from cache or computed)
if CONTROL_VALUE not in adata.uns["fingerprints"]:
    adata.uns["fingerprints"][CONTROL_VALUE] = np.zeros_like(
        next(iter(adata.uns["fingerprints"].values()))
    )
fp_dim = len(next(iter(adata.uns["fingerprints"].values())))
print(f"  Fingerprint dimension: {fp_dim}")
print(f"  Fingerprint entries: {len(adata.uns['fingerprints'])}")
print(f"  Fingerprint keys: {sorted(adata.uns['fingerprints'].keys())}")

# ===============================================================================
# 4. TRAIN/TEST SPLIT
# ===============================================================================
print("\n[4] Train/Test split ...")
adata.obs["mode"] = adata.obs.apply(
    lambda x: "test" if x[CONDITION_COL] in TEST_CONDITIONS else "train", axis=1
)

n_train_conds = adata.obs.loc[adata.obs["mode"] == "train", CONDITION_COL].nunique()
n_test_conds = len(TEST_CONDITIONS)
print(f"  Train conditions: {n_train_conds}")
print(f"  Test conditions:  {n_test_conds}  {TEST_CONDITIONS}")

adata_train = adata[(adata.obs["mode"] == "train")].copy()
adata_test = adata[(adata.obs["mode"] == "test") | adata.obs["is_control"]].copy()

adata_train.obs_names_make_unique()
adata_test.obs_names_make_unique()

adata_train.obs[CONDITION_COL] = adata_train.obs[CONDITION_COL].astype("category").cat.remove_unused_categories()
adata_train.obs["is_control"] = adata_train.obs["is_control"].astype(bool)

print(f"  Train adata: {adata_train.shape}")
print(f"  Test adata:  {adata_test.shape}")
print(f"  Train conditions: {sorted(adata_train.obs[CONDITION_COL].unique().tolist())}")

# ===============================================================================
# 5. COMPUTE TOP-20 DEGs ON TEST SET
# ===============================================================================
print(f"\n[5] Computing top-{N_TOP_DEGS} DEGs for {n_test_conds} test conditions ...")

ctrl_mask = adata.obs["is_control"].values.astype(bool)
control_X = np.asarray(adata.X[ctrl_mask].todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X[ctrl_mask])
control_mean = control_X.mean(axis=0)

deg_genes_per_condition = {}
all_deg_genes = set()

for cond in TEST_CONDITIONS:
    cond_mask = (adata.obs[CONDITION_COL] == cond).values
    cond_X = np.asarray(adata.X[cond_mask].todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X[cond_mask])
    cond_mean = cond_X.mean(axis=0)
    fc = cond_mean - control_mean
    top_idx = np.argsort(np.abs(fc))[-N_TOP_DEGS:]
    deg_genes = adata.var_names[top_idx].tolist()
    deg_genes_per_condition[cond] = deg_genes
    all_deg_genes.update(deg_genes)

all_deg_genes = sorted(all_deg_genes)
n_deg = len(all_deg_genes)
print(f"  Unique DEGs across test conditions: {n_deg}")

# ===============================================================================
# 6. PREPARE FOR PREDICTION
# ===============================================================================
print("\n[6] Preparing prediction data ...")

pca_loadings = adata.varm["PCs"]
pca_mean = np.asarray(adata.X.mean(axis=0)).ravel() if hasattr(adata.X, "toarray") else adata.X.mean(axis=0)

# Build covariate_data for prediction (one row per test condition)
cov_rows = []
for cond in TEST_CONDITIONS:
    sample_row = adata.obs[adata.obs[CONDITION_COL] == cond].iloc[0]
    cov_rows.append({
        CONDITION_COL: cond,
        DRUG1_COL: sample_row[DRUG1_COL],
        DRUG2_COL: sample_row[DRUG2_COL],
        "is_control": False,
    })
cov_df = pd.DataFrame(cov_rows)

# Real test data (non-control test cells only)
real_test_mask = (~adata.obs["is_control"]) & (adata.obs[CONDITION_COL].isin(TEST_CONDITIONS))
real_test_adata = adata[real_test_mask.values].copy()
real_test_adata.X = np.asarray(real_test_adata.X.todense()).astype(np.float32) if hasattr(real_test_adata.X, "todense") else np.asarray(real_test_adata.X).astype(np.float32)
print(f"  Real test cells: {real_test_adata.shape}")
print(f"  Covariate data: {len(cov_df)} rows")

# ===============================================================================
# 7. TRAINING & PREDICTION
# ===============================================================================
import cellflow
import optax
from cellflow.utils import match_linear
from cellflow.training._callbacks import WandbLogger


def train_and_predict(model_name, source_type, gmm_kwargs=None):
    print(f"\n{'=' * 70}")
    print(f"  TRAINING: {model_name}")
    print(f"  (training on {n_train_conds} conditions + control, predicting {n_test_conds} held-out)")
    print(f"{'=' * 70}")

    cf = cellflow.model.CellFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"drug_perturbation": (DRUG1_COL, DRUG2_COL)},
        perturbation_covariate_reps={"drug_perturbation": "fingerprints"},
        max_combination_length=2,
        null_value=0.0,
    )

    # Project fingerprints before and after pooling
    layers_before_pool = {
        "drug_perturbation": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
    }
    layers_after_pool = {
        "layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0,
    }

    match_fn = functools.partial(match_linear, epsilon=1.0, tau_a=1.0, tau_b=1.0)

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
        probability_path={"constant_noise": 1.5},
        match_fn=match_fn,
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

    # W&B callback
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
            "n_train_conds": n_train_conds,
            "n_test_conds": n_test_conds,
            "condition_embedding_dim": 256,
            "hidden_dims": "512x512x512",
            "conditioning": "film",
            "fingerprint_dim": fp_dim,
            "probability_path_noise": 1.5,
            "gmm_warmup": GMM_WARMUP if source_type == "gmm" else 0,
            "gmm_cooldown": GMM_COOLDOWN if source_type == "gmm" else 0,
            "source_matching_weight": SOURCE_MATCH_WEIGHT if source_type == "gmm" else 0,
            "geodesic_weight": GEODESIC_WEIGHT if source_type == "gmm" else 0,
            "gmm_lr": GMM_LR if source_type == "gmm" else 0,
            **(gmm_kwargs or {}),
        },
        name=f"{source_type}_{NUM_ITERS}it_combo_v8_s{SPLIT}",
    )

    t0 = time.time()
    cf.train(
        num_iterations=NUM_ITERS,
        batch_size=BATCH_SIZE,
        callbacks=[wandb_logger],
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

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

    gen_arrays, gen_conds_list = [], []
    for key, arr in preds.items():
        gen_arrays.append(np.asarray(arr))
        # Key is a tuple of covariates, e.g. ("Panobinostat", "Crizotinib")
        # Join with "+" to match the condition column format
        cname = "+".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        gen_conds_list.extend([cname] * arr.shape[0])
    print(f"  Generated conditions: {sorted(set(gen_conds_list))}")

    gen_X_pca = np.vstack(gen_arrays)
    gen_X = gen_X_pca @ pca_loadings.T + pca_mean

    gen_obs = pd.DataFrame({CONDITION_COL: gen_conds_list})
    gen_adata = ad.AnnData(X=gen_X.astype(np.float32), obs=gen_obs)
    gen_adata.var_names = adata.var_names.copy()
    gen_adata.obsm["X_pca"] = gen_X_pca.astype(np.float32)

    return gen_adata, train_time, pred_time


gen_vanilla, t_train_v, t_pred_v = train_and_predict(
    "Vanilla CellFlow (Control source)", source_type="gaussian",
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
# 8. EVALUATE ON TOP DEGs (TEST SET ONLY)
# ===============================================================================
print(f"\n{'=' * 70}")
print(f"  EVALUATION - {n_test_conds} held-out conditions, top-{N_TOP_DEGS} DEGs ({n_deg} genes)")
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
        condition_columns=[CONDITION_COL],
        metrics=GGE_METRICS,
        include_multivariate=True,
        verbose=True,
    )
    eval_time = time.time() - t0
    print(f"    Time: {eval_time:.1f}s")
    results_dict[name] = results

# ===============================================================================
# 9. DEG CORRELATION (top-20 LFC per condition)
# ===============================================================================
print(f"\n{'=' * 70}")
print(f"  DEG CORRELATION (top-{N_TOP_DEGS} per condition, test set)")
print(f"{'=' * 70}")

deg_corr_results = {}
for name, gen_adata in [("Vanilla", gen_vanilla), ("GMM", gen_gmm)]:
    corrs = []
    for cond in TEST_CONDITIONS:
        real_cells = real_test_adata[real_test_adata.obs[CONDITION_COL] == cond]
        real_mean = np.asarray(real_cells.X).mean(axis=0)
        gen_cells = gen_adata[gen_adata.obs[CONDITION_COL] == cond]
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
    print(f"  {name:>10s}: DEG corr = {mean_r:.4f} +/- {std_r:.4f}  (over {n_test_conds} test conds)")

# ===============================================================================
# 10. SUMMARY TABLE
# ===============================================================================
print(f"\n{'=' * 70}")
print("  COMPARISON SUMMARY (test set only)")
print(f"{'=' * 70}")

higher_better = {"pearson", "spearman", "r_squared"}

all_metrics = GGE_METRICS + ["multivariate_wasserstein", "multivariate_mmd"]
header = f"  {'Metric':<30s} | {'Vanilla':>20s} | {'GMM':>20s} | {'Better':>8s}"
print(f"\n  All metrics on top-{N_TOP_DEGS} DEGs ({n_deg} genes), {n_test_conds} held-out conditions")
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
    "n_train_conds": n_train_conds, "n_test_conds": n_test_conds,
    "test_conditions": TEST_CONDITIONS,
    "n_deg_genes": n_deg, "n_top_degs_per_condition": N_TOP_DEGS,
    "num_iters": NUM_ITERS, "seed": SEED, "split": SPLIT,
    "fingerprint_dim": fp_dim,
}
with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)
print(f"\n  Saved to {OUT_DIR}/comparison_summary.csv and split_info.json")

# ===============================================================================
# 11. PLOTS
# ===============================================================================
print(f"\n{'=' * 70}")
print("  GENERATING PLOTS")
print(f"{'=' * 70}")

# 11a. Bar chart
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
    f"Vanilla vs GMM CellFlow - ComboSciPlex - top-{N_TOP_DEGS} DEGs, "
    f"{n_test_conds} held-out conditions", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "metric_comparison_bars.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 11b. DEG correlation per test condition
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(n_test_conds)
w = 0.35
ax.bar(x - w / 2, deg_corr_results["Vanilla"], w, label="Vanilla", color="#4C72B0",
       edgecolor="black", linewidth=0.5)
ax.bar(x + w / 2, deg_corr_results["GMM"], w, label="GMM", color="#DD8452",
       edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(TEST_CONDITIONS, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("DEG Correlation (Pearson)")
ax.set_title(f"DEG Correlation (top-{N_TOP_DEGS} LFC) - Held-out Test Conditions", fontweight="bold")
ax.legend()
ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "deg_correlation_per_condition.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 11c. Per-condition scatter
ncols = 4
nrows = (n_test_conds + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 7, nrows * 3.5))
axes = np.atleast_2d(axes)

for i, cond in enumerate(TEST_CONDITIONS):
    row = i // ncols
    col_base = (i % ncols) * 2

    real_cells = real_test_adata[real_test_adata.obs[CONDITION_COL] == cond]
    real_mean = np.asarray(real_cells.X).mean(axis=0)

    for j, (name, gen_adata, color) in enumerate([
        ("Vanilla", gen_vanilla, "#4C72B0"), ("GMM", gen_gmm, "#DD8452")
    ]):
        ax = axes[row, col_base + j]
        gen_cells = gen_adata[gen_adata.obs[CONDITION_COL] == cond]
        gen_mean = np.asarray(gen_cells.X).mean(axis=0)
        r, _ = pearsonr(real_mean, gen_mean)
        ax.scatter(real_mean, gen_mean, alpha=0.3, s=3, c=color)
        mn = min(real_mean.min(), gen_mean.min())
        mx = max(real_mean.max(), gen_mean.max())
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, linewidth=0.5)
        ax.set_title(f"{name} | {cond} | r={r:.3f}", fontsize=8)
        ax.tick_params(labelsize=6)

for i in range(n_test_conds, nrows * ncols):
    row = i // ncols
    col_base = (i % ncols) * 2
    if row < axes.shape[0] and col_base + 1 < axes.shape[1]:
        axes[row, col_base].set_visible(False)
        axes[row, col_base + 1].set_visible(False)

plt.suptitle("Real vs Generated (mean expr) - Test Conditions", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scatter_per_condition.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 11d. UMAP
print("  Computing UMAP ...")
real_umap = real_test_adata.copy()
gen_v_umap = gen_vanilla[gen_vanilla.obs[CONDITION_COL].isin(TEST_CONDITIONS)].copy()
gen_g_umap = gen_gmm[gen_gmm.obs[CONDITION_COL].isin(TEST_CONDITIONS)].copy()

combined = ad.concat(
    [real_umap, gen_v_umap, gen_g_umap],
    keys=["Real", "Vanilla", "GMM"],
    label="source",
)
sc.pp.pca(combined, n_comps=30)
sc.pp.neighbors(combined, n_pcs=30)
sc.tl.umap(combined)

ncols_u = 4
nrows_u = (n_test_conds + ncols_u - 1) // ncols_u
fig, axes = plt.subplots(nrows_u, ncols_u, figsize=(ncols_u * 5, nrows_u * 5))
axes_flat = np.atleast_1d(axes).ravel()

for i, cond in enumerate(TEST_CONDITIONS):
    ax = axes_flat[i]
    umap_all = combined.obsm["X_umap"]
    ax.scatter(umap_all[:, 0], umap_all[:, 1], c="lightgrey", s=1, alpha=0.05, rasterized=True)
    for src, color in [("Real", "tab:blue"), ("Vanilla", "tab:green"), ("GMM", "tab:red")]:
        mask = (combined.obs["source"] == src) & (combined.obs[CONDITION_COL] == cond)
        umap_sub = combined[mask].obsm["X_umap"]
        ax.scatter(umap_sub[:, 0], umap_sub[:, 1], c=color, s=6, alpha=0.6, label=src)
    ax.set_title(cond, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")

for i in range(n_test_conds, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.suptitle("UMAP: Real (blue) vs Vanilla (green) vs GMM (red) - Test Conditions",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "umap_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 11e. Boxplot of per-condition metrics
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
