# GMM Base Distribution (SP-FM) — Tutorial

This tutorial shows how to use the **GMM base distribution** feature in CellFlow,
which implements the core idea from the SP-FM paper (Rubbi et al., 2025): instead of
transporting observed control cells through the learned flow, we learn a
**condition-dependent Gaussian mixture model** as the source distribution.

## Why a learned base distribution?

In standard CellFlow, the source distribution is the observed control population.
This works well when the control cells capture the diversity of unperturbed states,
but can limit out-of-distribution (OOD) generalization — for unseen perturbations
we still rely on the same control cells.

SP-FM replaces the fixed source with a *learnable*, *condition-dependent* GMM:

- **Component means** are predicted from the condition embedding via a small MLP
  (`H_θ` in the paper).
- **Mixture weights** are predicted from the condition embedding via another MLP
  (`H_p` in the paper), using Gumbel-Softmax for differentiable sampling.
- A **geodesic length loss** ($\|x_1 - x_0\|^2$) regularizes the learned base to be
  close to the target, encouraging shorter transport paths.

Training alternates between updating the velocity field (FM loss) and updating the
GMM parameters (FM loss w/ frozen VF + geodesic loss).

---

## Quick start

```python
import cellflow
import numpy as np
import pandas as pd

# ── 1. Load data ──────────────────────────────────────────────────────────────
# Assume `adata` is an AnnData with:
#   - .obsm["X_pca"]: PCA representation
#   - .obs["control"]: boolean control indicator
#   - .obs["drug1"]: categorical drug column
#   - .uns["drug"]: dict mapping drug names → embedding arrays

cf = cellflow.model.CellFlow(adata, solver="otfm")

# ── 2. Prepare data (same as standard CellFlow) ──────────────────────────────
cf.prepare_data(
    sample_rep="X_pca",
    control_key="control",
    perturbation_covariates={"drug": ["drug1"]},
    perturbation_covariate_reps={"drug": "drug"},
)

# ── 3. Prepare model with GMM source ─────────────────────────────────────────
cf.prepare_model(
    # ── CellFlow architecture (can be tuned as usual) ──
    condition_embedding_dim=32,
    hidden_dims=(256, 256),
    decoder_dims=(256, 256),
    # ── SP-FM: GMM base distribution ──
    source_type="gmm",                     # ← key switch
    gmm_kwargs={
        "num_modes": 16,                   # number of Gaussian components
        "variance": 0.01,                  # fixed isotropic variance per component
        "temperature": 0.5,                # Gumbel-Softmax temperature
        "hard_gumbel": False,              # soft relaxation (True → straight-through)
        "means_hidden_dims": (128,),       # MLP for predicting means
        "logits_hidden_dims": (128,),      # MLP for predicting mixture weights
    },
    geodesic_loss_weight=1.0,              # weight of ||x1-x0||² regularization
    # gmm_optimizer=optax.adam(1e-3),      # optional: custom optimizer for GMM
)

# ── 4. Train ──────────────────────────────────────────────────────────────────
cf.train(num_iterations=5000, batch_size=256)

# ── 5. Predict ────────────────────────────────────────────────────────────────
# Option A: predict_from_gmm — pure generative (no control cells needed)
cov_df = pd.DataFrame({
    "drug1": pd.Categorical(["drug_a"]),
    "control": [False],
})
preds = cf.predict_from_gmm(cov_df, n_samples=500)
# preds is a dict:  {condition_key: np.ndarray of shape (500, data_dim)}

# Option B: standard predict — GMM samples replace control cells automatically
adata_pred = adata.copy()
adata_pred.obs["control"] = True
pred = cf.predict(adata_pred, sample_rep="X_pca", covariate_data=adata_pred.obs)
# pred is a dict:  {condition_key: np.ndarray of shape (n_control, data_dim)}
```

---

## API reference

### `CellFlow.prepare_model(..., source_type="gmm", gmm_kwargs=..., ...)`

New parameters on `prepare_model`:

| Parameter               | Type                | Default         | Description |
|-------------------------|---------------------|-----------------|-------------|
| `source_type`           | `"control" \| "gmm"` | `"control"`   | Source distribution type. `"gmm"` enables SP-FM. |
| `gmm_kwargs`            | `dict \| None`      | `None`          | Keyword arguments for `GMMBaseDist` (see below). |
| `gmm_optimizer`         | `optax.GradientTransformation \| None` | `None` | Optimizer for GMM parameters. Defaults to `optax.adam(1e-3)`. |
| `geodesic_loss_weight`  | `float`             | `1.0`           | Weight of the geodesic length loss $\|x_1 - x_0\|^2$. |

### `gmm_kwargs` options

| Key                | Type            | Default   | Description |
|--------------------|-----------------|-----------|-------------|
| `num_modes`        | `int`           | `16`      | Number of Gaussian mixture components. |
| `variance`         | `float`         | `0.01`    | Fixed isotropic variance per component. |
| `temperature`      | `float`         | `0.5`     | Gumbel-Softmax temperature (lower = sharper). |
| `hard_gumbel`      | `bool`          | `False`   | Straight-through Gumbel-Softmax. |
| `means_hidden_dims`| `tuple[int,...]`| `(128,)`  | Hidden layers for the means predictor MLP. |
| `logits_hidden_dims`| `tuple[int,...]`| `(128,)` | Hidden layers for the logits predictor MLP. |
| `means_dropout`    | `float`         | `0.0`     | Dropout for means MLP. |
| `logits_dropout`   | `float`         | `0.0`     | Dropout for logits MLP. |

### `CellFlow.predict_from_gmm(covariate_data, n_samples, ...)`

Generate predictions by sampling from the learned GMM and transporting via the ODE.

| Parameter           | Type            | Default | Description |
|---------------------|-----------------|---------|-------------|
| `covariate_data`    | `pd.DataFrame`  | —       | Condition covariates (same columns as training data, including the control key). |
| `n_samples`         | `int`           | `1000`  | Number of cells to generate per condition. |
| `condition_id_key`  | `str \| None`   | `None`  | Column to use as condition name in the output dict. |
| `rep_dict`          | `dict \| None`  | `None`  | Representation dict (defaults to `adata.uns`). |
| `rng`               | `jax.Array \| None` | `None` | PRNG key. |
| `**kwargs`          |                 |         | Forwarded to `diffrax.diffeqsolve`. |

**Returns**: `dict[str, np.ndarray]` — maps condition keys to arrays of shape
`(n_samples, data_dim)`.

### `CellFlow.predict(...)` in GMM mode

When `source_type="gmm"`, the standard `predict()` method automatically replaces
control cells with GMM samples (matching the number of source cells). No code
changes needed compared to standard CellFlow prediction.

---

## How it works under the hood

### Training (each iteration)

1. **Sample target batch** from perturbed cells (same as standard CellFlow).
2. **Sample source from GMM**: the VF condition encoder produces a condition
   embedding, which the GMM module uses to predict means and mixture weights.
   Gumbel-Softmax selects a component, and isotropic noise is added.
3. **OT matching**: source (GMM samples) and target are matched via the
   configured `match_fn` (default: linear OT).
4. **VF update**: standard flow matching loss on the matched pairs.
5. **GMM update**: combined loss = FM loss (VF frozen via `stop_gradient`) +
   `geodesic_loss_weight` × geodesic loss ($\|x_1 - x_0\|^2$).
6. **EMA update** for both VF and GMM inference states.

### Architecture

```
condition → [VF ConditionEncoder] → condition_embedding
                                        ↓
                              [GMMBaseDist.means_net]  → (num_modes, data_dim) means
                              [GMMBaseDist.logits_net] → (num_modes,) logits
                                        ↓
                              Gumbel-Softmax → weighted mean + noise → x₀
                                        ↓
                    [OT matching with target x₁] → (x₀, x₁) pairs
                                        ↓
              [ConditionalVelocityField] → flow matching loss
```

---

## Tips

- **`num_modes`**: Start with 8–16. More modes allow richer base distributions
  but increase the parameter count of the GMM.
- **`variance`**: Controls the spread around each component mean. Lower values
  (0.001–0.01) produce tighter clusters.
- **`temperature`**: Lower temperature (0.1–0.5) makes component selection more
  discrete. Higher values (1.0+) produce softer mixtures.
- **`geodesic_loss_weight`**: Controls how much the base is encouraged to be
  close to the target. Higher values mean shorter transport paths but may
  constrain the learned base too much.
- **`gmm_optimizer`**: The default `optax.adam(1e-3)` works well generally. For
  stability, consider lower learning rates or warm-up schedules.
- Training typically needs the same or slightly more iterations than standard
  CellFlow since the GMM must also converge.
