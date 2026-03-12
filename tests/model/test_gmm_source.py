"""Integration tests for GMM base distribution (SP-FM mode)."""

import numpy as np
import pandas as pd
import pytest

import cellflow


def _make_small_adata(n_obs: int = 200, n_pca: int = 10):
    """Create a minimal AnnData for GMM source testing."""
    import anndata as ad

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_obs, 50)).astype(np.float32)
    X_pca = rng.standard_normal((n_obs, n_pca)).astype(np.float32)

    drugs = np.array(["drug_a", "drug_b", "control"])
    obs = pd.DataFrame(
        {
            "drug1": pd.Categorical(rng.choice(drugs, n_obs)),
        }
    )
    # Make ~20% control
    ctrl_idx = rng.choice(n_obs, n_obs // 5, replace=False)
    obs.iloc[ctrl_idx, obs.columns.get_loc("drug1")] = "control"
    obs["control"] = obs["drug1"] == "control"
    obs["drug1"] = obs["drug1"].astype("category")

    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = X_pca

    # Drug embeddings (dim=4)
    drug_emb = {d: rng.standard_normal((4, 1)).astype(np.float32) for d in obs["drug1"].cat.categories}
    adata.uns["drug"] = drug_emb

    return adata


class TestGMMSource:
    """Test suite for source_type='gmm' (SP-FM) integration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adata = _make_small_adata()

    def _build_model(self, **extra_prepare_model_kwargs):
        """Build a minimal CellFlow with GMM source."""
        cf = cellflow.model.CellFlow(self.adata, solver="otfm")
        cf.prepare_data(
            sample_rep="X_pca",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        cf.prepare_model(
            condition_embedding_dim=8,
            hidden_dims=(8, 8),
            decoder_dims=(8, 8),
            time_encoder_dims=(8, 8),
            source_type="gmm",
            gmm_kwargs={
                "num_modes": 4,
                "means_hidden_dims": (8,),
                "logits_hidden_dims": (8,),
                "variance": 0.01,
                "temperature": 0.5,
            },
            geodesic_loss_weight=1.0,
            **extra_prepare_model_kwargs,
        )
        return cf

    def test_prepare_model_creates_gmm(self):
        """GMM state should be initialised after prepare_model."""
        cf = self._build_model()
        solver = cf.solver
        assert solver.source_type == "gmm"
        assert solver.gmm is not None
        assert solver.gmm_state is not None
        assert solver.gmm_state_inference is not None

    def test_train_gmm(self):
        """Training with GMM source should complete without error."""
        cf = self._build_model()
        cf.train(num_iterations=3, batch_size=32)
        assert cf.solver.is_trained

    def test_predict_from_gmm(self):
        """predict_from_gmm should return arrays of the right shape."""
        cf = self._build_model()
        cf.train(num_iterations=3, batch_size=32)

        # Build covariate_data for a single condition (must include control column)
        cov_df = pd.DataFrame({
            "drug1": pd.Categorical(["drug_a"]),
            "control": [False],
        })
        preds = cf.predict_from_gmm(cov_df, n_samples=16)
        assert isinstance(preds, dict)
        for key, arr in preds.items():
            assert arr.shape == (16, cf._data_dim), f"Wrong shape for {key}: {arr.shape}"

    def test_predict_standard_api_uses_gmm(self):
        """cf.predict() in GMM mode should replace control cells with GMM samples."""
        cf = self._build_model()
        cf.train(num_iterations=3, batch_size=32)

        adata_pred = self.adata.copy()
        adata_pred.obs["control"] = True  # treat all as control
        pred = cf.predict(
            adata_pred,
            sample_rep="X_pca",
            covariate_data=adata_pred.obs,
            max_steps=3,
            throw=False,
        )
        assert isinstance(pred, dict)
        key, out = next(iter(pred.items()))
        assert out.shape[1] == cf._data_dim

    def test_sample_from_gmm(self):
        """Solver.sample_from_gmm should return properly shaped arrays."""
        cf = self._build_model()
        cf.train(num_iterations=3, batch_size=32)

        # Get a condition dict from the training data
        cond = {k: v[[0], :] for k, v in cf.train_data.condition_data.items()}
        samples = cf.solver.sample_from_gmm(cond, n_samples=32)
        assert samples.shape == (32, cf._data_dim)

    def test_predict_from_gmm_fails_on_control_mode(self):
        """predict_from_gmm should raise ValueError when source_type='control'."""
        cf = cellflow.model.CellFlow(self.adata, solver="otfm")
        cf.prepare_data(
            sample_rep="X_pca",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        cf.prepare_model(
            condition_embedding_dim=8,
            hidden_dims=(8, 8),
            decoder_dims=(8, 8),
            time_encoder_dims=(8, 8),
            source_type="control",
        )
        cf.train(num_iterations=2, batch_size=32)
        cov_df = pd.DataFrame({
            "drug1": pd.Categorical(["drug_a"]),
            "control": [False],
        })
        with pytest.raises(ValueError, match="predict_from_gmm requires source_type='gmm'"):
            cf.predict_from_gmm(cov_df, n_samples=16)
