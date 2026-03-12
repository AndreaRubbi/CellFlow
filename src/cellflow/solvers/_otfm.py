from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import frozen_dict
from flax.training import train_state
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._compat import BaseFlow
from cellflow._types import ArrayLike
from cellflow.networks._gmm_base import GMMBaseDist
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers.utils import ema_update, predict_multi_condition

__all__ = ["OTFlowMatching"]


def _energy_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Energy distance between two sets of empirical samples.

    Computes  E(X,Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]  >= 0,
    with equality iff X and Y have the same distribution.
    """

    def _mean_pairwise_dist(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        diffs = a[:, None, :] - b[None, :, :]
        return jnp.mean(jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8))

    cross = _mean_pairwise_dist(x, y)
    self_x = _mean_pairwise_dist(x, x)
    self_y = _mean_pairwise_dist(y, y)
    return 2 * cross - self_x - self_y


class OTFlowMatching:
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Optionally supports the SP-FM approach :cite:`rubbi:25` where the source
    distribution is a *learnable*, condition-dependent Gaussian mixture model
    (GMM) rather than the observed control population. Set
    ``source_type='gmm'`` and supply ``gmm_kwargs`` to enable this mode.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        probability_path
            Probability path between the source and the target distributions.
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`cellflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        source_type
            Type of source distribution: ``'control'`` (default, use control
            cells) or ``'gmm'`` (learn a condition-dependent GMM base).
        gmm_kwargs
            Keyword arguments forwarded to
            :class:`cellflow.networks.GMMBaseDist` when ``source_type='gmm'``.
        gmm_optimizer
            Optimizer for the GMM base distribution parameters. If :obj:`None`
            and ``source_type='gmm'``, falls back to Adam with lr 1e-3.
        geodesic_loss_weight
            Weight of the geodesic length regularisation loss (Eq. 7 in SP-FM).
            Only used when ``source_type='gmm'``.
        gmm_warmup_iters
            Number of initial iterations where only the VF is trained (GMM
            parameters frozen). Allows the VF to learn a reasonable condition
            embedding before the GMM starts adapting.
        gmm_cooldown_iters
            Number of final iterations where only the VF is trained (GMM
            parameters frozen). Stabilises the velocity field after the GMM
            has converged.
        source_matching_weight
            Weight of the energy distance loss between GMM samples and actual
            control cells.  Keeps the learned base distribution close to the
            true control distribution. Only used when ``source_type='gmm'``.
            Set to ``0.0`` (default) to disable.
        kwargs
            Keyword arguments for :meth:`cellflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        source_type: Literal["control", "gmm"] = "control",
        gmm_kwargs: dict[str, Any] | None = None,
        gmm_optimizer: optax.GradientTransformation | None = None,
        geodesic_loss_weight: float = 1.0,
        gmm_warmup_iters: int = 0,
        gmm_cooldown_iters: int = 0,
        source_matching_weight: float = 0.0,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn)
        self.ema = kwargs.pop("ema", 1.0)
        self.source_type = source_type
        self.geodesic_loss_weight = geodesic_loss_weight
        self.gmm_warmup_iters = gmm_warmup_iters
        self.gmm_cooldown_iters = gmm_cooldown_iters
        self.source_matching_weight = source_matching_weight

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)

        # GMM base distribution (SP-FM)
        self.gmm: GMMBaseDist | None = None
        self.gmm_state: train_state.TrainState | None = None
        self.gmm_state_inference: train_state.TrainState | None = None
        if source_type == "gmm":
            gmm_kwargs = gmm_kwargs or {}
            gmm_kwargs.setdefault("data_dim", self.vf.output_dims[-1])
            self.gmm = GMMBaseDist(**gmm_kwargs)
            gmm_optimizer = gmm_optimizer or optax.adam(1e-3)
            cond_dim = self.vf.condition_embedding_dim
            rng_gmm = jax.random.PRNGKey(42)
            dummy_cond = jnp.ones((1, cond_dim))
            dummy_rng = jax.random.PRNGKey(0)
            gmm_params = self.gmm.init(
                {"params": rng_gmm, "dropout": rng_gmm},
                condition_embedding=dummy_cond,
                rng=dummy_rng,
                training=False,
            )["params"]
            self.gmm_state = train_state.TrainState.create(
                apply_fn=self.gmm.apply, params=gmm_params, tx=gmm_optimizer
            )
            self.gmm_state_inference = train_state.TrainState.create(
                apply_fn=self.gmm.apply, params=gmm_params, tx=gmm_optimizer
            )

        # Flag to control whether GMM params are updated (warmup/cooldown)
        self.gmm_training_active: bool = True

        self.vf_step_fn = self._get_vf_step_fn()
        if source_type == "gmm":
            self.gmm_step_fn = self._get_gmm_step_fn()

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                return flow_matching_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(vf_state.params, time, source, target, conditions, encoder_noise, rng)
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            return vf_state.apply_gradients(grads=grads), loss, grad_norm

        return vf_step_fn

    def _get_gmm_step_fn(self) -> Callable:  # type: ignore[type-arg]
        """Return a JIT-compiled step function that updates the GMM base distribution.

        The GMM loss consists of:

        1. **Flow matching loss**: same as the velocity field loss but gradients
           flow only through the GMM parameters (the VF params are frozen).
        2. **Geodesic length loss** (Eq. 7 of SP-FM): ``E[||x1 - x0||^2]`` where
           ``x0`` is sampled from the GMM and ``x1`` is the matched target sample.
           This encourages the learned base to be close to the target distribution.

        The combined loss is
        ``flow_matching_loss + geodesic_loss_weight * geodesic_loss
        + source_matching_weight * energy_distance(gmm_samples, control_cells)``.
        """

        @jax.jit
        def gmm_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            gmm_state: train_state.TrainState,
            time: jnp.ndarray,
            target: jnp.ndarray,
            source_cells: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                gmm_params: jnp.ndarray,
                vf_params: jnp.ndarray,
                t: jnp.ndarray,
                target: jnp.ndarray,
                source_cells: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> tuple[jnp.ndarray, dict]:
                rng_gmm, rng_flow, rng_encoder, rng_dropout, rng_match = jax.random.split(rng, 5)

                # Get condition embedding from VF (frozen)
                cond_mean, _ = vf_state.apply_fn(
                    {"params": vf_params},
                    conditions,
                    method="get_condition_embedding",
                )  # (1, cond_dim)
                cond_emb = jnp.tile(cond_mean, (target.shape[0], 1))  # (n, cond_dim)

                # Sample from GMM
                source, gmm_means, gmm_logits = gmm_state.apply_fn(
                    {"params": gmm_params},
                    condition_embedding=cond_emb,
                    rng=rng_gmm,
                    training=True,
                    rngs={"dropout": rng_dropout},
                )

                # OT matching between GMM samples and target
                n = target.shape[0]
                tmat = self.match_fn(source, target)
                src_ixs, tgt_ixs = solver_utils.sample_joint(rng_match, tmat)
                source_matched = source[src_ixs]
                target_matched = target[tgt_ixs]

                # Geodesic loss: ||x1 - x0||^2
                geodesic_loss = jnp.mean(jnp.sum((target_matched - source_matched) ** 2, axis=-1))

                # Flow matching loss (VF params frozen via stop_gradient)
                x_t = self.probability_path.compute_xt(rng_flow, t, source_matched, target_matched)
                v_t, _, _ = vf_state.apply_fn(
                    {"params": jax.lax.stop_gradient(vf_params)},
                    t,
                    x_t,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                u_t = self.probability_path.compute_ut(t, x_t, source_matched, target_matched)
                fm_loss = jnp.mean((v_t - u_t) ** 2)

                total_loss = fm_loss + self.geodesic_loss_weight * geodesic_loss

                # Source matching loss: energy distance between GMM samples
                # and actual control cells to prevent GMM from drifting
                if self.source_matching_weight > 0:
                    source_match_loss = _energy_distance(source, source_cells)
                    total_loss = total_loss + self.source_matching_weight * source_match_loss
                else:
                    source_match_loss = jnp.float32(0.0)

                return total_loss, {
                    "fm_loss": fm_loss,
                    "geodesic_loss": geodesic_loss,
                    "source_match_loss": source_match_loss,
                }

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(
                gmm_state.params, vf_state.params, time, target, source_cells, conditions, encoder_noise, rng
            )
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            return gmm_state.apply_gradients(grads=grads), loss, aux, grad_norm

        return gmm_step_fn

    def _get_condition_embedding_for_gmm(
        self, conditions: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Get condition embedding from the VF encoder (for GMM sampling)."""
        cond_mean, _ = self.vf.apply(
            {"params": self.vf_state_inference.params},
            conditions,
            method="get_condition_embedding",
        )
        return cond_mean

    def sample_from_gmm(
        self,
        condition: dict[str, ArrayLike],
        n_samples: int,
        rng: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Sample from the learned GMM base distribution.

        Parameters
        ----------
        condition
            Condition data (same format as for prediction).
        n_samples
            Number of samples to draw.
        rng
            JAX PRNG key.

        Returns
        -------
        Samples of shape ``(n_samples, data_dim)``.
        """
        if self.gmm is None or self.gmm_state_inference is None:
            raise ValueError("GMM base distribution not initialised. Use source_type='gmm'.")
        rng = utils.default_prng_key(rng)

        cond_emb = self._get_condition_embedding_for_gmm(condition)
        cond_emb = jnp.tile(cond_emb, (n_samples, 1))

        samples, _, _ = self.gmm_state_inference.apply_fn(
            {"params": self.gmm_state_inference.params},
            condition_embedding=cond_emb,
            rng=rng,
            training=False,
        )
        return samples

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            optionally ``condition``.

        Returns
        -------
        Loss value.

        Side Effects
        ------------
        Updates ``self._step_diagnostics`` with per-step diagnostics
        (loss, gradient norms, GMM sub-losses) for external logging.
        """
        self._step_diagnostics: dict[str, float] = {}
        tgt = batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise, rng_gmm = jax.random.split(rng, 5)
        n = tgt.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))

        if self.source_type == "gmm":
            # --- SP-FM mode: source is sampled from the learned GMM ---
            cond_emb = self._get_condition_embedding_for_gmm(condition)
            cond_emb_tiled = jnp.tile(cond_emb, (n, 1))

            src, _, _ = self.gmm_state_inference.apply_fn(
                {"params": jax.lax.stop_gradient(self.gmm_state_inference.params)},
                condition_embedding=cond_emb_tiled,
                rng=rng_gmm,
                training=False,
            )

            if self.match_fn is not None:
                tmat = self.match_fn(src, tgt)
                src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
                src, tgt = src[src_ixs], tgt[tgt_ixs]

            # VF step (flow model update)
            self.vf_state, loss, vf_grad_norm = self.vf_step_fn(
                rng_step_fn, self.vf_state, time, src, tgt, condition, encoder_noise
            )

            # GMM step (base distribution update) — skipped during warmup/cooldown
            self._step_diagnostics["vf_grad_norm"] = float(vf_grad_norm)
            if self.gmm_training_active:
                rng_gmm_step = jax.random.fold_in(rng_step_fn, 1)
                tgt_for_gmm = batch["tgt_cell_data"]  # use unmatched target
                src_for_gmm = batch["src_cell_data"]  # actual control cells
                self.gmm_state, gmm_loss, gmm_aux, gmm_grad_norm = self.gmm_step_fn(
                    rng_gmm_step, self.vf_state, self.gmm_state, time, tgt_for_gmm, src_for_gmm, condition, encoder_noise
                )
                self._step_diagnostics.update({
                    "gmm_loss": float(gmm_loss),
                    "gmm_fm_loss": float(gmm_aux["fm_loss"]),
                    "gmm_geodesic_loss": float(gmm_aux["geodesic_loss"]),
                    "gmm_source_match_loss": float(gmm_aux["source_match_loss"]),
                    "gmm_grad_norm": float(gmm_grad_norm),
                })

                # Update GMM inference state (EMA)
                if self.ema == 1.0:
                    self.gmm_state_inference = self.gmm_state
                else:
                    self.gmm_state_inference = self.gmm_state_inference.replace(
                        params=ema_update(self.gmm_state_inference.params, self.gmm_state.params, self.ema)
                    )
        else:
            # --- Standard mode: source is the control population ---
            src = batch["src_cell_data"]
            if self.match_fn is not None:
                tmat = self.match_fn(src, tgt)
                src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
                src, tgt = src[src_ixs], tgt[tgt_ixs]

            self.vf_state, loss, vf_grad_norm = self.vf_step_fn(
                rng_step_fn, self.vf_state, time, src, tgt, condition, encoder_noise
            )
            self._step_diagnostics["vf_grad_norm"] = float(vf_grad_norm)

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        self._step_diagnostics["loss"] = float(loss)
        return loss

    def get_condition_embedding(self, condition: dict[str, ArrayLike], return_as_numpy=True) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.

        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def _predict_jit(
        self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None = None, **kwargs: Any
    ) -> ArrayLike:
        """See :meth:`OTFlowMatching.predict`."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params = self.vf_state_inference.params
            condition, encoder_noise = args
            return self.vf_state_inference.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]

        def solve_ode(x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=(condition, encoder_noise),
                **kwargs,
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None, None]))(x, condition, encoder_noise)
        return x_pred

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        n_samples: int | None = None,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        When ``source_type='gmm'`` and ``x`` is :obj:`None`, source samples are
        drawn from the learned GMM base distribution instead.

        Parameters
        ----------
        x
            A dictionary with keys indicating the name of the condition and values containing
            the input data as arrays. If ``batched=False`` provide an array of shape [batch_size, ...].
            When ``source_type='gmm'``, pass :obj:`None` to sample from the GMM (must also pass ``n_samples``).
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition of input data as arrays. If ``batched=False`` provide an array of shape
            [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        batched
            Whether to use batched prediction. This is only supported if the input has
            the same number of cells for each condition. For example, this works when using
            :class:`~cellflow.data.ValidationSampler` to sample the validation data.
        n_samples
            Number of samples to generate per condition when using GMM source
            (``source_type='gmm'`` and ``x is None``).
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if isinstance(x, dict) and not x:
            return {}

        if isinstance(x, dict):
            # If GMM mode and user wants to sample from GMM for each condition
            if self.source_type == "gmm" and x is not None:
                # User provided control cells but we're in GMM mode, sample from
                # GMM with same n_samples as the provided x to maintain compatibility
                rng_gmm = utils.default_prng_key(rng)
                x_gmm = {}
                for k in x:
                    rng_gmm, rng_sample = jax.random.split(rng_gmm)
                    n = x[k].shape[0]
                    cond_k = condition[k] if isinstance(condition, dict) and k in condition else condition
                    x_gmm[k] = self.sample_from_gmm(cond_k, n, rng=rng_sample)
                x = x_gmm

            return predict_multi_condition(
                predict_fn=lambda x, condition: self._predict_jit(x, condition, rng, **kwargs),
                predict_fn_unbatched=partial(self._predict_jit, rng=rng, **kwargs),
                x=x,
                condition=condition,
            )
        else:
            if x is None and self.source_type == "gmm":
                if n_samples is None:
                    raise ValueError("When x is None and source_type='gmm', n_samples must be provided.")
                rng_gmm = utils.default_prng_key(rng)
                x = self.sample_from_gmm(condition, n_samples, rng=rng_gmm)

            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
