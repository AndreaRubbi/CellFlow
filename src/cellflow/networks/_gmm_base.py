"""Gaussian Mixture Model (GMM) base distribution for SP-FM.

Implements the condition-dependent GMM base distribution from the SP-FM paper
(Rubbi et al., 2025). Instead of using control cells as the source distribution,
the base distribution is a learnable Gaussian mixture whose parameters (means
and mixture weights) are predicted from the condition embedding.

The module uses Gumbel-Softmax for differentiable sampling of mixture components
and supports a geodesic-length regularisation loss that encourages shorter
transport paths between the base distribution and the target.
"""

from collections.abc import Sequence
from dataclasses import field as dc_field
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from cellflow.networks._utils import MLPBlock

__all__ = ["GMMBaseDist"]


class GMMBaseDist(nn.Module):
    """Condition-dependent Gaussian Mixture Model base distribution.

    Given a condition embedding (produced by the velocity field's condition encoder),
    this module predicts:

    - **Component means** of shape ``(num_modes, data_dim)`` via a small MLP ``H_theta``.
    - **Mixture logits** of shape ``(num_modes,)`` via a small MLP ``H_p``.

    A single sample is drawn by:

    1. Applying Gumbel-Softmax to the logits to obtain a soft (or hard) one-hot vector.
    2. Computing the weighted mean of the component means.
    3. Adding isotropic Gaussian noise scaled by ``variance``.

    Parameters
    ----------
    data_dim
        Dimensionality of the data space (e.g. number of PCA components).
    num_modes
        Number of Gaussian mixture components.
    variance
        Fixed isotropic variance for each component.
    temperature
        Temperature for the Gumbel-Softmax.
    hard_gumbel
        If ``True``, use straight-through Gumbel-Softmax (hard samples with
        gradient through the soft relaxation).
    means_hidden_dims
        Hidden layer dimensions for the MLP predicting component means.
    logits_hidden_dims
        Hidden layer dimensions for the MLP predicting mixture logits.
    means_dropout
        Dropout rate for the means predictor.
    logits_dropout
        Dropout rate for the logits predictor.
    """

    data_dim: int = 50
    num_modes: int = 16
    variance: float = 0.01
    temperature: float = 0.5
    hard_gumbel: bool = False
    means_hidden_dims: Sequence[int] = (128,)
    logits_hidden_dims: Sequence[int] = (128,)
    means_dropout: float = 0.0
    logits_dropout: float = 0.0

    def setup(self):
        """Initialise sub-networks."""
        # H_theta: condition_embedding -> (num_modes * data_dim)
        self.means_net = MLPBlock(
            dims=tuple(self.means_hidden_dims) + (self.num_modes * self.data_dim,),
            dropout_rate=self.means_dropout,
            act_last_layer=False,
        )
        # H_p: condition_embedding -> (num_modes,)
        self.logits_net = MLPBlock(
            dims=tuple(self.logits_hidden_dims) + (self.num_modes,),
            dropout_rate=self.logits_dropout,
            act_last_layer=False,
        )

    def predict_params(
        self,
        condition_embedding: jnp.ndarray,
        training: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict GMM parameters from condition embedding.

        Parameters
        ----------
        condition_embedding
            Shape ``(batch, cond_dim)`` or ``(cond_dim,)``.
        training
            Whether the model is in training mode (affects dropout).

        Returns
        -------
        means
            Shape ``(batch, num_modes, data_dim)``.
        logits
            Shape ``(batch, num_modes)``.
        """
        squeeze = condition_embedding.ndim == 1
        if squeeze:
            condition_embedding = condition_embedding[None, :]

        means = self.means_net(condition_embedding, training=training)
        means = means.reshape(condition_embedding.shape[0], self.num_modes, self.data_dim)
        logits = self.logits_net(condition_embedding, training=training)

        if squeeze:
            means = means[0]
            logits = logits[0]
        return means, logits

    def __call__(
        self,
        condition_embedding: jnp.ndarray,
        rng: jax.Array,
        training: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample from the condition-dependent GMM.

        Parameters
        ----------
        condition_embedding
            Condition embedding of shape ``(batch, cond_dim)`` or ``(cond_dim,)``
            (will be broadcast).
        rng
            JAX PRNG key.
        training
            Whether the model is in training mode.

        Returns
        -------
        samples
            Sampled points of shape ``(batch, data_dim)`` (or ``(n, data_dim)``
            after tiling if needed).
        means
            Predicted component means ``(batch, num_modes, data_dim)``.
        logits
            Predicted mixture logits ``(batch, num_modes)``.
        """
        means, logits = self.predict_params(condition_embedding, training=training)
        squeeze = means.ndim == 2  # single sample case

        if squeeze:
            means = means[None, :]
            logits = logits[None, :]

        rng_gumbel, rng_noise = jax.random.split(rng)

        # Gumbel-Softmax
        gumbel_weights = _gumbel_softmax(
            logits, rng_gumbel, temperature=self.temperature, hard=self.hard_gumbel
        )  # (batch, num_modes)

        # Weighted mean: (batch, data_dim)
        weighted_mean = jnp.sum(
            gumbel_weights[:, :, None] * means,
            axis=1,
        )

        # Add noise
        noise = jax.random.normal(rng_noise, weighted_mean.shape) * jnp.sqrt(self.variance)
        samples = weighted_mean + noise

        if squeeze:
            samples = samples[0]
            means = means[0]
            logits = logits[0]

        return samples, means, logits


def _gumbel_softmax(
    logits: jnp.ndarray,
    rng: jax.Array,
    temperature: float = 0.5,
    hard: bool = False,
) -> jnp.ndarray:
    """Gumbel-Softmax with optional straight-through estimator.

    Parameters
    ----------
    logits
        Un-normalised log-probabilities of shape ``(..., num_classes)``.
    rng
        JAX PRNG key.
    temperature
        Temperature parameter.
    hard
        If ``True``, return one-hot vectors with gradients through the soft version.

    Returns
    -------
    Soft (or hard) categorical samples of the same shape as ``logits``.
    """
    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / temperature
    y_soft = jax.nn.softmax(y, axis=-1)

    if hard:
        y_hard = jnp.zeros_like(y_soft).at[jnp.arange(y_soft.shape[0]), jnp.argmax(y_soft, axis=-1)].set(1.0)
        # Straight-through: forward uses hard, backward uses soft
        return y_hard - jax.lax.stop_gradient(y_soft) + y_soft
    return y_soft
