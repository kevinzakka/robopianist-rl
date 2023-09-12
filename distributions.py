import functools
from typing import Type, Callable

import distrax
import flax.linen as nn
import jax.numpy as jnp

from networks import default_init


class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray) -> None:
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        super().__init__(
            distribution=distribution, bijector=distrax.Block(distrax.Tanh(), 1)
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class Normal(nn.Module):
    base_cls: Type[nn.Module] | Callable[..., nn.Module]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> distrax.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        stds = jnp.exp(log_stds)

        if self.squash_tanh:
            return TanhMultivariateNormalDiag(loc=means, scale_diag=stds)
        return distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)


TanhNormal = functools.partial(Normal, squash_tanh=True)


class TanhDeterministic(nn.Module):
    base_cls: Type[nn.Module] | Callable[..., nn.Module]
    action_dim: int

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> jnp.ndarray:
        x = self.base_cls()(inputs, *args, **kwargs)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        return nn.tanh(means)
