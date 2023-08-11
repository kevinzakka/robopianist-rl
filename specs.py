from dataclasses import dataclass

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
from dm_env import specs


@dataclass(frozen=True)
class EnvironmentSpec:
    observation: specs.Array
    action: specs.Array

    @staticmethod
    def make(env: dm_env.Environment) -> "EnvironmentSpec":
        return EnvironmentSpec(
            observation=env.observation_spec(),
            action=env.action_spec(),
        )

    def sample_action(self, random_state: np.random.RandomState) -> np.ndarray:
        if not isinstance(self.action, specs.BoundedArray):
            raise ValueError("Only BoundedArray action specs are supported.")

        action = random_state.uniform(
            low=self.action.minimum, high=self.action.maximum, size=self.action.shape
        )
        return action.astype(self.action.dtype)

    @property
    def observation_dim(self) -> int:
        return self.observation.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]


def zeros_like(spec: specs.Array) -> jnp.ndarray:
    return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), spec)
