from typing import NamedTuple, Optional
import numpy as np
import dm_env


class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_state: np.ndarray


class Buffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int,
        batch_size: int,
    ) -> None:
        self._max_size = max_size
        self._batch_size = batch_size

        # Storage.
        self._states = np.zeros((max_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self._next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self._rewards = np.zeros((max_size), dtype=np.float32)
        self._discounts = np.zeros((max_size), dtype=np.float32)

        self._ptr: int = 0
        self._size: int = 0
        self._prev: Optional[dm_env.TimeStep] = None
        self._action: Optional[np.ndarray] = None
        self._latest: Optional[dm_env.TimeStep] = None

    def insert(
        self,
        timestep: dm_env.TimeStep,
        action: Optional[np.ndarray],
    ) -> None:
        self._prev = self._latest
        self._action = action
        self._latest = timestep

        if action is not None:
            self._states[self._ptr] = self._prev.observation  # type: ignore
            self._actions[self._ptr] = action
            self._next_states[self._ptr] = self._latest.observation
            self._rewards[self._ptr] = self._latest.reward
            self._discounts[self._ptr] = self._latest.discount

            self._ptr = (self._ptr + 1) % self._max_size
            self._size = min(self._size + 1, self._max_size)

    def sample(self) -> Transition:
        self._ind = np.random.randint(0, self._size, size=self._batch_size)
        return Transition(
            state=self._states[self._ind],
            action=self._actions[self._ind],
            reward=self._rewards[self._ind],
            discount=self._discounts[self._ind],
            next_state=self._next_states[self._ind],
        )

    def is_ready(self) -> bool:
        return self._batch_size <= len(self)

    def __len__(self) -> int:
        return self._size
