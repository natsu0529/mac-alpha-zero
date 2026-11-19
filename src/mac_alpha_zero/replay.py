import mlx.core as mx
import numpy as np


class ReplayBuffer:
    """Fixed-capacity ring buffer of (observation, policy target, outcome)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self._next = 0
        self._obs = None
        self._policy = None
        self._z = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs: np.ndarray, policy: np.ndarray, z: np.ndarray) -> None:
        if len(obs) == 0:
            return
        if self._obs is None:
            self._obs = np.zeros((self.capacity, *obs.shape[1:]), dtype=np.float32)
            self._policy = np.zeros((self.capacity, policy.shape[1]), dtype=np.float32)
        # Only the newest `capacity` samples can survive.
        obs, policy, z = obs[-self.capacity :], policy[-self.capacity :], z[-self.capacity :]
        index = (self._next + np.arange(len(obs))) % self.capacity
        self._obs[index] = obs
        self._policy[index] = policy
        self._z[index] = z
        self._next = int((self._next + len(obs)) % self.capacity)
        self.size = min(self.capacity, self.size + len(obs))

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[mx.array, mx.array, mx.array]:
        index = rng.integers(0, self.size, size=batch_size)
        return (
            mx.array(self._obs[index]),
            mx.array(self._policy[index]),
            mx.array(self._z[index]),
        )
