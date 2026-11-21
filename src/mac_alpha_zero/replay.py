import mlx.core as mx
import numpy as np

from mac_alpha_zero.games.base import Symmetries


class ReplayBuffer:
    """Fixed-capacity ring buffer of (observation, policy target, outcome).

    With `symmetries`, every sampled position is transformed by one of them, chosen at
    random, so the stored samples stand for all their symmetric copies.
    """

    def __init__(self, capacity: int, symmetries: Symmetries | None = None):
        self.capacity = capacity
        self.symmetries = symmetries
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
        obs, policy = self._obs[index], self._policy[index]
        if self.symmetries is not None:
            squares, actions = self.symmetries
            k = rng.integers(0, len(squares), size=batch_size)
            flat = obs.reshape(batch_size, squares.shape[1], -1)
            obs = np.take_along_axis(flat, squares[k][:, :, None], axis=1).reshape(obs.shape)
            policy = np.take_along_axis(policy, actions[k], axis=1)
        return mx.array(obs), mx.array(policy), mx.array(self._z[index])

    def save(self, path: str) -> None:
        """Write the stored samples to an .npz file."""
        filled = slice(0, self.size)
        if self._obs is None:
            np.savez(path, next=self._next, z=self._z[filled])
        else:
            np.savez(
                path,
                next=self._next,
                obs=self._obs[filled],
                policy=self._policy[filled],
                z=self._z[filled],
            )

    def load(self, path: str) -> None:
        """Replace the contents with a file written by `save` for the same capacity."""
        data = np.load(path)
        size = len(data["z"])
        if size > self.capacity:
            raise ValueError(f"{path} holds {size} samples, more than capacity {self.capacity}")
        self.size = size
        self._next = int(data["next"]) % self.capacity
        self._z[:size] = data["z"]
        if "obs" in data:
            self._obs = np.zeros((self.capacity, *data["obs"].shape[1:]), dtype=np.float32)
            self._policy = np.zeros((self.capacity, data["policy"].shape[1]), dtype=np.float32)
            self._obs[:size] = data["obs"]
            self._policy[:size] = data["policy"]
