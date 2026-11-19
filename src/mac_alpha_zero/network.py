import mlx.core as mx
import mlx.nn as nn

from mac_alpha_zero.games.base import BatchedGame, State
from mac_alpha_zero.mcts import Evaluator


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return nn.relu(x + y)


class AlphaZeroNet(nn.Module):
    """AlphaZero-style ResNet with a policy head and a value head.

    Input is (B, H, W, C) (MLX convolutions are channels-last). Returns policy logits
    (B, num_actions) and a value (B,) in [-1, 1] for the player to move.
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        num_actions: int,
        channels: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        height, width, planes = observation_shape
        self.stem = nn.Conv2d(planes, channels, 3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm(channels)
        self.blocks = [ResidualBlock(channels) for _ in range(num_blocks)]

        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_norm = nn.BatchNorm(2)
        self.policy_out = nn.Linear(2 * height * width, num_actions)

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_norm = nn.BatchNorm(1)
        self.value_hidden = nn.Linear(height * width, channels)
        self.value_out = nn.Linear(channels, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = nn.relu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        batch = x.shape[0]

        p = nn.relu(self.policy_norm(self.policy_conv(x)))
        logits = self.policy_out(p.reshape(batch, -1))

        v = nn.relu(self.value_norm(self.value_conv(x)))
        v = nn.relu(self.value_hidden(v.reshape(batch, -1)))
        value = mx.tanh(self.value_out(v)).squeeze(-1)
        return logits, value


def network_evaluator(game: BatchedGame, net: AlphaZeroNet) -> Evaluator:
    """Search evaluator backed by `net`. Put the network in eval mode before searching."""

    def evaluate(state: State) -> tuple[mx.array, mx.array]:
        logits, value = net(game.observe(state))
        return mx.softmax(logits, axis=-1), value

    return evaluate
