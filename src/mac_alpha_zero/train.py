import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mac_alpha_zero.network import AlphaZeroNet


def loss_fn(net: AlphaZeroNet, obs: mx.array, policy: mx.array, z: mx.array) -> mx.array:
    """Cross-entropy to the search policy plus squared error to the game outcome."""
    logits, value = net(obs)
    policy_loss = -(policy * nn.log_softmax(logits, axis=-1)).sum(axis=-1).mean()
    value_loss = mx.square(value - z).mean()
    return policy_loss + value_loss


def train_step(
    net: AlphaZeroNet,
    optimizer: optim.Optimizer,
    obs: mx.array,
    policy: mx.array,
    z: mx.array,
) -> float:
    """One gradient step. The network must be in train mode."""
    loss, grads = nn.value_and_grad(net, loss_fn)(net, obs, policy, z)
    optimizer.update(net, grads)
    mx.eval(net.parameters(), optimizer.state)
    return loss.item()
