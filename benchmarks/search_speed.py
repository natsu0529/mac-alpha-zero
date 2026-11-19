"""Search speed on Connect4: simulations per second by batch size.

    uv run python benchmarks/search_speed.py --backend mlx
    uv run python benchmarks/search_speed.py --backend metal
    uv run --group bench python benchmarks/search_speed.py --backend mctx

By default the evaluator is trivial (uniform priors, zero value), so the numbers measure
tree operations and game logic only. `--network 64x4` evaluates leaves with a randomly
initialized ResNet of 64 channels and 4 blocks instead, which is what self-play does:

    uv run python benchmarks/search_speed.py --backend metal --network 64x4
    uv run --group bench python benchmarks/search_speed.py --backend mctx --network 64x4

The mctx baseline (JAX on the CPU, pgx Connect4) uses mctx's own PUCT formula and value
normalization, so it does the same kind of work but does not build identical trees. Its
network has the same layers as ours, with batch norm folded into a per-channel scale and
shift as in inference.
"""

import argparse
import time


def mlx_runner(batch: int, sims: int, network: tuple[int, int] | None, **search_kwargs):
    import mlx.core as mx

    from mac_alpha_zero import games
    from mac_alpha_zero.mcts import search
    from mac_alpha_zero.network import AlphaZeroNet, network_evaluator

    game = games.Connect4()
    root = game.init(batch)

    if network is None:

        def evaluator(state):
            return mx.full((batch, game.num_actions), 1.0 / game.num_actions), mx.zeros((batch,))
    else:
        mx.random.seed(0)
        net = AlphaZeroNet(game.observation_shape, game.num_actions, *network)
        net.eval()
        mx.eval(net.parameters())
        evaluator = network_evaluator(game, net)

    def run():
        result = search(game, root, evaluator, sims, **search_kwargs)
        mx.eval(result.visit_counts, result.root_value)

    return run


def metal_runner(batch: int, sims: int, network: tuple[int, int] | None):
    return mlx_runner(batch, sims, network, use_kernels=True)


def _jax_resnet(channels: int, num_blocks: int, num_actions: int):
    """Same layers as mac_alpha_zero.network.AlphaZeroNet in inference mode."""
    import jax
    import jax.numpy as jnp

    keys = iter(jax.random.split(jax.random.PRNGKey(0), 4 * num_blocks + 16))

    def conv_params(size, c_in, c_out):
        scale = (2.0 / (size * size * c_in)) ** 0.5
        return {
            "w": scale * jax.random.normal(next(keys), (size, size, c_in, c_out)),
            "scale": jnp.ones((c_out,)),
            "shift": jnp.zeros((c_out,)),
        }

    def linear_params(n_in, n_out):
        return {
            "w": jax.random.normal(next(keys), (n_in, n_out)) / n_in**0.5,
            "b": jnp.zeros((n_out,)),
        }

    params = {
        "stem": conv_params(3, 2, channels),
        "blocks": [
            (conv_params(3, channels, channels), conv_params(3, channels, channels))
            for _ in range(num_blocks)
        ],
        "policy_conv": conv_params(1, channels, 2),
        "policy_out": linear_params(2 * 42, num_actions),
        "value_conv": conv_params(1, channels, 1),
        "value_hidden": linear_params(42, channels),
        "value_out": linear_params(channels, 1),
    }

    def conv_norm(p, x):
        y = jax.lax.conv_general_dilated(
            x, p["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        return y * p["scale"] + p["shift"]

    def linear(p, x):
        return x @ p["w"] + p["b"]

    def apply(x):
        x = jax.nn.relu(conv_norm(params["stem"], x))
        for first, second in params["blocks"]:
            y = jax.nn.relu(conv_norm(first, x))
            x = jax.nn.relu(x + conv_norm(second, y))
        batch = x.shape[0]
        p = jax.nn.relu(conv_norm(params["policy_conv"], x)).reshape(batch, -1)
        v = jax.nn.relu(conv_norm(params["value_conv"], x)).reshape(batch, -1)
        v = jax.nn.relu(linear(params["value_hidden"], v))
        return linear(params["policy_out"], p), jnp.tanh(linear(params["value_out"], v))[:, 0]

    return apply


def mctx_runner(batch: int, sims: int, network: tuple[int, int] | None):
    import jax
    import jax.numpy as jnp
    import mctx
    import pgx

    env = pgx.make("connect_four")
    step = jax.vmap(env.step)
    root_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), batch))
    net = None if network is None else _jax_resnet(*network, env.num_actions)

    def evaluate(state):
        if net is None:
            logits, value = jnp.zeros(state.legal_action_mask.shape), jnp.zeros((batch,))
        else:
            logits, value = net(state.observation.astype(jnp.float32))
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        return logits, value

    def recurrent_fn(params, rng_key, action, state):
        mover = state.current_player
        state = step(state, action)
        reward = state.rewards[jnp.arange(batch), mover]
        discount = jnp.where(state.terminated, 0.0, -1.0)
        logits, value = evaluate(state)
        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=jnp.where(state.terminated, 0.0, value),
        )
        return output, state

    @jax.jit
    def policy(rng_key):
        logits, value = evaluate(root_state)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=root_state)
        return mctx.muzero_policy(
            params=None,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=sims,
            invalid_actions=~root_state.legal_action_mask,
            dirichlet_fraction=0.0,
        ).action_weights

    key = jax.random.PRNGKey(1)

    def run():
        policy(key).block_until_ready()

    return run


BACKENDS = {
    "mlx": mlx_runner,
    "metal": metal_runner,
    "mctx": mctx_runner,
}


def parse_network(text: str) -> tuple[int, int]:
    channels, blocks = text.lower().split("x")
    return int(channels), int(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="mlx")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 64, 1024, 4096])
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--network",
        type=parse_network,
        metavar="CHANNELSxBLOCKS",
        help="evaluate leaves with a ResNet of this size, e.g. 64x4",
    )
    args = parser.parse_args()

    network = "none" if args.network is None else "{}x{}".format(*args.network)
    print(f"backend={args.backend} sims={args.sims} network={network} (best of {args.repeats})")
    print(f"{'batch':>6} {'seconds':>9} {'ms/sim step':>12} {'sims/s':>12}")
    for batch in args.batch:
        run = BACKENDS[args.backend](batch, args.sims, args.network)
        run()  # warm-up: JIT compilation and allocation
        best = float("inf")
        for _ in range(args.repeats):
            start = time.perf_counter()
            run()
            best = min(best, time.perf_counter() - start)
        rate = batch * args.sims / best
        print(f"{batch:>6} {best:>9.4f} {best / args.sims * 1000:>12.3f} {rate:>12,.0f}")


if __name__ == "__main__":
    main()
