"""Search-only speed on Connect4: simulations per second by batch size.

The evaluator is trivial (uniform priors, zero value) so the numbers measure tree
operations and game logic, not a network.

    uv run python benchmarks/search_speed.py --backend mlx
    uv run python benchmarks/search_speed.py --backend metal
    uv run --group bench python benchmarks/search_speed.py --backend mctx

The mctx baseline (JAX on the CPU, pgx Connect4) uses mctx's own PUCT formula and value
normalization, so it does the same kind of work but does not build identical trees.
"""

import argparse
import time


def mlx_runner(batch: int, sims: int, **search_kwargs):
    import mlx.core as mx

    from mac_alpha_zero import games
    from mac_alpha_zero.mcts import search

    game = games.Connect4()
    root = game.init(batch)

    def evaluator(state):
        return mx.full((batch, game.num_actions), 1.0 / game.num_actions), mx.zeros((batch,))

    def run():
        result = search(game, root, evaluator, sims, **search_kwargs)
        mx.eval(result.visit_counts, result.root_value)

    return run


def mctx_runner(batch: int, sims: int):
    import jax
    import jax.numpy as jnp
    import mctx
    import pgx

    env = pgx.make("connect_four")
    step = jax.vmap(env.step)
    root_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), batch))

    def masked_uniform_logits(state):
        logits = jnp.zeros(state.legal_action_mask.shape)
        return jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    def recurrent_fn(params, rng_key, action, state):
        mover = state.current_player
        state = step(state, action)
        reward = state.rewards[jnp.arange(batch), mover]
        discount = jnp.where(state.terminated, 0.0, -1.0)
        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=masked_uniform_logits(state),
            value=jnp.zeros((batch,)),
        )
        return output, state

    @jax.jit
    def policy(rng_key):
        root = mctx.RootFnOutput(
            prior_logits=masked_uniform_logits(root_state),
            value=jnp.zeros((batch,)),
            embedding=root_state,
        )
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


def metal_runner(batch: int, sims: int):
    return mlx_runner(batch, sims, use_kernels=True)


BACKENDS = {
    "mlx": mlx_runner,
    "metal": metal_runner,
    "mctx": mctx_runner,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="mlx")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 64, 1024, 4096])
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print(f"backend={args.backend} sims={args.sims} (best of {args.repeats})")
    print(f"{'batch':>6} {'seconds':>9} {'ms/sim step':>12} {'sims/s':>12}")
    for batch in args.batch:
        run = BACKENDS[args.backend](batch, args.sims)
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
