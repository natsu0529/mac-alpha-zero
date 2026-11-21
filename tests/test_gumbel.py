import mlx.core as mx
import numpy as np
import pytest

from mac_alpha_zero import games, kernels, reference
from mac_alpha_zero.gumbel import gumbel_search
from mac_alpha_zero.sequential_halving import considered_visit_sequence

from .test_batched_mcts import batched_evaluator, cpu, random_positions, ref_evaluator  # noqa: F401
from .test_kernels import evaluator as evaluator32

ref = reference.Connect4()
game = games.Connect4()


def gumbel_noise(batch, seed):
    return np.random.default_rng(seed).gumbel(size=(batch, 7))


def test_sequence_halves_the_considered_actions():
    # 4 actions once each, then the best 2 twice each.
    assert considered_visit_sequence(4, 8) == [0, 0, 0, 0, 1, 1, 2, 2]
    assert considered_visit_sequence(1, 3) == [0, 1, 2]
    assert len(considered_visit_sequence(16, 5)) == 5


@pytest.mark.parametrize("max_considered", [2, 4, 16])
def test_reference_follows_sequential_halving(max_considered):
    result = reference.run_gumbel_mcts(
        ref,
        ref.initial_state(),
        ref_evaluator,
        num_simulations=33,
        gumbel=gumbel_noise(1, 0)[0],
        max_considered=max_considered,
    )
    visits = result.visit_counts
    assert visits.sum() == 32
    assert np.count_nonzero(visits) == min(max_considered, 7)
    # The played action is one of the most visited.
    assert visits[result.action] == visits.max()
    assert result.improved_policy.sum() == pytest.approx(1.0)


def test_reference_policy_is_zero_on_illegal_actions():
    state = ref.from_moves("444444")
    result = reference.run_gumbel_mcts(ref, state, ref_evaluator, 20, gumbel_noise(1, 1)[0])
    assert result.improved_policy[3] == 0.0
    assert result.visit_counts[3] == 0
    assert result.action != 3


def test_reference_finds_immediate_win():
    state = ref.from_moves("121212")
    evaluator = reference.random_rollout_evaluator(ref, np.random.default_rng(0))
    for seed in range(5):
        result = reference.run_gumbel_mcts(ref, state, evaluator, 64, gumbel_noise(1, seed)[0])
        assert result.action == 0
        assert np.argmax(result.improved_policy) == 0


@pytest.mark.parametrize(
    ("num_simulations", "max_considered"), [(1, 16), (2, 16), (17, 4), (64, 16)]
)
@pytest.mark.parametrize("improved_nonroot", [False, True])
@pytest.mark.usefixtures("cpu")
def test_batched_matches_reference_in_float64(
    num_simulations,
    max_considered,
    improved_nonroot,
):
    positions = [ref.initial_state(), ref.from_moves("121212"), ref.from_moves("444444")]
    positions += random_positions(13, seed=num_simulations)
    noise = gumbel_noise(len(positions), seed=num_simulations)
    result = gumbel_search(
        game,
        game.from_reference(positions),
        batched_evaluator,
        num_simulations,
        gumbel=mx.array(noise, dtype=mx.float64),
        max_considered=max_considered,
        dtype=mx.float64,
        improved_nonroot=improved_nonroot,
    )
    for i, state in enumerate(positions):
        expected = reference.run_gumbel_mcts(
            ref,
            state,
            ref_evaluator,
            num_simulations,
            noise[i],
            max_considered=max_considered,
            improved_nonroot=improved_nonroot,
        )
        np.testing.assert_array_equal(np.array(result.visit_counts[i]), expected.visit_counts)
        assert int(result.action[i]) == expected.action
        # MLX 0.32.4 evaluates exp at float32 precision even for float64 inputs, so the
        # softmax output is only close.
        np.testing.assert_allclose(
            np.array(result.improved_policy[i]), expected.improved_policy, rtol=1e-6, atol=1e-9
        )
        assert float(result.root_value[i]) == pytest.approx(expected.root_value, abs=1e-12)


def test_nonroot_rule_changes_the_search_below_identical_root_allocations():
    setup = ref, ref.initial_state(), ref_evaluator, 32, gumbel_noise(1, 0)[0]
    puct = reference.run_gumbel_mcts(*setup)
    improved = reference.run_gumbel_mcts(*setup, improved_nonroot=True)
    np.testing.assert_array_equal(puct.visit_counts, improved.visit_counts)
    assert improved.root_value != puct.root_value


def test_samples_noise_when_none_is_given():
    a = gumbel_search(game, game.init(8), evaluator32, 16, key=mx.random.key(0))
    b = gumbel_search(game, game.init(8), evaluator32, 16, key=mx.random.key(0))
    assert np.array(a.action).tolist() == np.array(b.action).tolist()
    assert np.array(a.visit_counts.sum(axis=1)).tolist() == [15] * 8


@pytest.mark.skipif(not kernels.is_available(), reason="needs the Metal GPU")
@pytest.mark.parametrize("improved_nonroot", [False, True])
def test_kernels_match_pure_mlx(improved_nonroot):
    root = game.from_reference(random_positions(64, seed=5))
    noise = mx.array(gumbel_noise(64, seed=5), dtype=mx.float32)
    expected = gumbel_search(
        game, root, evaluator32, 64, gumbel=noise, improved_nonroot=improved_nonroot
    )
    result = gumbel_search(
        game,
        root,
        evaluator32,
        64,
        gumbel=noise,
        use_kernels=True,
        improved_nonroot=improved_nonroot,
    )
    np.testing.assert_array_equal(np.array(result.visit_counts), np.array(expected.visit_counts))
    np.testing.assert_array_equal(np.array(result.action), np.array(expected.action))
