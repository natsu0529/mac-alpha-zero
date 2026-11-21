import numpy as np
import pytest

from mac_alpha_zero import games, reference
from mac_alpha_zero.replay import ReplayBuffer


@pytest.mark.parametrize(
    ("game", "oracle", "state"),
    [
        (games.Connect4(), reference.Connect4(), reference.Connect4().from_moves("111111234")),
        (
            games.Othello(),
            reference.Othello(),
            reference.Othello().step(reference.Othello().initial_state(), 19),
        ),
        (
            games.Othello(),
            reference.Othello(),
            reference.OthelloState(np.pad([[-1, 1]], ((0, 7), (0, 6))), 1),
        ),
    ],
)
def test_sampled_symmetries_match_transformed_positions(game, oracle, state):
    squares, actions = game.symmetries()
    assert len(np.unique(squares, axis=0)) == oracle.num_symmetries
    obs = np.array(game.observe(game.from_reference([state])))
    policy = np.arange(game.num_actions, dtype=np.float32)
    buffer = ReplayBuffer(1, (squares, actions))
    buffer.add(obs, policy[None, :], np.array([-1.0]))
    observed, targets, outcomes = map(np.array, buffer.sample(256, np.random.default_rng(19)))
    np.testing.assert_array_equal(outcomes, -np.ones(256))
    for k in range(oracle.num_symmetries):
        transformed = oracle.transform_state(state, k)
        masked = oracle.legal_mask(state)[actions[k]]
        np.testing.assert_array_equal(masked, oracle.legal_mask(transformed))
        expected_policy = oracle.transform_policy(policy, k)
        selected = np.all(targets == expected_policy, axis=1)
        assert selected.any()
        expected_obs = np.array(game.observe(game.from_reference([transformed])))
        np.testing.assert_equal(observed[selected], np.repeat(expected_obs, selected.sum(), axis=0))
