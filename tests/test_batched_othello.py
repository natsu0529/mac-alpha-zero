import mlx.core as mx
import numpy as np
import pytest

from mac_alpha_zero import arena, games, reference
from mac_alpha_zero.selfplay import SelfPlay

ref = reference.Othello()
game = games.Othello()

# Number of move sequences of each length from the starting position.
PERFT = [1, 4, 12, 56, 244, 1396, 8200, 55092, 390216, 3005288]


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    if request.param == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal is not available")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if request.param == "cpu" else mx.gpu)
    yield
    mx.set_default_device(previous)


def test_random_games_match_reference(device):
    rng = np.random.default_rng(0)
    batch = 32
    ref_states = [ref.initial_state() for _ in range(batch)]
    state = game.init(batch)
    finished = np.zeros(batch, dtype=bool)
    while not finished.all():
        done, value = game.terminal(state)
        legal = np.array(game.legal_mask(state))
        boards = np.array(game.boards(state))
        observation = np.array(game.observe(state))
        for i, s in enumerate(ref_states):
            expected = ref.terminal_value(s)
            assert bool(done[i]) == (expected is not None)
            assert float(value[i]) == (expected or 0.0)
            np.testing.assert_array_equal(legal[i], ref.legal_mask(s))
            np.testing.assert_array_equal(boards[i], s.board)
            np.testing.assert_array_equal(observation[i, :, :, 0], s.board == s.player)
            np.testing.assert_array_equal(observation[i, :, :, 1], s.board == -s.player)
            assert int(state["player"][i]) == s.player
        finished = np.array(done)

        # Finished games get an arbitrary action, which the batched step must tolerate.
        actions = [
            int(rng.choice(np.flatnonzero(legal[i]))) if legal[i].any() else 0 for i in range(batch)
        ]
        ref_states = [
            s if finished[i] else ref.step(s, actions[i]) for i, s in enumerate(ref_states)
        ]
        stepped = game.step(state, mx.array(actions, dtype=mx.int32))
        state = {k: mx.where(done, state[k], v) for k, v in stepped.items()}


def test_perft(device):
    state = game.init(1)
    for depth, expected in enumerate(PERFT):
        assert state["own"].shape[0] == expected
        if depth == len(PERFT) - 1:
            break
        index, action = np.nonzero(np.array(game.legal_mask(state)))
        parents = {k: v[mx.array(index)] for k, v in state.items()}
        state = game.step(parents, mx.array(action.astype(np.int32)))
        mx.eval(*state.values())


def test_from_reference_round_trips():
    rng = np.random.default_rng(1)
    states = [ref.initial_state()]
    while ref.terminal_value(states[-1]) is None:
        legal = np.flatnonzero(ref.legal_mask(states[-1]))
        states.append(ref.step(states[-1], int(rng.choice(legal))))
    batched = game.from_reference(states)
    np.testing.assert_array_equal(np.array(game.boards(batched)), [s.board for s in states])
    np.testing.assert_array_equal(np.array(batched["player"]), [s.player for s in states])


def test_self_play_and_arena_run_whole_games():
    results = arena.play_match(game, arena.random_agent(game, 1), arena.random_agent(game, 2), 16)
    assert set(np.unique(results)) <= {-1.0, 0.0, 1.0}

    agent = arena.random_agent(game, 3)

    def actor(state):
        return agent(state), game.legal_mask(state).astype(mx.float32)

    selfplay = SelfPlay(game, 8)
    samples = 0
    for _ in range(70):
        observations, policies, outcomes = selfplay.step(actor)
        assert observations.shape[1:] == (8, 8, 2)
        assert policies.shape[1:] == (65,)
        assert set(np.unique(outcomes)) <= {-1.0, 0.0, 1.0}
        samples += len(outcomes)
    assert selfplay.games_finished >= 8
    assert samples >= 8 * 9
