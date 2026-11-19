import mlx.core as mx
import numpy as np

from mac_alpha_zero import games, reference

ref = reference.Connect4()
game = games.Connect4()


def test_random_games_match_reference():
    rng = np.random.default_rng(0)
    batch = 64
    ref_states = [ref.initial_state() for _ in range(batch)]
    state = game.init(batch)
    for _ in range(42):
        done, value = game.terminal(state)
        legal = np.array(game.legal_mask(state))
        for i, s in enumerate(ref_states):
            expected = ref.terminal_value(s)
            assert bool(done[i]) == (expected is not None)
            assert float(value[i]) == (expected or 0.0)
            np.testing.assert_array_equal(legal[i], ref.legal_mask(s))
            np.testing.assert_array_equal(np.array(state["board"][i]), s.board)
            assert int(state["player"][i]) == s.player
            assert int(state["winner"][i]) == s.winner

        # Finished games get an arbitrary action, which the batched step must tolerate.
        actions = [
            int(rng.choice(np.flatnonzero(legal[i]))) if legal[i].any() else 0 for i in range(batch)
        ]
        ref_states = [
            ref.step(s, a) if ref.terminal_value(s) is None else s
            for s, a in zip(ref_states, actions, strict=True)
        ]
        stepped = game.step(state, mx.array(actions, dtype=mx.int32))
        keep = done
        state = {
            k: mx.where(keep.reshape(-1, *([1] * (v.ndim - 1))), state[k], v)
            for k, v in stepped.items()
        }


def test_from_reference_round_trips():
    states = [ref.from_moves(m) for m in ["", "4", "1213141", "444444"]]
    batched = game.from_reference(states)
    for i, s in enumerate(states):
        np.testing.assert_array_equal(np.array(batched["board"][i]), s.board)
        assert int(batched["player"][i]) == s.player
        assert int(batched["winner"][i]) == s.winner
