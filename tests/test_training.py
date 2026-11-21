import argparse
import json

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest

from mac_alpha_zero import arena, games, runner
from mac_alpha_zero.gumbel import gumbel_search
from mac_alpha_zero.network import AlphaZeroNet, network_evaluator
from mac_alpha_zero.replay import ReplayBuffer
from mac_alpha_zero.selfplay import SelfPlay
from mac_alpha_zero.train import loss_fn, train_step

game = games.Connect4()


def small_net():
    mx.random.seed(0)
    return AlphaZeroNet(game.observation_shape, game.num_actions, channels=16, num_blocks=1)


def test_observation_is_from_the_player_to_move():
    state = game.step(game.init(1), mx.array([3], dtype=mx.int32))
    obs = np.array(game.observe(state))
    assert obs.shape == (1, 6, 7, 2)
    # The second player is to move and sees the first player's stone as the opponent's.
    assert obs[0, 0, 3].tolist() == [0.0, 1.0]
    assert obs.sum() == 1.0


def test_network_output_shapes_and_range():
    net = small_net()
    logits, value = net(game.observe(game.init(5)))
    assert logits.shape == (5, 7)
    assert value.shape == (5,)
    assert float(mx.abs(value).max()) <= 1.0


def test_network_evaluator_drives_search():
    net = small_net()
    net.eval()
    result = gumbel_search(game, game.init(4), network_evaluator(game, net), 16)
    assert np.array(result.visit_counts.sum(axis=1)).tolist() == [15] * 4
    np.testing.assert_allclose(np.array(result.improved_policy.sum(axis=1)), 1.0, rtol=1e-5)


def scripted_actor(state):
    # The first player stacks column 0 and the second column 1, so the first player wins
    # with their fourth stone, on the 7th move.
    batch = state["player"].shape[0]
    actions = mx.where(state["player"] == 1, 0, 1).astype(mx.int32)
    policy = mx.zeros((batch, 7))
    policy[mx.arange(batch), actions] = 1.0
    return actions, policy


def test_selfplay_assigns_outcomes_from_each_movers_view():
    selfplay = SelfPlay(game, batch_size=3)
    for _ in range(6):
        obs, policy, z = selfplay.step(scripted_actor)
        assert len(obs) == 0
    obs, policy, z = selfplay.step(scripted_actor)
    assert selfplay.games_finished == 3
    assert obs.shape == (21, 6, 7, 2)
    assert policy.shape == (21, 7)
    # Samples come out newest first per game: the winner's move, the loser's, ...
    assert z[:7].tolist() == [1, -1, 1, -1, 1, -1, 1]
    # The winner moved to column 0, the loser to column 1.
    assert policy[z == 1][:, 0].all() and policy[z == -1][:, 1].all()
    # Every game restarted from the empty board.
    assert int(mx.abs(selfplay.state["board"]).sum()) == 0
    assert np.array(selfplay.state["player"]).tolist() == [1, 1, 1]


def test_replay_buffer_keeps_the_newest_samples():
    buffer = ReplayBuffer(capacity=5)
    for start in (0, 3, 6):
        z = np.arange(start, start + 3, dtype=np.float32)
        buffer.add(np.zeros((3, 6, 7, 2)), np.zeros((3, 7)), z)
    assert buffer.size == 5
    _, _, z = buffer.sample(200, np.random.default_rng(0))
    assert set(np.array(z).tolist()) == {4.0, 5.0, 6.0, 7.0, 8.0}


def test_train_step_reduces_loss_on_a_fixed_batch():
    net = small_net()
    net.train()
    rng = np.random.default_rng(0)
    obs = mx.array(rng.integers(0, 2, size=(32, 6, 7, 2)).astype(np.float32))
    policy = mx.array(np.eye(7, dtype=np.float32)[rng.integers(0, 7, size=32)])
    z = mx.array(rng.choice([-1.0, 1.0], size=32).astype(np.float32))
    before = loss_fn(net, obs, policy, z).item()
    optimizer = optim.Adam(learning_rate=1e-2)
    for _ in range(30):
        train_step(net, optimizer, obs, policy, z)
    assert loss_fn(net, obs, policy, z).item() < 0.5 * before


def test_arena_counts_every_game():
    result = arena.evaluate(game, arena.random_agent(game, 1), arena.random_agent(game, 2), 32)
    assert result.wins + result.draws + result.losses == 32
    assert 0.0 <= result.score <= 1.0


def test_search_agent_beats_random():
    def heuristic(state):
        batch = state["player"].shape[0]
        return mx.full((batch, 7), 1 / 7), mx.zeros((batch,))

    result = arena.evaluate(
        game, arena.puct_agent(game, heuristic, 64), arena.random_agent(game, 3), 32
    )
    assert result.score > 0.8


def test_replay_buffer_round_trips_through_a_file(tmp_path):
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=10)
    buffer.add(rng.random((14, 6, 7, 2)), rng.random((14, 7)), rng.random(14))
    buffer.save(str(tmp_path / "buffer.npz"))
    loaded = ReplayBuffer(capacity=10)
    loaded.load(str(tmp_path / "buffer.npz"))
    assert loaded.size == buffer.size
    for a, b in zip(
        buffer.sample(32, np.random.default_rng(1)),
        loaded.sample(32, np.random.default_rng(1)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    # New samples go to the same slot in both.
    extra = rng.random((1, 6, 7, 2)), rng.random((1, 7)), rng.random(1)
    buffer.add(*extra)
    loaded.add(*extra)
    np.testing.assert_array_equal(buffer._obs, loaded._obs)


def run_training(tmp_path, *extra):
    parser = argparse.ArgumentParser()
    runner.add_arguments(parser)
    arguments = [
        *("--checkpoint", str(tmp_path / "net.safetensors")),
        *("--games", "8", "--simulations", "4", "--channels", "8", "--blocks", "1"),
        *("--steps-per-iteration", "8", "--train-steps", "2", "--batch-size", "16"),
        *("--eval-every", "2", "--eval-games", "4", "--minutes", "0.02"),
    ]
    args = parser.parse_args([*arguments, *extra])
    opponents = {"random": (arena.random_agent(game, seed=1), None)}
    runner.run(game, args, opponents)
    log = (tmp_path / "net.log.jsonl").read_text().splitlines()
    return [json.loads(line) for line in log], json.loads((tmp_path / "net.json").read_text())


def test_training_run_logs_and_resumes(tmp_path):
    log, counters = run_training(tmp_path)
    assert log[0]["iteration"] == 0
    assert log[0]["selfplay_games_per_second"] is None
    assert log[-1]["selfplay_games_per_second"] >= 0
    assert set(log[-1]["scores"]) == {"policy vs random", "search vs random"}
    assert log[-1]["iteration"] == counters["iteration"] > 0

    # Starting again on the same files without --resume is refused.
    with pytest.raises(SystemExit):
        run_training(tmp_path)

    resumed_log, resumed = run_training(tmp_path, "--resume")
    assert resumed_log[: len(log)] == log
    assert resumed["iteration"] > counters["iteration"]
    assert resumed["seconds"] > counters["seconds"]
    assert resumed["games"] >= counters["games"]
    assert [r["iteration"] for r in resumed_log] == sorted({r["iteration"] for r in resumed_log})


def test_learning_rate_decay_survives_resume(tmp_path):
    log, counters = run_training(tmp_path, "--lr-half-life-minutes", "0.05")
    assert counters["lr_schedule"] == {"initial": 0.002, "half_life_minutes": 0.05}
    resumed_log, resumed = run_training(tmp_path, "--resume")
    assert resumed["lr_schedule"] == counters["lr_schedule"]
    assert 0 < resumed_log[-1]["learning_rate"] < log[-1]["learning_rate"]
    assert resumed_log[-1]["learning_rate"] <= 0.002 * 0.5 ** (log[-1]["minutes"] / 0.05) + 1e-9
