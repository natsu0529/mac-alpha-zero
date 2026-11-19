"""Slow, single-game NumPy implementations used as the correctness oracle.

Everything here favours clarity over speed. The batched MLX implementations and
the Metal kernels are tested against these.
"""

from mac_alpha_zero.reference.connect4 import Connect4, Connect4State
from mac_alpha_zero.reference.game import Game
from mac_alpha_zero.reference.gumbel import GumbelResult, run_gumbel_mcts
from mac_alpha_zero.reference.mcts import (
    Evaluator,
    MCTSResult,
    random_rollout_evaluator,
    run_mcts,
)
from mac_alpha_zero.reference.solver import solve

__all__ = [
    "Connect4",
    "Connect4State",
    "Evaluator",
    "Game",
    "GumbelResult",
    "MCTSResult",
    "random_rollout_evaluator",
    "run_gumbel_mcts",
    "run_mcts",
    "solve",
]
