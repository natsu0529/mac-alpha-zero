import math


def considered_visit_sequence(num_considered: int, num_simulations: int) -> list[int]:
    """Visit count the next root action must have, per simulation, for Sequential Halving.

    Simulation t goes to the best root action among those visited exactly sequence[t]
    times. Starting from `num_considered` actions, this visits all of them equally, then
    halves the set, and so on down to two actions. Same schedule as mctx.
    """
    if num_considered <= 1:
        return list(range(num_simulations))
    log2max = math.ceil(math.log2(num_considered))
    sequence = []
    visits = [0] * num_considered
    remaining = num_considered
    while len(sequence) < num_simulations:
        extra_visits = max(1, int(num_simulations / (log2max * remaining)))
        for _ in range(extra_visits):
            sequence.extend(visits[:remaining])
            for i in range(remaining):
                visits[i] += 1
        remaining = max(2, remaining // 2)
    return sequence[:num_simulations]
