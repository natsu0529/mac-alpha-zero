import pytest

from examples.play_othello import play


@pytest.mark.parametrize("human", [1, -1])
def test_terminal_game_accepts_coordinates_and_handles_passing(human, capsys):
    from mac_alpha_zero import reference

    game = reference.Othello()

    def first_legal(state):
        return game.legal_mask(state).nonzero()[0][0]

    def first_listed(prompt):
        return prompt.split("(")[1].split(")")[0].split(",")[0].upper()

    play(first_legal, human, first_listed)
    output = capsys.readouterr().out
    assert "passes." in output
    assert "Black 19, white 45." in output
    assert ("Agent wins." if human == 1 else "You win.") in output


def test_illegal_move_retries_and_quit_stops_before_agent_turn(capsys):
    answers = iter(["a1", "q"])
    play(lambda state: pytest.fail("unexpected agent turn"), 1, lambda _: next(answers))
    assert "Choose one of the listed squares." in capsys.readouterr().out
