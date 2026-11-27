from benchmarks.mac_report import (
    QuickstartResult,
    format_elapsed,
    parse_quickstart,
    parse_speed,
    quickstart_table,
    speed_table,
)

TIME_OUTPUT = """\
      897.31 real       812.02 user        40.11 sys
          1371045888  maximum resident set size
                   0  average shared memory size
              123456  page reclaims
           235994339  instructions retired
          1813103552  peak memory footprint
"""

COMPLETED = (
    """\
kernels=True games=256 simulations=32 symmetries=False
score columns (policy = no search, search = Gumbel search):
  policy vs random, search vs random
 iter minutes    games    loss  scores
    0     0.0        0   0.000    0.50    0.55
   20     5.1     9000   2.345    0.90    0.98
Self-play: 17.52 games/s this run
saved /tmp/report/othello.safetensors
"""
    + TIME_OUTPUT
)

FAILED = """\
kernels=False games=256 simulations=32 symmetries=False
 iter minutes    games    loss  scores
Traceback (most recent call last):
  File "examples/train_othello.py", line 50, in <module>
RuntimeError: [metal::malloc] Resource limit exceeded
""" + TIME_OUTPUT.replace("1813103552", "999")


def test_parse_completed_quickstart():
    result = parse_quickstart("othello", COMPLETED)
    assert result == QuickstartResult(
        game="othello",
        saved=True,
        kernels="True",
        games_per_second=17.52,
        peak_footprint=1813103552,
        elapsed_seconds=897.31,
        error=[],
    )


def test_parse_failed_quickstart_keeps_last_lines_before_time_output():
    result = parse_quickstart("othello", FAILED, error_lines=2)
    assert not result.saved
    assert result.kernels == "False"
    assert result.games_per_second is None
    assert result.peak_footprint == 999
    assert result.error == [
        '  File "examples/train_othello.py", line 50, in <module>',
        "RuntimeError: [metal::malloc] Resource limit exceeded",
    ]


def test_parse_quickstart_without_time_output():
    result = parse_quickstart("connect4", "kernels=True games=256\n")
    assert result.elapsed_seconds is None
    assert result.peak_footprint is None
    assert result.error == ["kernels=True games=256"]


def test_parse_speed():
    output = """\
backend=metal sims=64 network=64x4 (best of 5)
 batch   seconds  ms/sim step       sims/s
     1    0.0418        0.653        1,532
    64    0.0757        1.183       54,096
  4096    4.2711       66.736       61,374
"""
    assert parse_speed(output) == {1: 1532, 64: 54096, 4096: 61374}


def test_tables():
    results = [
        parse_quickstart("connect4", COMPLETED),
        parse_quickstart("othello", FAILED, error_lines=1),
    ]
    table = quickstart_table(results)
    assert "| Connect4 | yes | True | 14:57 | 17.52 | 1,813,103,552 |" in table
    assert "| Othello | failed | False | 14:57 |  | 999 |" in table
    assert "RuntimeError: [metal::malloc] Resource limit exceeded" in table

    speed = speed_table({"a": {1: 10, 64: 2000}, "b": {1: 5}}, batches=(1, 64))
    assert speed.splitlines() == [
        "| Batch | a | b |",
        "|---:|---:|---:|",
        "| 1 | 10 | 5 |",
        "| 64 | 2,000 |  |",
    ]


def test_format_elapsed():
    assert format_elapsed(None) == ""
    assert format_elapsed(59.6) == "1:00"
    assert format_elapsed(2248) == "37:28"
