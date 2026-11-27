# Contributing

On an Apple Silicon Mac, install [uv](https://docs.astral.sh/uv/) and run the same checks as CI from the repository root:

```sh
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Every fast implementation needs an equivalence test against the single-game implementation in `src/mac_alpha_zero/reference/`. Give both paths the same positions and inputs, then compare their moves, visit counts, or values. See [CLAUDE.md](CLAUDE.md#testing-conventions) for the numerical tolerances and tie-breaking rules.

Hosted CI runners may lack a Metal GPU. Keep the suite runnable on the MLX CPU device and skip Metal-only checks when Metal is unavailable. Measure speed locally and include the chip, batch size, and simulation count when reporting results.

Edax is GPL-3.0. Build it locally with `scripts/build_edax.sh` for Othello matches. Keep its source, binary, and evaluation files in the ignored `third_party/` directory. Never copy Edax code into the library or commit its binary.

## Report an Apple Silicon quickstart

When your Mac is available for a measurement, plug it in, leave the lid open, close other apps, and run from the repository root:

```sh
uv sync
uv run python benchmarks/mac_report.py
```

This runs the 10-minute Connect4 and Othello quickstarts, then times search with `benchmarks/search_speed.py`. Expect it to take an hour or more, because the quickstarts also evaluate the agent outside their 10-minute training budget. Each command runs under `caffeinate -i` so the Mac does not sleep, and under macOS `time -l` for peak memory. Weights go to a temporary report directory, not `checkpoints/`. Output streams to the terminal and to log files in that directory, and the script ends with a Markdown report saved there as `report.md`. Pass `--no-speed` to run only the quickstarts, `--quickstart connect4` to run one game, or `--quickstart` alone to time search only.

Paste the system lines and the quickstart table into [issue #10](https://github.com/natsu0529/mac-alpha-zero/issues/10), and the system lines and the search-speed table into [issue #31](https://github.com/natsu0529/mac-alpha-zero/issues/31). The system lines give the commit, chip, installed memory, macOS, MLX and Python versions, and whether Metal and Edax are available. A failed run is useful too. The report marks it as failed and includes its last lines.

The quickstart table uses the final `Self-play:` line for throughput. It counts completed games over time spent generating and adding samples during this invocation. Peak memory is the `peak memory footprint` line from `time -l` for the Python process. The footprint covers the whole run, including evaluation. If your macOS version omits this field, the cell stays blank. The `saved` line confirms the training command completed.

The speed table gives simulations per second at batch sizes 1, 64, 256, 1,024 and 4,096 with 64 simulations. It covers the Metal kernels and pure MLX, each with no network and with a random 64-channel, 4-block ResNet. To add mctx on the CPU as a baseline, run `uv run --group bench python benchmarks/mac_report.py --quickstart --mctx`.

If the script fails, run the steps by hand and fill in the same table:

```sh
report_dir=$(mktemp -d)
caffeinate -i /usr/bin/time -l .venv/bin/python -u examples/train_connect4.py --minutes 10 --checkpoint "$report_dir/connect4.safetensors" 2>&1 | tee "$report_dir/connect4.txt"
caffeinate -i /usr/bin/time -l .venv/bin/python -u examples/train_othello.py --minutes 10 --checkpoint "$report_dir/othello.safetensors" 2>&1 | tee "$report_dir/othello.txt"
```

| Game | Completed and saved weights? | Metal kernels (`kernels=`) | Elapsed (min:sec) | Self-play games/s | Peak memory footprint (bytes) |
|---|---|---|---:|---:|---:|
| Connect4 | | | | | |
| Othello | | | | | |

Maintainers can follow the [v0.1 release guide](docs/v0.1-release.md) to prepare and verify the model, GitHub assets, and wheel.
