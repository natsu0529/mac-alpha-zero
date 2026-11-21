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

When your Mac is available for a measurement, plug it in, leave the lid open, and run each game separately. From the repository root:

```sh
uv sync
report_dir=$(mktemp -d)
git rev-parse --short HEAD
sysctl -n machdep.cpu.brand_string hw.memsize
sw_vers -productVersion
.venv/bin/python -c 'from importlib.metadata import version; print(version("mlx"))'

caffeinate -i /usr/bin/time -l .venv/bin/python -u examples/train_connect4.py --minutes 10 --checkpoint "$report_dir/connect4.safetensors" 2>&1 | tee "$report_dir/connect4.txt"
caffeinate -i /usr/bin/time -l .venv/bin/python -u examples/train_othello.py --minutes 10 --checkpoint "$report_dir/othello.safetensors" 2>&1 | tee "$report_dir/othello.txt"
```

Copy this table into [issue #10](https://github.com/natsu0529/mac-alpha-zero/issues/10). Include the commit, chip, installed memory, macOS and MLX versions, and whether Edax was installed. A failed run is useful too. Include the last error and mark its result as failed.

| Game | Completed and saved weights? | Metal kernels (`kernels=`) | Self-play games/s | Peak memory footprint (bytes) |
|---|---|---|---:|---:|
| Connect4 | | | | |
| Othello | | | | |

Use the final `Self-play:` line for throughput. It counts completed games over time spent generating and adding samples during this invocation. Use the `peak memory footprint` line from macOS `time -l` for the Python process. The footprint covers the whole run, including evaluation. If your macOS version omits this field, leave it blank. The `saved` line confirms the training command completed.

Maintainers can follow the [v0.1 release guide](docs/v0.1-release.md) to prepare and verify the model, GitHub assets, and wheel.
