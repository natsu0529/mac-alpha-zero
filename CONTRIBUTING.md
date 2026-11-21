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
