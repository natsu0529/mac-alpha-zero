"""Measure this Mac and print a Markdown report for the Apple Silicon issues.

    uv run python benchmarks/mac_report.py                       # quickstarts and search speed
    uv run python benchmarks/mac_report.py --no-speed            # quickstarts only
    uv run python benchmarks/mac_report.py --quickstart          # search speed only
    uv run --group bench python benchmarks/mac_report.py --mctx  # also time mctx on the CPU

Each quickstart runs `examples/train_<game>.py --minutes N` under `caffeinate -i` and
`/usr/bin/time -l`, with weights saved to a temporary directory instead of `checkpoints/`.
Search speed runs `benchmarks/search_speed.py` for the Metal kernels and pure MLX, with and
without the 64x4 network. Output streams to the terminal and to log files. The final report
and logs stay in the printed report directory. Paste the quickstart table into issue #10 and
the speed table into issue #31.
"""

import argparse
import importlib.metadata
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES = ("connect4", "othello")
GAME_NAMES = {"connect4": "Connect4", "othello": "Othello"}
SPEED_BATCHES = (1, 64, 256, 1024, 4096)
SPEED_NETWORKS = (None, "64x4")

_TIME_REAL = re.compile(r"^\s*([\d.]+) real\s+[\d.]+ user\s+[\d.]+ sys\s*$")
_FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$")
_KERNELS = re.compile(r"^kernels=(\w+)\b")
_SELFPLAY = re.compile(r"^Self-play: ([\d.]+) games/s")
_SAVED = re.compile(r"^saved \S")
_SPEED_HEADER = re.compile(r"^backend=(\w+) sims=(\d+) network=(\S+)")
_SPEED_ROW = re.compile(r"^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+([\d,]+)\s*$")


@dataclass
class QuickstartResult:
    game: str
    saved: bool
    kernels: str | None
    games_per_second: float | None
    peak_footprint: int | None
    elapsed_seconds: float | None
    error: list[str]


def parse_quickstart(game: str, output: str, error_lines: int = 10) -> QuickstartResult:
    """Read a quickstart log written by the runner followed by macOS `time -l`."""
    lines = output.splitlines()
    kernels = games_per_second = footprint = elapsed = None
    saved = False
    time_start = len(lines)
    for index, line in enumerate(lines):
        if match := _KERNELS.match(line):
            kernels = match.group(1)
        elif match := _SELFPLAY.match(line):
            games_per_second = float(match.group(1))
        elif _SAVED.match(line):
            saved = True
        elif match := _TIME_REAL.match(line):
            elapsed = float(match.group(1))
            time_start = min(time_start, index)
        elif match := _FOOTPRINT.match(line):
            footprint = int(match.group(1))
    error = []
    if not saved:
        error = [line for line in lines[:time_start] if line.strip()][-error_lines:]
    return QuickstartResult(game, saved, kernels, games_per_second, footprint, elapsed, error)


def parse_speed(output: str) -> dict[int, int]:
    """Simulations per second by batch size from one `search_speed.py` run."""
    rates = {}
    for line in output.splitlines():
        if match := _SPEED_ROW.match(line):
            rates[int(match.group(1))] = int(match.group(2).replace(",", ""))
    return rates


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def quickstart_table(results: list[QuickstartResult]) -> str:
    rows = [
        "| Game | Completed and saved weights? | Metal kernels (`kernels=`) | Elapsed (min:sec) "
        "| Self-play games/s | Peak memory footprint (bytes) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        footprint = "" if r.peak_footprint is None else f"{r.peak_footprint:,}"
        rate = "" if r.games_per_second is None else f"{r.games_per_second:.2f}"
        rows.append(
            f"| {GAME_NAMES.get(r.game, r.game)} | {'yes' if r.saved else 'failed'} "
            f"| {r.kernels or ''} | {format_elapsed(r.elapsed_seconds)} | {rate} | {footprint} |"
        )
    for r in results:
        if r.error:
            rows += [
                "",
                f"Last lines before the {GAME_NAMES.get(r.game, r.game)} run failed:",
                "",
                "```",
            ]
            rows += r.error + ["```"]
    return "\n".join(rows)


def speed_table(runs: dict[str, dict[int, int]], batches=SPEED_BATCHES) -> str:
    """One row per batch size, one column per backend and network."""
    columns = list(runs)
    rows = [
        "| Batch | " + " | ".join(columns) + " |",
        "|---:|" + "---:|" * len(columns),
    ]
    for batch in batches:
        cells = [f"{runs[c][batch]:,}" if batch in runs[c] else "" for c in columns]
        rows.append(f"| {batch:,} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def system_info() -> dict[str, str]:
    import mlx.core as mx

    from mac_alpha_zero.edax import DEFAULT_PATH

    commit = _command_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    if _command_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]):
        commit += " (with local changes)"
    memory = _command_output(["sysctl", "-n", "hw.memsize"])
    if memory.isdigit():
        memory = f"{int(memory) / 2**30:.0f} GB ({int(memory):,} bytes)"
    return {
        "Commit": commit,
        "Chip": _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "Memory": memory,
        "macOS": _command_output(["sw_vers", "-productVersion"]),
        "MLX": importlib.metadata.version("mlx"),
        "Python": platform.python_version(),
        "Metal available": str(mx.metal.is_available()),
        "Edax installed": str(DEFAULT_PATH.exists()),
    }


def system_table(info: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in info.items())


def run_logged(command: list[str], log: Path) -> str:
    """Run `command`, streaming its output to the terminal and to `log`."""
    print("$ " + " ".join(command), flush=True)
    output = []
    with (
        log.open("w") as file,
        subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process,
    ):
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            file.write(line)
            output.append(line)
    return "".join(output)


def awake(command: list[str]) -> list[str]:
    caffeinate = shutil.which("caffeinate")
    return [caffeinate, "-i", *command] if caffeinate else command


def run_quickstart(game: str, minutes: float, report_dir: Path) -> QuickstartResult:
    command = awake(
        [
            "/usr/bin/time",
            "-l",
            sys.executable,
            "-u",
            str(ROOT / "examples" / f"train_{game}.py"),
            "--minutes",
            f"{minutes:g}",
            "--checkpoint",
            str(report_dir / f"{game}.safetensors"),
        ]
    )
    return parse_quickstart(game, run_logged(command, report_dir / f"{game}.txt"))


def run_speed(backend: str, network: str | None, report_dir: Path) -> dict[int, int]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "benchmarks" / "search_speed.py"),
        "--backend",
        backend,
        "--batch",
        *map(str, SPEED_BATCHES),
    ]
    if network is not None:
        command += ["--network", network]
    log = report_dir / f"speed-{backend}-{network or 'none'}.txt"
    return parse_speed(run_logged(awake(command), log))


SPEED_LABELS = {
    "metal": "Metal kernels (GPU)",
    "mlx": "Pure MLX (GPU)",
    "mctx": "mctx (JAX, CPU)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--quickstart",
        nargs="*",
        choices=GAMES,
        default=list(GAMES),
        help="quickstarts to run (default: both; pass the flag alone to skip them)",
    )
    parser.add_argument("--minutes", type=float, default=10, help="training minutes per game")
    parser.add_argument(
        "--speed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="time search with benchmarks/search_speed.py (default: on)",
    )
    parser.add_argument("--mctx", action="store_true", help="also time mctx (needs --group bench)")
    parser.add_argument("--report-dir", type=Path, help="default: a new temporary directory")
    args = parser.parse_args()
    if args.mctx and importlib.util.find_spec("mctx") is None:
        parser.error("mctx is not installed; run with `uv run --group bench python ...`")

    report_dir = args.report_dir or Path(tempfile.mkdtemp(prefix="mac-alpha-zero-report-"))
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"report directory: {report_dir}", flush=True)

    sections = ["### System", "", system_table(system_info())]
    if args.quickstart:
        results = [run_quickstart(game, args.minutes, report_dir) for game in args.quickstart]
        sections += [
            "",
            f"### Quickstarts ({args.minutes:g}-minute training budget, for issue #10)",
            "",
            quickstart_table(results),
        ]
    if args.speed:
        runs = {}
        for network in SPEED_NETWORKS:
            for backend in ("metal", "mlx") + (("mctx",) if args.mctx else ()):
                label = f"{SPEED_LABELS[backend]}, network {network or 'none'}"
                runs[label] = run_speed(backend, network, report_dir)
        sections += [
            "",
            "### Search speed (Connect4, 64 simulations, simulations/s, for issue #31)",
            "",
            speed_table(runs),
        ]

    report = "\n".join(sections) + "\n"
    (report_dir / "report.md").write_text(report)
    print("\n" + report)
    print(f"saved {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
