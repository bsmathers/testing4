#!/usr/bin/env python3
"""Convert raw Showdown protocol logs into Metamon replay trajectories.

The input is a flat directory of ``gen1ou-<id>.log`` files containing the
Showdown battle protocol. Metamon's replay parser normally consumes downloaded
raw-replay JSON objects whose ``log`` field contains that same protocol. This
adapter extracts the missing envelope metadata and then delegates battle-state
and action reconstruction to Metamon's existing forward/backward parser.

Only perspectives belonging to the explicitly selected strong-model players
are saved. A battle between two selected players produces two trajectories; a
battle with one selected player produces one; all other battles are recorded as
completed without running expensive state reconstruction. Progress is recorded
as human-readable TSV so a stopped conversion can safely resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable


_PARSER = None
_OUTPUT_DIR: Path | None = None

TARGET_PLAYERS = frozenset(
    {
        "Tauros-v1",
        "Tauros-v2",
        "SG1Online-v1-5",
        "SmallG1Online-v1",
        "SmallG1Online-v2",
        "SmallG1Online-v3",
        "Exeggcute-v0",
    }
)


def _workspace_defaults() -> tuple[Path, Path]:
    workspace = Path(__file__).resolve().parents[2]
    return (
        workspace / "strong_model_replays",
        workspace / "strong_model_replays_parsed",
    )


def _clean_field(value: object) -> str:
    """Keep the TSV manifest one physical line per source replay."""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _extract_envelope(path: Path, raw_log: str) -> tuple[list[str], int]:
    players: dict[str, str] = {}
    upload_time: int | None = None
    generation: str | None = None
    tier: str | None = None
    winner: str | None = None

    for line in raw_log.splitlines():
        if not line.startswith("|"):
            continue
        fields = line.split("|")[1:]
        if not fields:
            continue
        message = fields[0]
        args = fields[1:]
        if message == "player" and len(args) >= 2 and args[0] in {"p1", "p2"}:
            if args[1]:
                players[args[0]] = args[1]
        elif message == "t:" and args and upload_time is None:
            upload_time = int(args[0])
        elif message == "gen" and args:
            generation = args[0]
        elif message == "tier" and args:
            tier = args[0]
        elif message == "win" and args:
            winner = args[0]

    missing = [slot for slot in ("p1", "p2") if slot not in players]
    if missing:
        raise ValueError(f"missing player metadata for {', '.join(missing)}")
    if upload_time is None:
        raise ValueError("missing |t:| timestamp")
    if generation != "1":
        raise ValueError(f"expected |gen|1, found {generation!r}")
    if tier is None or "gen 1 ou" not in tier.lower():
        raise ValueError(f"expected a Gen 1 OU tier, found {tier!r}")
    if winner is None:
        raise ValueError("missing terminal |win| message")
    if winner not in players.values():
        raise ValueError(f"winner {winner!r} is not one of {list(players.values())!r}")
    if not path.stem.startswith("gen1ou-"):
        raise ValueError(f"unexpected replay filename: {path.name}")
    return [players["p1"], players["p2"]], upload_time


def _init_worker(output_dir: str, cache_dir: str | None) -> None:
    global _PARSER, _OUTPUT_DIR
    if cache_dir:
        os.environ["METAMON_CACHE_DIR"] = cache_dir

    # Delay Metamon imports until after METAMON_CACHE_DIR is fixed. Several
    # modules resolve their usage-stat paths at import time.
    from metamon.backend.replay_parser.parse_replays import ReplayParser
    from metamon.backend.team_prediction.predictor import NaiveUsagePredictor

    _OUTPUT_DIR = Path(output_dir)
    _PARSER = ReplayParser(
        replay_output_dir=output_dir,
        verbose=False,
        sleep_on_handled_exception=0.0,
        team_predictor=NaiveUsagePredictor(),
    )


def _validate_output(path: Path) -> None:
    import lz4.frame

    with lz4.frame.open(path, "rb") as stream:
        payload = json.loads(stream.read().decode("utf-8"))
    states = payload.get("states")
    actions = payload.get("actions")
    if not isinstance(states, list) or not isinstance(actions, list):
        raise ValueError(f"{path.name}: missing states/actions lists")
    if len(states) < 2 or len(actions) != len(states):
        raise ValueError(
            f"{path.name}: malformed trajectory lengths "
            f"states={len(states)}, actions={len(actions)}"
        )


def _output_name(replay, player: str, opponent: str, played: datetime) -> str:
    won = "WIN" if replay.winner else "LOSS"
    return (
        f"{replay.gameid}_{replay.rating}_{player}_vs_{opponent}_"
        f"{played.strftime('%m-%d-%Y')}_{won}.json.lz4"
    )


def _convert_one(source_text: str) -> tuple[str, str, str]:
    """Return ``(source, status, detail)`` for the parent manifest writer."""
    source = Path(source_text)
    try:
        raw_log = source.read_text(encoding="utf-8")
        players, upload_time = _extract_envelope(source, raw_log)
        selected_slots = [index for index, name in enumerate(players) if name in TARGET_PLAYERS]
        if not selected_slots:
            return source.name, "no-target-player", ""

        # Team completion can involve sampling from usage statistics. Seed from
        # the immutable source name so output is stable across worker counts and
        # interrupted/resumed conversions.
        seed = int.from_bytes(
            hashlib.sha256(source.name.encode("utf-8")).digest()[:8], "big"
        )
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except ImportError:
            pass

        from metamon.backend.replay_parser import backward, forward
        from metamon.backend.replay_parser.exceptions import (
            BackwardException,
            ForwardException,
        )

        played = datetime.fromtimestamp(upload_time)
        replay = forward.ParsedReplay(
            gameid=source.stem,
            format="gen1ou",
            time_played=played,
        )
        protocol = _PARSER.clean_log({"log": raw_log})
        try:
            replay = forward.forward_fill(replay, protocol, verbose=False)
            p1_replay, p2_replay = backward.backward_fill(
                replay,
                team_predictor=_PARSER.team_predictor,
            )
            outputs = []
            perspectives = (
                (p1_replay, players[0], players[1]),
                (p2_replay, players[1], players[0]),
            )
            for slot in selected_slots:
                pov, player, opponent = perspectives[slot]
                _PARSER.save_to_disk(
                    pov,
                    time_played=played,
                    player_username=player,
                    opponenent_username=opponent,
                )
                output = _OUTPUT_DIR / _output_name(pov, player, opponent, played)
                _validate_output(output)
                outputs.append(output.name)
        except (ForwardException, BackwardException) as exc:
            return source.name, "parser-rejected", f"{type(exc).__name__}: {exc}"

        return source.name, "parsed", ",".join(outputs)
    except (OSError, UnicodeError, ValueError) as exc:
        return source.name, "invalid-input", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # preserve the batch while surfacing code defects
        detail = f"{type(exc).__name__}: {exc}; {traceback.format_exc(limit=3)}"
        return source.name, "unexpected-error", _clean_field(detail)


def _completed_sources(manifest: Path, output_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not manifest.exists():
        return completed
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) >= 2 and fields[1] == "no-target-player":
                completed.add(fields[0])
            elif len(fields) >= 3 and fields[1] == "parsed":
                outputs = [name for name in fields[2].split(",") if name]
                if 1 <= len(outputs) <= 2 and all(
                    (output_dir / name).is_file() for name in outputs
                ):
                    completed.add(fields[0])
    return completed


def _iter_results(
    sources: list[Path], workers: int, output_dir: Path, cache_dir: Path | None
) -> Iterable[tuple[str, str, str]]:
    if workers == 1:
        _init_worker(str(output_dir), str(cache_dir) if cache_dir else None)
        for source in sources:
            yield _convert_one(str(source))
        return

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(output_dir), str(cache_dir) if cache_dir else None),
    ) as executor:
        yield from executor.map(
            _convert_one,
            (str(source) for source in sources),
            chunksize=8,
        )


def main() -> int:
    default_input, default_output = _workspace_defaults()
    parser = argparse.ArgumentParser(
        description="Convert raw Gen1OU Showdown logs into Metamon .json.lz4 trajectories."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="GOOD_REPLAYS_DIR root; converted files go in its gen1ou/ child.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Metamon cache containing Gen1 usage statistics. Defaults to METAMON_CACHE_DIR.",
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    output_dir = output_root / "gen1ou"
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    if input_dir == output_root or input_dir == output_dir:
        parser.error("output must be separate from the raw input directory")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    sources = sorted(input_dir.glob("gen1ou-*.log"))
    if not sources:
        parser.error(f"no gen1ou-*.log files found in {input_dir}")
    manifest = output_root / "conversion-progress.tsv"
    completed = _completed_sources(manifest, output_dir)
    sources = [source for source in sources if source.name not in completed]
    if args.limit is not None:
        sources = sources[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Raw input: {input_dir}")
    print(f"Metamon output: {output_dir}")
    print(f"Already completed: {len(completed):,}")
    print(f"Queued this run: {len(sources):,}")
    if not sources:
        print("Nothing to convert.")
        return 0

    counts: dict[str, int] = {}
    trajectory_count = 0
    with manifest.open("a", encoding="utf-8", buffering=1) as progress:
        for index, (source, status, detail) in enumerate(
            _iter_results(sources, args.workers, output_dir, args.cache_dir), start=1
        ):
            counts[status] = counts.get(status, 0) + 1
            if status == "parsed":
                trajectory_count += len([name for name in detail.split(",") if name])
            progress.write(
                f"{_clean_field(source)}\t{_clean_field(status)}\t"
                f"{_clean_field(detail)}\n"
            )
            if index == 1 or index % 1000 == 0 or index == len(sources):
                summary = ", ".join(
                    f"{name}={count:,}" for name, count in sorted(counts.items())
                )
                print(f"Progress {index:,}/{len(sources):,}: {summary}", flush=True)

    parsed = counts.get("parsed", 0)
    print(
        f"Converted {parsed:,} battles into {trajectory_count:,} "
        "allowlisted, validated trajectories."
    )
    unexpected = counts.get("unexpected-error", 0)
    if unexpected:
        print(
            f"Encountered {unexpected:,} unexpected parser errors; inspect {manifest}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
