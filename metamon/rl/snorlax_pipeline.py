"""Bookkeeping for overlapping Snorlax collection with learning.

The overlapped learner runs collection round ``e + 1`` in background processes
while it trains epoch ``e``. Only one round is ever in flight. This module holds
the parts of that protocol that do not need torch or AMAGO, so they can be unit
tested directly:

* a durable per-round ledger (a baseline of replay names taken at launch and a
  completion marker), which lets a restarted learner delete the partial files
  of an interrupted round and never mistake a half-finished round for data;
* CPU-set parsing and affinity helpers that keep learner data loading and the
  Showdown simulators on separate cores.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

REPLAY_SUFFIXES = (".json", ".json.lz4")
_ROUND_FILE = re.compile(r"^\.collection-round-(\d+)-(baseline\.txt|complete)$")


def replay_names(root: Path) -> set[str]:
    """Names of replay files directly inside ``root`` (empty if missing)."""
    try:
        with os.scandir(root) as entries:
            return {
                entry.name
                for entry in entries
                if entry.is_file() and entry.name.endswith(REPLAY_SUFFIXES)
            }
    except FileNotFoundError:
        return set()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class RoundLedger:
    """Durable state of collection rounds in ``log_dir`` for ``replay_dir``.

    A round is *complete* only once its marker exists. A baseline file without
    a marker means the round was interrupted; every replay that appeared after
    the baseline belongs to it and is removed by :meth:`purge_incomplete`.
    """

    def __init__(self, log_dir: Path, replay_dir: Path):
        self.log_dir = Path(log_dir)
        self.replay_dir = Path(replay_dir)

    def baseline_path(self, round_idx: int) -> Path:
        return self.log_dir / f".collection-round-{int(round_idx)}-baseline.txt"

    def complete_path(self, round_idx: int) -> Path:
        return self.log_dir / f".collection-round-{int(round_idx)}-complete"

    def begin(self, round_idx: int, names: Optional[Iterable[str]] = None) -> set[str]:
        """Record the pre-round replay set; returns it."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        baseline = set(replay_names(self.replay_dir) if names is None else names)
        self.complete_path(round_idx).unlink(missing_ok=True)
        _atomic_write(
            self.baseline_path(round_idx),
            "".join(f"{name}\n" for name in sorted(baseline)),
        )
        return baseline

    def baseline(self, round_idx: int) -> set[str]:
        text = self.baseline_path(round_idx).read_text(encoding="utf-8")
        return {line for line in text.splitlines() if line}

    def fresh(self, round_idx: int, baseline: Optional[set[str]] = None) -> set[str]:
        base = self.baseline(round_idx) if baseline is None else baseline
        return replay_names(self.replay_dir) - base

    def mark_complete(self, round_idx: int, games: int) -> None:
        _atomic_write(
            self.complete_path(round_idx),
            f"collection round {int(round_idx)} complete: {int(games)} games\n",
        )
        self.baseline_path(round_idx).unlink(missing_ok=True)

    def is_complete(self, round_idx: int) -> bool:
        return self.complete_path(round_idx).is_file()

    def _rounds(self, kind: str) -> list[int]:
        rounds = []
        try:
            names = os.listdir(self.log_dir)
        except FileNotFoundError:
            return rounds
        for name in names:
            match = _ROUND_FILE.match(name)
            if match and match.group(2).startswith(kind):
                rounds.append(int(match.group(1)))
        return sorted(rounds)

    def purge_incomplete(self) -> dict[int, int]:
        """Delete partial replays of every interrupted round.

        Returns ``{round: files_removed}``. Rounds with a completion marker are
        left untouched.
        """
        removed: dict[int, int] = {}
        for round_idx in self._rounds("baseline"):
            if self.is_complete(round_idx):
                self.baseline_path(round_idx).unlink(missing_ok=True)
                continue
            count = 0
            for name in sorted(self.fresh(round_idx)):
                try:
                    (self.replay_dir / name).unlink()
                    count += 1
                except FileNotFoundError:
                    pass
            self.baseline_path(round_idx).unlink(missing_ok=True)
            removed[round_idx] = count
        return removed

    def forget_through(self, round_idx: int) -> None:
        """Drop ledger files of rounds ``<= round_idx`` (already consumed)."""
        for kind in ("baseline", "complete"):
            for existing in self._rounds(kind):
                if existing <= round_idx:
                    path = (
                        self.baseline_path(existing)
                        if kind == "baseline"
                        else self.complete_path(existing)
                    )
                    path.unlink(missing_ok=True)


def parse_cpu_list(text: Optional[str]) -> Optional[set[int]]:
    """Parse ``"0-3,8,10-11"``; ``None``/blank means "no restriction"."""
    if text is None or not str(text).strip():
        return None
    cpus: set[int] = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            low_i, high_i = int(low), int(high)
            if high_i < low_i:
                raise ValueError(f"invalid CPU range {part!r}")
            cpus.update(range(low_i, high_i + 1))
        else:
            cpus.add(int(part))
    if not cpus or min(cpus) < 0:
        raise ValueError(f"invalid CPU list {text!r}")
    return cpus


def container_cpus() -> set[int]:
    """CPUs this container may use (cgroup cpuset, else all online CPUs)."""
    for path in (
        Path("/sys/fs/cgroup/cpuset.cpus.effective"),
        Path("/sys/fs/cgroup/cpuset/cpuset.effective_cpus"),
    ):
        try:
            parsed = parse_cpu_list(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if parsed:
            return parsed
    return set(range(os.cpu_count() or 1))


def collector_cpu_set(
    learner: Optional[set[int]], collector: Optional[set[int]], available: set[int]
) -> Optional[set[int]]:
    """CPUs for collector/validation children.

    An explicit collector set wins. Otherwise, when the learner is pinned, the
    children get every other available CPU (children inherit the learner's
    mask by default, which would squeeze the simulators onto its cores).
    """
    if collector is not None:
        return collector & available or collector
    if learner is None:
        return None
    rest = available - learner
    if not rest:
        raise ValueError(
            "SNORLAX_LEARNER_CPUS leaves no CPUs for collectors; set "
            "SNORLAX_COLLECTOR_CPUS explicitly"
        )
    return rest


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Inverse of :func:`parse_cpu_list` (compact ranges)."""
    values = sorted(set(int(c) for c in cpus))
    parts: list[str] = []
    start = prev = None
    for cpu in values:
        if start is None:
            start = prev = cpu
        elif cpu == prev + 1:
            prev = cpu
        else:
            parts.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = cpu
    if start is not None:
        parts.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ",".join(parts)


def affinity_command_prefix(cpus: Optional[set[int]]) -> list[str]:
    """``taskset`` prefix pinning a child command (and its descendants).

    A ``taskset`` prefix is used instead of ``preexec_fn`` because the learner
    is multi-threaded (CUDA, pin-memory, loader threads), where ``preexec_fn``
    is documented as unsafe.
    """
    if cpus is None:
        return []
    import shutil

    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError(
            "CPU pinning was requested but `taskset` (util-linux) is not installed"
        )
    return [taskset, "-c", format_cpu_list(cpus)]
