import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "snorlax_pipeline",
    Path(__file__).resolve().parents[1] / "metamon" / "rl" / "snorlax_pipeline.py",
)
pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pipeline)


def _touch(root: Path, *names: str) -> None:
    for name in names:
        (root / name).write_bytes(b"{}")


@pytest.fixture
def dirs(tmp_path):
    replay = tmp_path / "online_buffer" / "gen1ou"
    logs = tmp_path / "logs"
    replay.mkdir(parents=True)
    logs.mkdir()
    return logs, replay


def test_replay_names_filters_suffixes(dirs):
    _, replay = dirs
    _touch(replay, "a.json", "b.json.lz4", "c.json.lz4.tmp", "notes.txt")
    assert pipeline.replay_names(replay) == {"a.json", "b.json.lz4"}
    assert pipeline.replay_names(replay / "missing") == set()


def test_completed_round_is_kept_and_fresh_is_exact(dirs):
    logs, replay = dirs
    _touch(replay, "old1.json.lz4", "old2.json.lz4")
    ledger = pipeline.RoundLedger(logs, replay)
    baseline = ledger.begin(7)
    _touch(replay, "new1.json.lz4", "new2.json.lz4")
    assert ledger.fresh(7, baseline) == {"new1.json.lz4", "new2.json.lz4"}
    assert ledger.fresh(7) == {"new1.json.lz4", "new2.json.lz4"}
    ledger.mark_complete(7, 2)
    assert ledger.is_complete(7)
    assert not ledger.baseline_path(7).exists()
    assert ledger.purge_incomplete() == {}
    assert len(pipeline.replay_names(replay)) == 4


def test_interrupted_round_partials_are_purged(dirs):
    logs, replay = dirs
    _touch(replay, "kept.json.lz4")
    ledger = pipeline.RoundLedger(logs, replay)
    ledger.begin(3)
    _touch(replay, "partial1.json.lz4", "partial2.json")
    assert ledger.purge_incomplete() == {3: 2}
    assert pipeline.replay_names(replay) == {"kept.json.lz4"}
    assert not ledger.baseline_path(3).exists()
    assert not ledger.is_complete(3)


def test_begin_clears_stale_completion(dirs):
    logs, replay = dirs
    ledger = pipeline.RoundLedger(logs, replay)
    ledger.begin(4)
    ledger.mark_complete(4, 0)
    ledger.begin(4)
    assert not ledger.is_complete(4)


def test_forget_through_keeps_future_rounds(dirs):
    logs, replay = dirs
    ledger = pipeline.RoundLedger(logs, replay)
    for round_idx in (1, 2, 3):
        ledger.begin(round_idx)
        ledger.mark_complete(round_idx, 0)
    ledger.begin(4)
    ledger.forget_through(2)
    assert not ledger.is_complete(1) and not ledger.is_complete(2)
    assert ledger.is_complete(3)
    assert ledger.baseline_path(4).exists()


@pytest.mark.parametrize(
    "text, expected",
    [
        (None, None),
        ("", None),
        ("3", {3}),
        ("0-3,8,10-11", {0, 1, 2, 3, 8, 10, 11}),
        (" 2 - 4 , 6 ", {2, 3, 4, 6}),
    ],
)
def test_parse_cpu_list(text, expected):
    assert pipeline.parse_cpu_list(text) == expected


@pytest.mark.parametrize("bad", ["4-2", "-1", "a"])
def test_parse_cpu_list_rejects(bad):
    with pytest.raises(ValueError):
        pipeline.parse_cpu_list(bad)


def test_format_round_trips():
    cpus = {0, 1, 2, 3, 8, 10, 11, 15}
    text = pipeline.format_cpu_list(cpus)
    assert text == "0-3,8,10-11,15"
    assert pipeline.parse_cpu_list(text) == cpus


def test_collector_cpu_set():
    available = set(range(16))
    assert pipeline.collector_cpu_set(None, None, available) is None
    assert pipeline.collector_cpu_set(set(range(4)), None, available) == set(range(4, 16))
    assert pipeline.collector_cpu_set(set(range(4)), {9, 10}, available) == {9, 10}
    with pytest.raises(ValueError):
        pipeline.collector_cpu_set(available, None, available)


def test_affinity_prefix():
    assert pipeline.affinity_command_prefix(None) == []
    prefix = pipeline.affinity_command_prefix({0, 1, 2})
    assert prefix[1:] == ["-c", "0-2"] and prefix[0].endswith("taskset")
