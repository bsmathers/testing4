import pytest

online = pytest.importorskip("metamon.rl.ataraxos_online")
groups_for = online.AtaraxosExperiment._opponent_groups_per_collector


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SNORLAX_OPPONENT_GROUPS_PER_COLLECTOR", raising=False)
    monkeypatch.delenv("SNORLAX_MIN_LANES_PER_OPPONENT", raising=False)


@pytest.mark.parametrize(
    "processes, lanes, rows, expected",
    [
        (8, 64, 12, 1),  # 8x64 presets: single-opponent path, no extra forwards
        (6, 64, 12, 1),
        (4, 64, 12, 2),  # mid-cpu preset: 8 rows per epoch
        (1, 96, 12, 3),  # fallback: 3 groups of 32 lanes
        (1, 32, 12, 1),  # too few lanes to split at 32 per group
    ],
)
def test_default_groups(processes, lanes, rows, expected):
    assert groups_for(processes=processes, lanes=lanes, pool_rows=rows) == expected


def test_override(monkeypatch):
    monkeypatch.setenv("SNORLAX_OPPONENT_GROUPS_PER_COLLECTOR", "4")
    assert groups_for(processes=8, lanes=64, pool_rows=12) == 4
    monkeypatch.setenv("SNORLAX_OPPONENT_GROUPS_PER_COLLECTOR", "1000")
    assert groups_for(processes=8, lanes=64, pool_rows=12) == 64
