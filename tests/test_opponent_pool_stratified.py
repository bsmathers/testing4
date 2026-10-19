from collections import Counter

from metamon.rl.evaluate.common import PolicySpec, stratified_choices
from metamon.rl.evaluate.opponent_pool import OpponentPoolConfig


def _row(model_name, checkpoints):
    return {
        "model_name": model_name,
        "checkpoints": checkpoints,
        "temperatures": [0.75, 1.0, 1.25, 1.5],
        "team_set": ["hl_05_26", "snorlax_v1_expert"],
        "battle_backend": "metamon",
        "num_agents": 1,
    }


def _snorlax_like_pool():
    rows = []
    for i in range(3):
        rows.append((f"TaurosV0-{i}", _row("TaurosV0", list(range(50, 63, 2)))))
    for i in range(3):
        rows.append((f"Kakuna-{i}", _row("Kakuna", list(range(24, 35, 2)))))
    rows.append(("Epoch475Original", _row("Epoch475Frozen", [475])))
    for i, ckpt in enumerate([0, 1, 4, 16, 64]):
        rows.append((f"SnorlaxHistory-{i}", _row("SnorlaxV1", [ckpt])))
    return OpponentPoolConfig(rows, battle_format="gen1ou")


def test_stratified_opponents_balance_snorlax_rows_over_three_phases():
    pool = _snorlax_like_pool()
    selected = []
    for phase in range(3):
        specs = pool.stratified_opponents(8, phase=phase, seed=73_001)
        assert len(specs) == 8
        assert len({spec.name for spec in specs}) == 8
        selected.extend(spec.name for spec in specs)

    # 8 of 12 rows per phase. Rotating the systematic-sampling offset by one
    # row makes every row appear exactly twice over a three-phase period.
    assert set(Counter(selected).values()) == {2}


def test_stratified_opponents_balance_temperature_and_team_each_phase():
    pool = _snorlax_like_pool()
    for phase in range(6):
        specs = pool.stratified_opponents(8, phase=phase, seed=73_001)
        assert Counter(spec.temperature for spec in specs) == {
            0.75: 2,
            1.0: 2,
            1.25: 2,
            1.5: 2,
        }
        assert Counter(spec.team_set for spec in specs) == {
            "hl_05_26": 4,
            "snorlax_v1_expert": 4,
        }


def test_stratified_opponents_are_reproducible():
    pool = _snorlax_like_pool()
    a = pool.stratified_opponents(8, phase=5, seed=11)
    b = pool.stratified_opponents(8, phase=5, seed=11)
    assert a == b


def test_single_spec_pool_is_exact():
    spec = PolicySpec(
        name="fixed",
        model_name="TaurosV0",
        checkpoint=62,
        temperature=1.25,
        team_set="hl_05_26",
        battle_backend="metamon",
    )
    pool = OpponentPoolConfig.from_policy_spec(spec, battle_format="gen1ou")
    assert [pool.sample_opponent() for _ in range(5)] == [spec] * 5


def test_stratified_choices_balance_equal_weight_team_sets():
    value = {"weighted": {"hl_05_26": 1, "snorlax_v1_expert": 1}}
    for phase in range(8):
        chosen = stratified_choices(value, 8, phase=phase)
        assert Counter(chosen) == {
            "hl_05_26": 4,
            "snorlax_v1_expert": 4,
        }
        assert all(chosen[i] != chosen[i + 1] for i in range(len(chosen) - 1))


def test_stratified_choices_cycle_plain_lists():
    assert stratified_choices(["a", "b"], 6, phase=0) == [
        "a", "b", "a", "b", "a", "b"
    ]
    assert stratified_choices(["a", "b"], 6, phase=1) == [
        "b", "a", "b", "a", "b", "a"
    ]

def test_validation_style_schedule_covers_each_anchor_twice():
    rows = [
        ("TaurosV0", _row("TaurosV0", [62])),
        ("Kakuna", _row("Kakuna", [34])),
        ("Epoch475", _row("Epoch475Frozen", [475])),
    ]
    pool = OpponentPoolConfig(rows, battle_format="gen1ou")
    specs = pool.stratified_opponents(6, phase=0, seed=91_337)
    assert Counter(spec.model_name for spec in specs) == {
        "TaurosV0": 2,
        "Kakuna": 2,
        "Epoch475Frozen": 2,
    }
    assert Counter(spec.team_set for spec in specs) == {
        "hl_05_26": 3,
        "snorlax_v1_expert": 3,
    }