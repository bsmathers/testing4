"""Drive the real overlapped learner loop with stub collector processes.

Collector children are tiny Python processes that write minimal schema-v3
replays; the learner's training step, dataset, and checkpointing are stubbed.
This checks the scheduling protocol: which round is in flight during each
epoch, that ingest happens only between epochs, and that an interrupted round
is purged and recollected on resume while completed rounds are reused.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

online = pytest.importorskip("metamon.rl.ataraxos_online")
from metamon.rl.snorlax_pipeline import replay_names  # noqa: E402

WRITER = textwrap.dedent(
    """
    import sys, time, lz4.frame, orjson
    root, prefix, count, delay = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    payload = {
        "ataraxos_collector_version": 3,
        "states": [{}, {"battle_won": True}],
        "actions": [0, -1],
        "behavior_values": [0.0],
        "behavior_log_probs": [[0.0]],
        "behavior_legal_masks": [[True]],
    }
    for i in range(count):
        time.sleep(delay)
        import os
        path = f"{root}/{prefix}-{os.getpid()}-{i}.json.lz4"
        with lz4.frame.open(path + ".tmp", "wb") as f:
            f.write(orjson.dumps(payload))
        os.rename(path + ".tmp", path)
    """
)


class _Recorder:
    def __init__(self):
        self.events = []


class _Dataset:
    def __init__(self, recorder):
        self.recorder = recorder
        self.ready_for_training = True

    def on_end_of_collection(self, experiment):
        self.recorder.events.append(("ingest", experiment.epoch))
        return {}


class _Null:
    def eval(self):
        pass

    def train(self):
        pass

    def wait_for_everyone(self):
        pass


def _make_experiment(tmp_path, recorder, *, epochs, crash_at=None, start_epoch=0):
    exp = object.__new__(online.AtaraxosExperiment)
    exp.epoch = start_epoch
    exp.epochs = epochs
    exp.policy_aclr = _Null()
    exp.accelerator = _Null()
    exp.dataset = _Dataset(recorder)
    exp.train_dloader = []
    exp._resident_loader_workers_started = True
    exp.always_save_latest = True
    exp.ckpt_interval = 1
    exp.ckpt_dir = str(tmp_path / "ckpts")
    exp.run_name = "test"
    exp.train_batches_per_epoch = 10
    exp._k = 0

    exp.log = lambda *a, **k: None
    exp._optimizer_update_step = lambda: exp._k
    exp._external_validation_phase = lambda epoch: None
    exp.write_latest_policy = lambda: None

    def write_behavior(path=None):
        path = path or os.path.join(exp.ckpt_dir, "latest", "behavior_policy.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(f"y@{exp._k}")
        return path

    exp._write_behavior_policy = write_behavior

    def save_checkpoint():
        recorder.events.append(("checkpoint", exp.epoch))

    exp.save_checkpoint = save_checkpoint

    def launch(*, epoch, processes, game_quotas, buffer_dir, log_dir, **_):
        children = []
        for idx, quota in enumerate(game_quotas):
            stream = open(log_dir / f"collector-epoch-{epoch}-{idx}.log", "a")
            child = subprocess.Popen(
                [sys.executable, "-c", WRITER, str(buffer_dir / "gen1ou"),
                 f"r{epoch}p{idx}", str(quota), "0.05"],
                stdout=stream, stderr=subprocess.STDOUT,
            )
            children.append((child, stream))
        recorder.events.append(("launch", epoch))
        return {"epoch": epoch, "children": children, "target": sum(game_quotas),
                "launched": time.monotonic()}

    exp._launch_collector_children = launch

    def train(epoch, progress_hook=None):
        pending = exp._pending_round
        recorder.events.append(("train", epoch, None if pending is None else pending["round"]))
        if progress_hook is not None:
            progress_hook()
        if crash_at is not None and epoch == crash_at:
            time.sleep(0.25)  # let the background round write some files
            raise KeyboardInterrupt("simulated preemption")
        time.sleep(0.1)
        exp._k += 10

    exp._train_one_epoch = train
    return exp


@pytest.fixture
def env(tmp_path, monkeypatch):
    buffer_dir = tmp_path / "online_buffer"
    (buffer_dir / "gen1ou").mkdir(parents=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    for name, value in {
        "SNORLAX_LOG_DIR": str(log_dir),
        "SNORLAX_BUFFER_DIR": str(buffer_dir),
        "ONLINE_FIFO_SIZE": "4",
        "GAMES_PER_LEARNER_EPOCH": "4",
        "COLLECTOR_PROCESSES": "2",
        "LANES": "4",
        "SNORLAX_RUN_CONFIG": "unused.yaml",
        "METAMON_SAVE_DIR": str(tmp_path),
        "SNORLAX_COLLECT_POLL_SECONDS": "0.5",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SNORLAX_LEARNER_CPUS", raising=False)
    # Seed the FIFO as the shell prefill would.
    subprocess.check_call(
        [sys.executable, "-c", WRITER, str(buffer_dir / "gen1ou"), "prefill", "4", "0"]
    )
    online._FIFO_SUMMARY_CACHE.clear()
    return tmp_path, buffer_dir / "gen1ou", log_dir


def test_fresh_run_overlaps_next_round_with_training(env):
    tmp_path, replay_dir, log_dir = env
    recorder = _Recorder()
    exp = _make_experiment(tmp_path, recorder, epochs=4)
    exp._learn_overlapped()

    trains = [e for e in recorder.events if e[0] == "train"]
    # Epoch e trains while round e + 1 is in flight; the last epoch has none.
    assert trains == [("train", 0, 1), ("train", 1, 2), ("train", 2, 3), ("train", 3, None)]
    launches = [e[1] for e in recorder.events if e[0] == "launch"]
    assert launches == [0, 1, 2, 3]  # round 0 synchronous, then one ahead
    # Ingest strictly precedes each epoch's training.
    order = [(e[0], e[1]) for e in recorder.events if e[0] in ("ingest", "train")]
    assert order == [("ingest", 0), ("train", 0), ("ingest", 1), ("train", 1),
                     ("ingest", 2), ("train", 2), ("ingest", 3), ("train", 3)]
    # 4 prefill + 4 games per round x 4 rounds, nothing partial.
    assert len(replay_names(replay_dir)) == 20
    # Consumed ledger entries are cleaned; every round left an outcome report.
    assert not list(log_dir.glob(".collection-round-*"))
    for round_idx in range(4):
        assert (log_dir / f"collection-epoch-{round_idx}-outcomes.txt").is_file()
    assert exp._pending_round is None


def test_resume_purges_interrupted_round_and_reuses_completed(env):
    tmp_path, replay_dir, log_dir = env
    recorder = _Recorder()
    exp = _make_experiment(tmp_path, recorder, epochs=5, crash_at=2)
    with pytest.raises(KeyboardInterrupt):
        exp._learn_overlapped()
    # Crash during epoch 2: checkpoints 0 and 1 exist; round 2 is complete
    # (consumed by the crashed epoch), round 3 was in flight and is unmarked.
    assert [e[1] for e in recorder.events if e[0] == "checkpoint"] == [0, 1]
    assert (log_dir / ".collection-round-2-complete").is_file()
    assert (log_dir / ".collection-round-3-baseline.txt").is_file()
    before = len(replay_names(replay_dir))
    partial = {n for n in replay_names(replay_dir) if n.startswith("r3")}
    assert partial, "crash should leave unvalidated round-3 files behind"

    # Resume from the epoch-1 checkpoint: next epoch is 2.
    recorder2 = _Recorder()
    exp2 = _make_experiment(tmp_path, recorder2, epochs=5, start_epoch=2)
    exp2._learn_overlapped()
    launches = [e[1] for e in recorder2.events if e[0] == "launch"]
    assert launches == [3, 4]  # round 2 reused, round 3 recollected from scratch
    trains = [e for e in recorder2.events if e[0] == "train"]
    assert trains == [("train", 2, 3), ("train", 3, 4), ("train", 4, None)]
    files = replay_names(replay_dir)
    # prefill + rounds 0..4 at 4 games each; round 3's partial files are gone.
    assert len(files) == 4 + 5 * 4
    assert before >= 4 + 3 * 4
    assert sum(name.startswith("r3") for name in files) == 4
    assert not (partial & files), "interrupted round-3 files must be purged"
