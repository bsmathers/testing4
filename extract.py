#!/usr/bin/env python3

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser import forward, backward
from metamon.backend.team_prediction.predictor import ReplayPredictor


REPLAY_RE = re.compile(
    r"https?://replay\.pokemonshowdown\.com/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def load_replay_ids(path: Path) -> list[str]:
    """Extract and deduplicate Pokemon Showdown replay IDs."""
    text = path.read_text()

    replay_ids = REPLAY_RE.findall(text)

    seen = set()
    result = []

    for replay_id in replay_ids:
        replay_id = replay_id.rstrip("/")

        if replay_id not in seen:
            seen.add(replay_id)
            result.append(replay_id)

    return result


def download_replay_json(
    replay_id: str,
    output_path: Path,
    retries: int = 5,
):
    """Download replay JSON, or reuse the cached copy."""

    if output_path.exists():
        try:
            with output_path.open() as f:
                return json.load(f)
        except Exception:
            print(f"  Cached JSON invalid; redownloading.")

    url = (
        f"https://replay.pokemonshowdown.com/"
        f"{replay_id}.json"
    )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "metamon-team-extractor/1.0"
            )
        },
    )

    last_error = None

    for attempt in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()

            data = json.loads(raw.decode("utf-8"))

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            output_path.write_bytes(raw)

            return data

        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as e:
            last_error = e

            if attempt + 1 < retries:
                delay = 2 ** attempt
                print(
                    f"  Download failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Could not download {replay_id}: {last_error}"
    )


def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "unknown"


def parse_replay(data, replay_id, predictor):
    """
    Run Metamon's replay reconstruction.

    Returns:
        p1_replay, p2_replay, battle_date
    """

    parser = ReplayParser(
        team_predictor=predictor,
    )

    timestamp = int(data["uploadtime"])
    time_played = datetime.fromtimestamp(timestamp)
    battle_date = time_played.date()

    replay = forward.ParsedReplay(
        gameid=replay_id,
        format=data["formatid"],
        time_played=time_played,
    )

    log = parser.clean_log(data)

    replay = forward.forward_fill(
        replay,
        log,
        verbose=False,
    )

    p1_replay, p2_replay = backward.backward_fill(
        replay,
        team_predictor=predictor,
    )

    return p1_replay, p2_replay, battle_date


def fully_predict_team(
    predictor,
    revealed_team,
    battle_date,
    rating,
    replay_id,
):
    """
    IMPORTANT:

    POVReplay.revealed_team is the maximally revealed team from
    the replay. It may still contain $$missing-move$$ and/or
    $$missing-name$$.

    Explicitly run ReplayPredictor.predict() to fill those fields.
    """

    return predictor.predict(
        revealed_team,
        date=battle_date,
        rating=rating,
        gameid=replay_id,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download Pokemon Showdown replays and reconstruct "
            "complete teams using Metamon ReplayPredictor."
        )
    )

    parser.add_argument(
        "urls",
        type=Path,
        help="Text file containing replay URLs.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("extracted_replay_teams"),
        help="Output directory.",
    )

    parser.add_argument(
        "--replay-stats-dir",
        default=None,
        help=(
            "Optional Metamon replay-statistics directory. "
            "If omitted, Metamon uses/downloads its default data."
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Seconds to wait between replay downloads.",
    )

    args = parser.parse_args()

    output_dir = args.out
    raw_dir = output_dir / "raw_json"
    team_dir = output_dir / "teams"

    raw_dir.mkdir(parents=True, exist_ok=True)
    team_dir.mkdir(parents=True, exist_ok=True)

    replay_ids = load_replay_ids(args.urls)

    print(
        f"Found {len(replay_ids)} unique replay URLs."
    )

    #
    # This is the important predictor.
    #
    predictor = ReplayPredictor(
        replay_stats_dir=args.replay_stats_dir,
    )

    failures = []
    combined_teams = []

    successful_replays = 0
    written_teams = 0

    for index, replay_id in enumerate(
        replay_ids,
        start=1,
    ):
        print(
            f"[{index}/{len(replay_ids)}] "
            f"{replay_id}"
        )

        try:
            #
            # Download replay.
            #
            json_path = (
                raw_dir /
                f"{replay_id}.json"
            )

            data = download_replay_json(
                replay_id,
                json_path,
            )

            #
            # Sanity check.
            #
            replay_format = data.get("formatid")

            if replay_format != "gen1ou":
                raise ValueError(
                    f"Expected gen1ou but replay reports "
                    f"{replay_format!r}"
                )

            players = data.get(
                "players",
                ["p1", "p2"],
            )

            if len(players) < 2:
                players = ["p1", "p2"]

            p1_name = players[0]
            p2_name = players[1]

            #
            # First reconstruct the replay.
            #
            p1_replay, p2_replay, battle_date = (
                parse_replay(
                    data,
                    replay_id,
                    predictor,
                )
            )

            rating = data.get("rating")

            #
            # CRITICAL STEP:
            #
            # p1_replay.revealed_team and
            # p2_replay.revealed_team are NOT necessarily
            # complete.
            #
            # Explicitly pass each through ReplayPredictor.
            #
            p1_team = fully_predict_team(
                predictor=predictor,
                revealed_team=p1_replay.revealed_team,
                battle_date=battle_date,
                rating=rating,
                replay_id=replay_id,
            )

            p2_team = fully_predict_team(
                predictor=predictor,
                revealed_team=p2_replay.revealed_team,
                battle_date=battle_date,
                rating=rating,
                replay_id=replay_id,
            )

            teams = [
                (
                    "p1",
                    p1_name,
                    p1_team,
                ),
                (
                    "p2",
                    p2_name,
                    p2_team,
                ),
            ]

            #
            # Write the completed teams.
            #
            for side, username, team in teams:

                team_text = (
                    team.to_str().strip()
                    + "\n"
                )

                #
                # DO NOT silently accept incomplete prediction.
                #
                if "$$missing" in team_text:
                    raise RuntimeError(
                        "\n"
                        "ReplayPredictor returned an "
                        "incomplete team.\n"
                        f"Replay: {replay_id}\n"
                        f"Side: {side}\n"
                        f"Player: {username}\n\n"
                        f"{team_text}"
                    )

                filename = (
                    f"{replay_id}"
                    f"__{side}"
                    f"__{safe_filename(username)}"
                    f".txt"
                )

                team_path = (
                    team_dir /
                    filename
                )

                team_path.write_text(
                    team_text
                )

                combined_teams.append(
                    f"### {replay_id} | "
                    f"{side} | {username}\n\n"
                    f"{team_text}"
                )

                written_teams += 1

            successful_replays += 1

        except Exception as e:
            failure = (
                f"{replay_id}\t"
                f"{type(e).__name__}: "
                f"{e}"
            )

            failures.append(failure)

            print(
                f"  FAILED: "
                f"{type(e).__name__}: {e}"
            )

        time.sleep(args.sleep)

    #
    # Save deduplicated input URLs.
    #
    dedup_path = (
        output_dir /
        "deduplicated_replay_urls.txt"
    )

    dedup_path.write_text(
        "".join(
            f"https://replay.pokemonshowdown.com/"
            f"{replay_id}\n"
            for replay_id in replay_ids
        )
    )

    #
    # Save one aggregate human-readable file.
    #
    combined_path = (
        output_dir /
        "all_teams.txt"
    )

    combined_path.write_text(
        "\n\n".join(combined_teams)
    )

    #
    # Save failures.
    #
    failure_path = (
        output_dir /
        "failures.txt"
    )

    failure_path.write_text(
        "\n".join(failures)
        + ("\n" if failures else "")
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Unique replay URLs:  "
        f"{len(replay_ids)}"
    )

    print(
        f"Parsed successfully: "
        f"{successful_replays}"
    )

    print(
        f"Failed:              "
        f"{len(failures)}"
    )

    print(
        f"Complete teams:      "
        f"{written_teams}"
    )

    print()
    print(
        f"Teams:    {team_dir}"
    )

    print(
        f"Combined: {combined_path}"
    )

    print(
        f"Failures: {failure_path}"
    )


if __name__ == "__main__":
    main()
