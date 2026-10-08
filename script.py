#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


HEADER_RE = re.compile(
    r"^\s*===\s*\[gen1ou\]\s*(.*?)\s*===\s*$",
    re.IGNORECASE,
)

DEFAULT_EVS = (
    "EVs: 252 HP / 252 Atk / 252 Def / "
    "252 SpA / 252 SpD / 252 Spe"
)

DEFAULT_IVS = (
    "IVs: 31 HP / 31 Atk / 31 Def / "
    "31 SpA / 31 SpD / 31 Spe"
)


def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_. -]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("._") or "unnamed"


def split_teams(text: str):
    """Split backup into (team_name, team_text) pairs."""
    teams = []
    current_name = None
    current_lines = []

    for line in text.splitlines():
        match = HEADER_RE.match(line)

        if match:
            if current_name is not None:
                body = "\n".join(current_lines).strip()
                if body:
                    teams.append((current_name, body))

            current_name = match.group(1).strip()
            current_lines = []

        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        body = "\n".join(current_lines).strip()
        if body:
            teams.append((current_name, body))

    return teams


def strip_nickname(first_line: str) -> str:
    """
    Convert, e.g.

        MATERIAL BALANCE (Jynx) (F)
        Welcome to New York (Starmie)
        Chansey (F)
        Tauros

    to

        Jynx (F)
        Starmie
        Chansey (F)
        Tauros
    """

    # Showdown nickname syntax:
    #     Nickname (Species)
    # optionally followed by gender:
    #     Nickname (Species) (F)
    #
    # Grab the LAST parenthesized expression before an optional gender.
    match = re.match(
        r"^.*\(([^()]+)\)(\s+\((?:M|F)\))?\s*$",
        first_line,
    )

    if match:
        species = match.group(1).strip()
        gender = match.group(2) or ""

        # Don't mistake an ordinary gender marker for a species.
        if species not in {"M", "F"}:
            return species + gender

    # Already has no nickname.
    return first_line.strip()


def normalize_pokemon_block(block: str) -> str:
    lines = [
        line.rstrip()
        for line in block.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    # Strip nickname from first line.
    lines[0] = strip_nickname(lines[0])

    has_evs = any(
        line.strip().lower().startswith("evs:")
        for line in lines
    )

    has_ivs = any(
        line.strip().lower().startswith("ivs:")
        for line in lines
    )

    #
    # Put EVs/IVs immediately before the moves.
    #
    first_move = next(
        (
            i
            for i, line in enumerate(lines)
            if line.lstrip().startswith("- ")
        ),
        len(lines),
    )

    additions = []

    if not has_evs:
        additions.append(DEFAULT_EVS)

    if not has_ivs:
        additions.append(DEFAULT_IVS)

    lines[first_move:first_move] = additions

    return "\n".join(lines)


def normalize_team(team_text: str) -> str:
    blocks = [
        block
        for block in re.split(
            r"\n\s*\n",
            team_text.strip(),
        )
        if block.strip()
    ]

    normalized = [
        normalize_pokemon_block(block)
        for block in blocks
    ]

    return "\n\n".join(normalized)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split an RBY OU backup into individual normalized "
            "Pokemon Showdown teams."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="RBY team backup text file",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backup_teams"),
        help="Output directory",
    )

    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    teams = split_teams(text)

    if not teams:
        raise RuntimeError(
            "No teams found. Expected headers like:\n"
            "=== [gen1ou] Team Name ==="
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    for i, (name, raw_team) in enumerate(
        teams,
        start=1,
    ):
        team = normalize_team(raw_team)

        blocks = [
            b
            for b in re.split(r"\n\s*\n", team)
            if b.strip()
        ]

        if len(blocks) != 6:
            print(
                f"WARNING: {name!r} contains "
                f"{len(blocks)} Pokemon"
            )

        filename = (
            f"backup__{i:03d}__"
            f"{safe_filename(name)}.txt"
        )

        path = args.out / filename

        path.write_text(
            team.rstrip() + "\n",
            encoding="utf-8",
        )

        print(f"{filename}: {name}")

    print()
    print(
        f"Wrote {len(teams)} teams to "
        f"{args.out}/"
    )


if __name__ == "__main__":
    main()
