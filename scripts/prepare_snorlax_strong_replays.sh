#!/usr/bin/env bash
set -Eeuo pipefail

# Download and safely unpack the pinned SnorlaxV1 strong-model replay corpus.
# The output layout is the one expected by MetamonDataset:
#   GOOD_REPLAYS_DIR/gen1ou/*.json[.lz4]

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-/workspace/good_replays}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STRONG_REPLAYS_REVISION="${STRONG_REPLAYS_REVISION:-623ce1886eb93a2a893638ed413443dd7b4123aa}"
STRONG_REPLAYS_SHA256="${STRONG_REPLAYS_SHA256:-b43668b1a4342b2e27125ee5d52fb3ec14d7b8d3342d68ae92651d90348139e4}"
STRONG_REPLAYS_URL="${STRONG_REPLAYS_URL:-https://huggingface.co/datasets/bsmathers/scraped_replays_parsed/resolve/${STRONG_REPLAYS_REVISION}/gen1ou.zip?download=true}"
STRONG_REPLAYS_ARCHIVE="${STRONG_REPLAYS_ARCHIVE:-${GOOD_REPLAYS_DIR%/}.gen1ou.zip}"

if [[ -d "${GOOD_REPLAYS_DIR}/gen1ou" ]] \
    && find "${GOOD_REPLAYS_DIR}/gen1ou" -maxdepth 1 -type f \
      \( -name '*.json' -o -name '*.json.lz4' \) -print -quit | grep -q .; then
  log "Reusing strong-model trajectories in ${GOOD_REPLAYS_DIR}/gen1ou"
  exit 0
fi

command -v curl >/dev/null || die "curl is required to download strong-model replays"
command -v "${PYTHON_BIN}" >/dev/null || die "Python is required to unpack strong-model replays"

mkdir -p "$(dirname "${GOOD_REPLAYS_DIR}")" "$(dirname "${STRONG_REPLAYS_ARCHIVE}")"
partial_archive="${STRONG_REPLAYS_ARCHIVE}.partial"
if [[ ! -f "${STRONG_REPLAYS_ARCHIVE}" ]]; then
  log "Downloading pinned strong-model replay archive"
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    --output "${partial_archive}" "${STRONG_REPLAYS_URL}"
  mv "${partial_archive}" "${STRONG_REPLAYS_ARCHIVE}"
fi

export GOOD_REPLAYS_DIR STRONG_REPLAYS_ARCHIVE STRONG_REPLAYS_REVISION STRONG_REPLAYS_SHA256
"${PYTHON_BIN}" - <<'PY'
import hashlib
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

archive = Path(os.environ["STRONG_REPLAYS_ARCHIVE"])
output_root = Path(os.environ["GOOD_REPLAYS_DIR"])
expected_sha256 = os.environ["STRONG_REPLAYS_SHA256"].lower()

digest = hashlib.sha256()
with archive.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
actual_sha256 = digest.hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"Archive SHA-256 mismatch for {archive}: "
        f"{actual_sha256} != {expected_sha256}"
    )

if output_root.exists():
    existing_files = [path for path in output_root.rglob("*") if path.is_file()]
    if existing_files:
        raise SystemExit(
            "Refusing to replace replay directory containing unrecognized files: "
            f"{output_root} (for example {existing_files[0]})"
        )

extract_root = output_root.with_name(output_root.name + ".extracting")
if extract_root.exists():
    shutil.rmtree(extract_root)
trajectory_dir = extract_root / "gen1ou"
trajectory_dir.mkdir(parents=True)

count = 0
try:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            member_path = PurePosixPath(member.filename)
            if member.is_dir() or member_path.is_absolute() or ".." in member_path.parts:
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise SystemExit(f"Unsafe archive member: {member.filename!r}")
                continue
            if not member_path.name.endswith((".json", ".json.lz4")):
                continue
            destination = trajectory_dir / member_path.name
            if destination.exists():
                raise SystemExit(f"Duplicate trajectory filename in archive: {member_path.name}")
            with source.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            count += 1
    if count == 0:
        raise SystemExit(f"No .json or .json.lz4 trajectories found in {archive}")

    (extract_root / "SOURCE.txt").write_text(
        "Hugging Face dataset: bsmathers/scraped_replays_parsed\n"
        f"Revision: {os.environ['STRONG_REPLAYS_REVISION']}\n"
        f"Archive SHA-256: {actual_sha256}\n"
        f"Trajectory files: {count}\n",
        encoding="utf-8",
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    extract_root.replace(output_root)
except BaseException:
    shutil.rmtree(extract_root, ignore_errors=True)
    raise

print(f"Prepared {count:,} strong-model trajectories in {output_root / 'gen1ou'}")
PY
