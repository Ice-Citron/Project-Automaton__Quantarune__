#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${DATA_ROOT}/.." && pwd)"
PYTHON_BIN="${QUANTARUNE_DATA_PYTHON:-python3}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" quantarune_data/scripts/gazebo_data.py cheatcode "$@"
