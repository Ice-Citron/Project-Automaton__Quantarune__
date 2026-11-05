#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOX_CHARS = {"╔", "╗", "╚", "╝", "═", "║"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the final aic_engine scoring block from an engine log."
    )
    parser.add_argument("engine_log", type=Path, help="Path to engine.log")
    parser.add_argument("output_yaml", type=Path, help="Path to write scoring.yaml")
    return parser.parse_args()


def normalize_engine_line(line: str) -> str | None:
    if "[aic_engine]:" not in line:
        return None

    payload = line.split("[aic_engine]:", 1)[1]
    payload = ANSI_RE.sub("", payload.rstrip("\n"))
    if payload.startswith(" "):
        payload = payload[1:]
    return payload


def is_box_line(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(ch in BOX_CHARS for ch in stripped)


def extract_scoring_lines(engine_log: Path) -> list[str]:
    started = False
    collected: list[str] = []

    for raw_line in engine_log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = normalize_engine_line(raw_line)
        if line is None:
            continue
        if is_box_line(line):
            continue
        if not started:
            if line.startswith("total:"):
                started = True
                collected.append(line)
            continue

        if not line.strip():
            continue

        if line.startswith("trial_") or line.startswith(" "):
            collected.append(line)
            continue

        break

    if not collected:
        raise ValueError(f"No scoring block found in {engine_log}")

    return collected


def load_scoring_payload(lines: list[str]) -> dict[str, object]:
    text = "\n".join(lines) + "\n"
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict) or "total" not in payload:
        raise ValueError("Extracted scoring block is not valid scoring YAML")
    return payload


def main() -> int:
    args = parse_args()
    if not args.engine_log.is_file():
        raise FileNotFoundError(f"Engine log not found: {args.engine_log}")

    lines = extract_scoring_lines(args.engine_log)
    payload = load_scoring_payload(lines)

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Wrote scoring YAML to: {args.output_yaml}")
    print(f"Total score: {payload['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
