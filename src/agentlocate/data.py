"""Dataset adapters for the original Who&When JSON trajectory files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Turn:
    """One recorded multi-agent turn, indexed from zero."""

    index: int
    agent: str
    content: str


@dataclass(frozen=True)
class Trace:
    """A failed trajectory with its query and optional benchmark annotation."""

    case_id: str
    question: str
    turns: list[Turn]
    mistake_agent: str | None = None
    mistake_step: int | None = None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _agent(turn: dict[str, Any]) -> str:
    for key in ("name", "role", "agent_name", "agent"):
        candidate = _text(turn.get(key))
        if candidate:
            return candidate
    return "Unknown"


def _content(turn: dict[str, Any]) -> str:
    for key in ("content", "message"):
        candidate = _text(turn.get(key))
        if candidate:
            return candidate
    return ""


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_trace(path: Path) -> Trace:
    """Load a single original Who&When JSON sample without mutating it."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} has no non-empty history")

    turns = [
        Turn(index=index, agent=_agent(turn), content=_content(turn))
        for index, turn in enumerate(history)
        if isinstance(turn, dict)
    ]
    if not turns:
        raise ValueError(f"{path} has no readable turns")

    return Trace(
        case_id=path.stem,
        question=_text(payload.get("question") or payload.get("task")),
        turns=turns,
        mistake_agent=_text(payload.get("mistake_agent")) or None,
        mistake_step=_optional_int(payload.get("mistake_step")),
    )


def load_split_ids(path: Path) -> set[str]:
    """Load one ID-per-line split manifest committed with the benchmark."""

    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_traces(dataset_dir: Path, split_file: Path | None = None) -> list[Trace]:
    """Load a deterministic, numerically sorted set of trajectory files."""

    ids = load_split_ids(split_file) if split_file else None

    def sort_key(path: Path) -> tuple[int, str]:
        return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)

    paths = sorted(dataset_dir.glob("*.json"), key=sort_key)
    if ids is not None:
        paths = [path for path in paths if path.stem in ids]
        missing = ids - {path.stem for path in paths}
        if missing:
            raise ValueError(f"Split {split_file} references missing cases: {sorted(missing)}")
    return [load_trace(path) for path in paths]


def format_history(trace: Trace, stop_at: int | None = None) -> str:
    """Render a trajectory with stable zero-based step identifiers."""

    turns = trace.turns if stop_at is None else trace.turns[: stop_at + 1]
    return "\n".join(f"[Step {turn.index}] {turn.agent}: {turn.content}" for turn in turns)
