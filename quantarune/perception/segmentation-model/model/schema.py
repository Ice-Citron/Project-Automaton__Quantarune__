from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

TASK_CONDITION_NAMES = (
    "task.cable_type_id",
    "task.plug_type_id",
    "task.port_type_id",
    "task.target_module_id",
    "task.target_port_id",
    "task.target_module_index",
    "task.target_port_index",
)

# These ids are small integers in the current task contract. Scaling them down
# keeps the broadcast condition planes numerically comparable to RGB values.
TASK_CONDITION_SCALES = np.array(
    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    dtype=np.float32,
)

SEGMENTATION_LABEL_NAMES = ("target_port", "plug")


def _mapping_get(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]

    short_key = key.removeprefix("task.")
    if short_key in mapping:
        return mapping[short_key]

    dotted_short_key = f"task.{short_key}"
    if dotted_short_key in mapping:
        return mapping[dotted_short_key]

    return None


def task_condition_array_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    normalized: bool = True,
) -> np.ndarray:
    if mapping is None:
        values = np.zeros(len(TASK_CONDITION_NAMES), dtype=np.float32)
    else:
        raw_values: list[float] = []
        for name in TASK_CONDITION_NAMES:
            value = _mapping_get(mapping, name)
            try:
                raw_values.append(float(value))
            except (TypeError, ValueError):
                raw_values.append(0.0)
        values = np.asarray(raw_values, dtype=np.float32)

    if normalized:
        return values / TASK_CONDITION_SCALES
    return values


@dataclass(frozen=True, slots=True)
class TaskCondition:
    cable_type_id: float = 0.0
    plug_type_id: float = 0.0
    port_type_id: float = 0.0
    target_module_id: float = 0.0
    target_port_id: float = 0.0
    target_module_index: float = 0.0
    target_port_index: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TaskCondition":
        arr = task_condition_array_from_mapping(mapping, normalized=False)
        return cls(*[float(v) for v in arr.tolist()])

    def as_array(self, *, normalized: bool = True) -> np.ndarray:
        arr = np.asarray(
            [
                self.cable_type_id,
                self.plug_type_id,
                self.port_type_id,
                self.target_module_id,
                self.target_port_id,
                self.target_module_index,
                self.target_port_index,
            ],
            dtype=np.float32,
        )
        if normalized:
            return arr / TASK_CONDITION_SCALES
        return arr

    def as_tensor(self, *, normalized: bool = True) -> torch.Tensor:
        return torch.from_numpy(self.as_array(normalized=normalized))
