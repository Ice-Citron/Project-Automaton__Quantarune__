from __future__ import annotations

import importlib
from pathlib import Path

import av
import numpy as np
import torch

def _pts_from_seconds(seconds: float, time_base: float) -> int:
    if time_base <= 0:
        return 0
    return max(0, int(seconds / time_base))


def _decode_video_frames_pyav(
    video_path: str | Path,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
) -> torch.Tensor:
    del backend
    if not timestamps:
        raise RuntimeError(f"No timestamps requested for {video_path}")

    sorted_targets = sorted(
        ((index, float(target_time)) for index, target_time in enumerate(timestamps)),
        key=lambda item: item[1],
    )
    resolved_frames: list[np.ndarray | None] = [None] * len(sorted_targets)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        seek_time = max(0.0, sorted_targets[0][1] - max(tolerance_s, 0.0))
        container.seek(
            _pts_from_seconds(seek_time, time_base),
            stream=stream,
            backward=True,
        )

        target_cursor = 0
        for frame in container.decode(stream):
            current_pts = float(frame.pts or 0)
            current_time = current_pts * time_base

            while target_cursor < len(sorted_targets):
                output_index, target_time = sorted_targets[target_cursor]
                if current_time + tolerance_s < target_time:
                    break
                resolved_frames[output_index] = frame.to_ndarray(format="rgb24")
                target_cursor += 1

            if target_cursor >= len(sorted_targets):
                break

    if any(frame_value is None for frame_value in resolved_frames):
        missing_timestamps = [
            sorted_targets[index][1]
            for index, frame_value in enumerate(resolved_frames)
            if frame_value is None
        ]
        raise RuntimeError(
            f"Could not decode all requested frames from {video_path}. "
            f"missing_timestamps={missing_timestamps}"
        )

    selected_frames = np.stack(
        [frame_value for frame_value in resolved_frames if frame_value is not None],
        axis=0,
    )
    return torch.from_numpy(selected_frames).permute(0, 3, 1, 2)


def patch_lerobot_video_backend() -> None:
    candidates = (
        "lerobot.datasets.video_utils",
        "lerobot.common.datasets.video_utils",
    )

    video_utils = None
    for module_name in candidates:
        try:
            video_utils = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue

    if video_utils is None:
        raise ModuleNotFoundError(
            "Could not locate LeRobot video utils module. "
            f"Tried: {', '.join(candidates)}"
        )

    video_utils.decode_video_frames_torchvision = _decode_video_frames_pyav
    video_utils.decode_video_frames = _decode_video_frames_pyav
