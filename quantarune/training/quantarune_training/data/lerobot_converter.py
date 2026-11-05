"""Convert LeRobot datasets to the training format for flow matching.

Reads LeRobot episodes from HuggingFace or local paths and produces
action chunks in either velocity or position mode.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from quantarune_training.data.transforms import (
    VELOCITY_ACTION_DIM,
    POSITION_ACTION_DIM,
    build_velocity_chunks,
    build_position_chunks,
    extract_tcp_pose_from_state,
)
from quantarune_training.data.normalization import compute_stats, save_stats

logger = logging.getLogger(__name__)

CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]


def convert_lerobot_dataset(
    repo_id: str | None = None,
    local_path: str | None = None,
    output_dir: str = "data/converted",
    action_mode: str = "velocity",
    chunk_horizon: int = 16,
    image_scale: float = 0.25,
    compute_normalization: bool = True,
    max_episodes: int | None = None,
) -> Path:
    """Convert a LeRobot dataset to flow matching training format.

    Args:
        repo_id: HuggingFace dataset repo ID.
        local_path: Local path to LeRobot dataset (alternative to repo_id).
        output_dir: Directory to write converted data.
        action_mode: "velocity" or "position".
        chunk_horizon: Number of future steps per action chunk.
        image_scale: Image scaling factor (should match training).
        compute_normalization: Whether to compute and save normalization stats.
        max_episodes: Optional limit for the number of episodes to convert.

    Returns:
        Path to the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if repo_id is not None:
        dataset = LeRobotDataset(repo_id, video_backend="pyav")
    elif local_path is not None:
        dataset = LeRobotDataset(local_path, video_backend="pyav")
    else:
        raise ValueError("Either repo_id or local_path must be provided")

    # Read state dim from dataset metadata
    state_dim = dataset.meta.info["features"]["observation.state"]["shape"][0]
    logger.info(f"Loaded dataset: {len(dataset)} frames, state_dim={state_dim}")

    # Collect all states, actions, and images by episode
    all_states = []
    all_chunks = []
    all_images = {cam: [] for cam in CAMERA_NAMES}

    # Get dataset row positions grouped by episode from parquet data.
    import pyarrow.parquet as pq

    data_dir_path = dataset.root / "data"
    parquet_files = sorted(data_dir_path.glob("**/*.parquet"))
    all_ep_indices = set()
    ep_rows = {}

    row_offset = 0
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["episode_index"])
        episode_ids = table.column("episode_index").to_pylist()
        for local_row, ep in enumerate(episode_ids):
            all_ep_indices.add(ep)
            ep_rows.setdefault(ep, []).append(row_offset + local_row)
        row_offset += len(episode_ids)

    episode_indices = sorted(all_ep_indices)
    logger.info(
        "Found %d episodes in parquet rows (metadata says %s)",
        len(episode_indices),
        dataset.meta.info.get("total_episodes"),
    )
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]
        logger.info("Limiting conversion to first %d episode(s)", len(episode_indices))

    for ep_idx in episode_indices:
        ep_row_indices = ep_rows[ep_idx]

        if len(ep_row_indices) < chunk_horizon + 1:
            logger.warning(
                f"Episode {ep_idx} has {len(ep_row_indices)} frames, "
                f"skipping (need at least {chunk_horizon + 1})"
            )
            continue

        logger.info(
            f"Processing episode {ep_idx}: {len(ep_row_indices)} frames..."
        )

        ep_states = []
        ep_actions = []
        ep_images = {cam: [] for cam in CAMERA_NAMES}

        for i, row_idx in enumerate(ep_row_indices):
            sample = dataset[row_idx]

            # State: use full state tensor as-is
            state = sample["observation.state"].numpy().astype(np.float32)
            ep_states.append(state)

            # Action: 6D velocity
            action = sample["action"].numpy().astype(np.float32)
            ep_actions.append(action[:VELOCITY_ACTION_DIM])

            # Images: already (C, H, W) float32 [0, 1] from LeRobot
            for cam in CAMERA_NAMES:
                key = f"observation.images.{cam}"
                img = sample[key]  # (3, H, W) float32
                # Convert to (H, W, C) uint8 for storage
                img_hwc = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ep_images[cam].append(img_hwc)

            if (i + 1) % 100 == 0:
                logger.info(f"  Episode {ep_idx}: {i + 1}/{len(ep_row_indices)} frames")

        ep_states = np.stack(ep_states)
        ep_actions = np.stack(ep_actions)

        # Build action chunks
        if action_mode == "velocity":
            chunks = build_velocity_chunks(ep_actions, chunk_horizon)
        else:
            tcp_pos, tcp_quat = extract_tcp_pose_from_state(ep_states)
            chunks = build_position_chunks(tcp_pos, tcp_quat, chunk_horizon)

        num_chunks = len(chunks)
        ep_states_trimmed = ep_states[:num_chunks]

        all_states.append(ep_states_trimmed)
        all_chunks.append(chunks)

        for cam in CAMERA_NAMES:
            all_images[cam].extend(ep_images[cam][:num_chunks])

        # Save per-episode data
        ep_dir = output_path / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(exist_ok=True)
        torch.save({
            "states": torch.from_numpy(ep_states_trimmed),
            "action_chunks": torch.from_numpy(chunks),
        }, ep_dir / "data.pt")

        # Save images
        for cam in CAMERA_NAMES:
            imgs = np.stack(ep_images[cam][:num_chunks])
            np.save(ep_dir / f"{cam}.npy", imgs)

        logger.info(
            f"Episode {ep_idx}: {num_chunks} chunks "
            f"(state: {ep_states_trimmed.shape}, chunks: {chunks.shape})"
        )

    # Save metadata
    action_dim = VELOCITY_ACTION_DIM if action_mode == "velocity" else POSITION_ACTION_DIM
    metadata = {
        "action_mode": action_mode,
        "chunk_horizon": chunk_horizon,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "num_episodes": len(episode_indices),
        "image_scale": image_scale,
        "camera_names": CAMERA_NAMES,
        "state_feature_names": dataset.meta.info["features"]["observation.state"].get("names", []),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Compute normalization stats
    if compute_normalization and all_states:
        all_states_arr = np.concatenate(all_states)
        all_chunks_arr = np.concatenate(all_chunks)
        images_for_stats = {
            cam: imgs for cam, imgs in all_images.items() if imgs
        } or None

        stats = compute_stats(all_states_arr, all_chunks_arr, images_for_stats)
        save_stats(stats, output_path / "normalization_stats.safetensors")
        logger.info("Normalization stats saved")

    logger.info(f"Conversion complete: {output_path}")
    return output_path
