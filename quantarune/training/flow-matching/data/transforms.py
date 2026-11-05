"""Action space transforms for converting between absolute and relative representations.

Supports two action modes:
- velocity: 6D twist (linear.x/y/z, angular.x/y/z) — already relative
- position: 9D relative pose delta (dx, dy, dz, 6D_rot_delta)
"""

import torch
import numpy as np

from quantarune_training.utils.rotation import (
    quaternion_to_6d,
    relative_rotation_6d,
    apply_rotation_delta_6d,
    rotation_6d_to_quaternion,
)


# Observation state field indices matching ObservationState in aic_robot_aic_controller.py
TCP_POS_INDICES = slice(0, 3)       # tcp_pose.position x,y,z
TCP_QUAT_INDICES = slice(3, 7)      # tcp_pose.orientation x,y,z,w
TCP_LIN_VEL_INDICES = slice(7, 10)  # tcp_velocity.linear x,y,z
TCP_ANG_VEL_INDICES = slice(10, 13) # tcp_velocity.angular x,y,z
TCP_ERROR_INDICES = slice(13, 19)   # tcp_error x,y,z,rx,ry,rz
JOINT_POS_INDICES = slice(19, 26)   # joint_positions 0-6

VELOCITY_ACTION_DIM = 6
POSITION_ACTION_DIM = 9  # 3 pos + 6 rot (6D continuous)


def build_velocity_chunks(
    actions: np.ndarray, horizon: int
) -> np.ndarray:
    """Build velocity action chunks from a sequence of velocity commands.

    Velocity actions are already relative, so we just window them.

    Args:
        actions: (T, 6) velocity actions [linear.xyz, angular.xyz].
        horizon: Number of future steps per chunk.

    Returns:
        (T - horizon + 1, horizon, 6) action chunks.
    """
    T = len(actions)
    num_chunks = T - horizon + 1
    if num_chunks <= 0:
        return np.zeros((0, horizon, VELOCITY_ACTION_DIM), dtype=np.float32)

    chunks = np.stack(
        [actions[i : i + horizon] for i in range(num_chunks)], axis=0
    )
    return chunks.astype(np.float32)


def build_position_chunks(
    tcp_positions: np.ndarray,
    tcp_quaternions: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Build relative position action chunks from absolute TCP trajectory.

    For each timestep t, computes deltas to the next H poses relative to
    the pose at time t:
        delta_pos[h] = tcp_pos[t + h + 1] - tcp_pos[t]
        delta_rot[h] = 6D_repr(quat[t + h + 1] * quat[t]^-1)

    Args:
        tcp_positions: (T, 3) TCP positions.
        tcp_quaternions: (T, 4) TCP quaternions (x, y, z, w).
        horizon: Number of future steps per chunk.

    Returns:
        (T - horizon, horizon, 9) relative action chunks [dx,dy,dz, 6D_rot].
    """
    T = len(tcp_positions)
    num_chunks = T - horizon
    if num_chunks <= 0:
        return np.zeros((0, horizon, POSITION_ACTION_DIM), dtype=np.float32)

    pos_t = torch.from_numpy(tcp_positions).float()
    quat_t = torch.from_numpy(tcp_quaternions).float()

    chunks = []
    for i in range(num_chunks):
        current_pos = pos_t[i]
        current_quat = quat_t[i]

        delta_pos = pos_t[i + 1 : i + 1 + horizon] - current_pos.unsqueeze(0)

        future_quats = quat_t[i + 1 : i + 1 + horizon]
        current_quat_expanded = current_quat.unsqueeze(0).expand_as(future_quats)
        delta_rot_6d = relative_rotation_6d(current_quat_expanded, future_quats)

        chunk = torch.cat([delta_pos, delta_rot_6d], dim=-1)
        chunks.append(chunk)

    return torch.stack(chunks).numpy()


def relative_position_to_absolute(
    current_position: np.ndarray,
    current_quaternion: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a single relative position delta to absolute pose.

    Args:
        current_position: (3,) current TCP position.
        current_quaternion: (4,) current TCP quaternion (x, y, z, w).
        delta: (9,) relative delta [dx, dy, dz, 6D_rot_delta].

    Returns:
        target_position: (3,) absolute target position.
        target_quaternion: (4,) absolute target quaternion (x, y, z, w).
    """
    delta_t = torch.from_numpy(delta).float()
    current_pos_t = torch.from_numpy(current_position).float()
    current_quat_t = torch.from_numpy(current_quaternion).float()

    target_pos = current_pos_t + delta_t[:3]
    target_quat = apply_rotation_delta_6d(current_quat_t, delta_t[3:9])

    return target_pos.numpy(), target_quat.numpy()


def extract_tcp_pose_from_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract TCP position and quaternion from a 26D state vector.

    Args:
        state: (26,) or (T, 26) state vector(s).

    Returns:
        positions: (..., 3) TCP positions.
        quaternions: (..., 4) TCP quaternions (x, y, z, w).
    """
    return state[..., TCP_POS_INDICES], state[..., TCP_QUAT_INDICES]
