"""Rotation utilities: 6D continuous rotation representation (Zhou et al. 2019).

Provides conversions between 6D rotation representation, rotation matrices,
and quaternions. The 6D representation is continuous and avoids the
discontinuities of quaternions and Euler angles, making it better suited
for neural network regression.

Reference: "On the Continuity of Rotation Representations in Neural Networks"
           Zhou et al., CVPR 2019.
"""

import torch
import torch.nn.functional as F


def rotation_matrix_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation.

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) 6D rotation representation (first two columns of R).
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix.

    Uses Gram-Schmidt orthogonalization to recover a valid rotation matrix
    from the first two column vectors.

    Args:
        rot_6d: (..., 6) 6D rotation representation.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to rotation matrix.

    Args:
        quaternion: (..., 4) quaternions in (x, y, z, w) format.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    x, y, z, w = quaternion.unbind(-1)

    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy),
        ],
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)

    return matrix


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (x, y, z, w).

    Uses Shepperd's method for numerical stability.

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions in (x, y, z, w) format.
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    batch_size = m.shape[0]

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    quat = torch.zeros(batch_size, 4, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0  # s = 4*w
    mask = trace > 0
    quat[mask, 3] = 0.25 * s[mask]
    quat[mask, 0] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    quat[mask, 1] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    quat[mask, 2] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

    # Case 2: m[0,0] is largest diagonal
    mask2 = (~mask) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10)) * 2.0
    quat[mask2, 3] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s2[mask2]
    quat[mask2, 0] = 0.25 * s2[mask2]
    quat[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s2[mask2]
    quat[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s2[mask2]

    # Case 3: m[1,1] is largest diagonal
    mask3 = (~mask) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10)) * 2.0
    quat[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s3[mask3]
    quat[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s3[mask3]
    quat[mask3, 1] = 0.25 * s3[mask3]
    quat[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s3[mask3]

    # Case 4: m[2,2] is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10)) * 2.0
    quat[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s4[mask4]
    quat[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s4[mask4]
    quat[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s4[mask4]
    quat[mask4, 2] = 0.25 * s4[mask4]

    # Normalize
    quat = F.normalize(quat, dim=-1)

    return quat.reshape(*batch_shape, 4)


def quaternion_to_6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to 6D rotation representation.

    Args:
        quaternion: (..., 4) quaternions in (x, y, z, w) format.

    Returns:
        (..., 6) 6D rotation representation.
    """
    return rotation_matrix_to_6d(quaternion_to_matrix(quaternion))


def rotation_6d_to_quaternion(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to quaternion (x, y, z, w).

    Args:
        rot_6d: (..., 6) 6D rotation representation.

    Returns:
        (..., 4) quaternions in (x, y, z, w) format.
    """
    return matrix_to_quaternion(rotation_6d_to_matrix(rot_6d))


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in (x, y, z, w) format.

    Args:
        q1: (..., 4) first quaternion.
        q2: (..., 4) second quaternion.

    Returns:
        (..., 4) product quaternion.
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([x, y, z, w], dim=-1)


def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion inverse (conjugate for unit quaternions).

    Args:
        q: (..., 4) quaternions in (x, y, z, w) format.

    Returns:
        (..., 4) inverse quaternions.
    """
    conj = q.clone()
    conj[..., :3] = -conj[..., :3]
    return conj


def relative_rotation_6d(
    current_quat: torch.Tensor, target_quat: torch.Tensor
) -> torch.Tensor:
    """Compute relative rotation in 6D representation.

    Computes q_delta such that target = q_delta * current,
    then returns the 6D representation of q_delta.

    Args:
        current_quat: (..., 4) current orientation as quaternion (x, y, z, w).
        target_quat: (..., 4) target orientation as quaternion (x, y, z, w).

    Returns:
        (..., 6) relative rotation in 6D representation.
    """
    q_delta = quaternion_multiply(target_quat, quaternion_inverse(current_quat))
    return quaternion_to_6d(q_delta)


def apply_rotation_delta_6d(
    current_quat: torch.Tensor, delta_6d: torch.Tensor
) -> torch.Tensor:
    """Apply a 6D rotation delta to a current quaternion orientation.

    Args:
        current_quat: (..., 4) current orientation as quaternion (x, y, z, w).
        delta_6d: (..., 6) rotation delta in 6D representation.

    Returns:
        (..., 4) new orientation as quaternion (x, y, z, w).
    """
    q_delta = rotation_6d_to_quaternion(delta_6d)
    return F.normalize(quaternion_multiply(q_delta, current_quat), dim=-1)
