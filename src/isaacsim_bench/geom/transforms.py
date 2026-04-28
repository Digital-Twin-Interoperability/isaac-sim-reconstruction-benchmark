"""Quaternion and pose math for anchor-based rigid alignment.

Pure functions, no project state.  Two quaternion conventions appear:

* ``wxyz`` — USD's authoring convention (real first), used in
  ``data/asset_anchors.json``.
* ``xyzw`` — the project's ``PredictedComponent.orientation_xyzw``.

Everything internal to this module works in ``wxyz`` form.  Conversion
helpers live at the bottom for use at the ``PredictedComponent`` boundary.

A "pose" is the pair ``(position, quat_wxyz)``.  A 4x4 row-major
homogeneous transform built from a pose is what ``align_pose`` composes.
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]
Mat4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def quat_normalize(q: QuatWXYZ) -> QuatWXYZ:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def quat_multiply(a: QuatWXYZ, b: QuatWXYZ) -> QuatWXYZ:
    """Hamilton product ``a * b`` in wxyz form."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conjugate(q: QuatWXYZ) -> QuatWXYZ:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_invert(q: QuatWXYZ) -> QuatWXYZ:
    """Inverse of a unit quaternion = its conjugate (after normalization)."""
    return quat_conjugate(quat_normalize(q))


def quat_rotate(q: QuatWXYZ, v: Vec3) -> Vec3:
    """Rotate a 3-vector by a unit quaternion."""
    qw, qx, qy, qz = quat_normalize(q)
    vx, vy, vz = v
    # t = 2 * (q.xyz x v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return (rx, ry, rz)


def quat_from_axis_angle(axis: Vec3, angle: float) -> QuatWXYZ:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(angle / 2.0) / n
    return (math.cos(angle / 2.0), ax * s, ay * s, az * s)


# Convenience: 180° rotation about world Z (the "opposed_frame" mate).
QUAT_RZ_PI: QuatWXYZ = (0.0, 0.0, 0.0, 1.0)


def pose_compose(
    a_pos: Vec3, a_quat: QuatWXYZ,
    b_pos: Vec3, b_quat: QuatWXYZ,
) -> tuple[Vec3, QuatWXYZ]:
    """Compose pose ``a * b``: rotate b's translation by a, then add."""
    ax, ay, az = a_pos
    rb = quat_rotate(a_quat, b_pos)
    new_pos = (ax + rb[0], ay + rb[1], az + rb[2])
    new_quat = quat_normalize(quat_multiply(a_quat, b_quat))
    return new_pos, new_quat


def pose_invert(pos: Vec3, quat: QuatWXYZ) -> tuple[Vec3, QuatWXYZ]:
    """Inverse of a rigid pose: ``(R, t)^-1 = (R^T, -R^T t)``."""
    inv_q = quat_invert(quat)
    nx, ny, nz = quat_rotate(inv_q, pos)
    return ((-nx, -ny, -nz), inv_q)


def align_pose(
    fixed_world_pos: Vec3,
    fixed_world_quat: QuatWXYZ,
    fixed_anchor_local_pos: Vec3,
    fixed_anchor_local_quat: QuatWXYZ,
    moving_anchor_local_pos: Vec3,
    moving_anchor_local_quat: QuatWXYZ,
    facing: str = "same_frame",
) -> tuple[Vec3, QuatWXYZ]:
    """Compute moving component's world pose so its anchor mates with fixed's.

    Formula::

        F_w  = (fixed_world_pos, fixed_world_quat)
        Fa_l = (fixed_anchor_local_pos, fixed_anchor_local_quat)
        Ma_l = (moving_anchor_local_pos, moving_anchor_local_quat)
        mate = I             if facing == "same_frame"
        mate = Rz(pi)        if facing == "opposed_frame"

        Fa_w = F_w * Fa_l
        M_w  = Fa_w * mate * Ma_l^{-1}

    Returns ``(moving_world_pos, moving_world_quat_wxyz)``.
    """
    fa_w_pos, fa_w_quat = pose_compose(
        fixed_world_pos, fixed_world_quat,
        fixed_anchor_local_pos, fixed_anchor_local_quat,
    )

    if facing == "same_frame":
        mate_quat: QuatWXYZ = (1.0, 0.0, 0.0, 0.0)
    elif facing == "opposed_frame":
        mate_quat = QUAT_RZ_PI
    else:
        raise ValueError(
            f"facing must be 'same_frame' or 'opposed_frame', got {facing!r}",
        )

    mated_pos, mated_quat = pose_compose(
        fa_w_pos, fa_w_quat,
        (0.0, 0.0, 0.0), mate_quat,
    )

    inv_pos, inv_quat = pose_invert(
        moving_anchor_local_pos, moving_anchor_local_quat,
    )
    return pose_compose(mated_pos, mated_quat, inv_pos, inv_quat)


# ---- xyzw <-> wxyz boundary helpers ----------------------------------------


def wxyz_to_xyzw(q: QuatWXYZ) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (x, y, z, w)


def xyzw_to_wxyz(q: tuple[float, float, float, float]) -> QuatWXYZ:
    x, y, z, w = q
    return (w, x, y, z)
