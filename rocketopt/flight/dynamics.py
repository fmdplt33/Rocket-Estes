"""Rigid-body kinematics: quaternions and the equations of motion.

Quaternion convention
---------------------
Quaternions are stored as ``[w, x, y, z]`` with ``w`` the scalar part, and
represent the rotation that takes a vector **from the body frame to the
inertial frame**. They are kept normalised at every integration step; a
6-DOF integration will otherwise drift off the unit sphere and silently
introduce a spurious scaling into every rotated vector.

Body frame
----------
* ``x`` forward, out through the nose
* ``y`` to starboard
* ``z`` completing the right-handed set

Inertial frame
--------------
East-north-up, anchored at the launch pad.

References
----------
[1] Kuipers, J. B. (1999). *Quaternions and Rotation Sequences*. Princeton
    University Press, Chs. 5-7.
[2] Stevens, B. L., Lewis, F. L., & Johnson, E. N. (2015). *Aircraft Control
    and Simulation*, 3rd ed., Sec. 1.4 (strapdown equations) and Sec. 2.5
    (rigid-body equations of motion).
[3] Zipfel, P. H. (2007). *Modeling and Simulation of Aerospace Vehicle
    Dynamics*, 2nd ed., Ch. 4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "quaternion_normalise",
    "quaternion_multiply",
    "quaternion_to_matrix",
    "quaternion_from_axis_angle",
    "quaternion_from_vectors",
    "quaternion_derivative",
    "rotate_body_to_inertial",
    "rotate_inertial_to_body",
    "euler_angles",
]


def quaternion_normalise(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``q`` scaled to unit length.

    Parameters
    ----------
    q:
        Quaternion ``[w, x, y, z]``.

    Returns
    -------
    numpy.ndarray
        Unit quaternion. If ``q`` has negligible norm the identity rotation is
        returned, which keeps an integration step from producing NaNs.
    """
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quaternion_multiply(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Hamilton product ``a * b`` [1], Sec. 5.4.

    Parameters
    ----------
    a, b:
        Quaternions ``[w, x, y, z]``.

    Returns
    -------
    numpy.ndarray
        The product quaternion.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quaternion_to_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix taking body-frame vectors to the inertial frame.

    Parameters
    ----------
    q:
        Unit quaternion ``[w, x, y, z]``.

    Returns
    -------
    numpy.ndarray
        A 3x3 orthonormal rotation matrix [1], Eq. 7.7.
    """
    w, x, y, z = quaternion_normalise(q)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_from_axis_angle(
    axis: NDArray[np.float64], angle: float
) -> NDArray[np.float64]:
    """Quaternion for a rotation of ``angle`` about ``axis``.

    Parameters
    ----------
    axis:
        Rotation axis; need not be normalised.
    angle:
        Rotation angle [rad], right-handed about ``axis``.

    Returns
    -------
    numpy.ndarray
        Unit quaternion ``[w, x, y, z]``.
    """
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    unit = np.asarray(axis, dtype=np.float64) / n
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], unit * np.sin(half)])


def quaternion_from_vectors(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Shortest-arc quaternion rotating ``source`` onto ``target``.

    Used at initialisation to point the body ``x`` axis along the launch rail.

    Parameters
    ----------
    source, target:
        Vectors; need not be normalised.

    Returns
    -------
    numpy.ndarray
        Unit quaternion taking ``source`` to ``target``.
    """
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)

    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1e-12:
        # Antiparallel: rotate 180 degrees about any perpendicular axis.
        perpendicular = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(perpendicular)) < 1e-9:
            perpendicular = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return quaternion_from_axis_angle(perpendicular, np.pi)

    axis = np.cross(a, b)
    return quaternion_normalise(np.concatenate([[1.0 + dot], axis]))


def quaternion_derivative(
    q: NDArray[np.float64], omega_body: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Time derivative of a body-to-inertial quaternion.

    ``dq/dt = 0.5 * q * [0, omega_body]`` [2], Eq. 1.4-11. The angular rate is
    expressed in the body frame, which is what a rate gyro measures and what
    the rigid-body moment equation produces.

    Parameters
    ----------
    q:
        Current quaternion ``[w, x, y, z]``.
    omega_body:
        Angular velocity in the body frame [rad/s].

    Returns
    -------
    numpy.ndarray
        Quaternion derivative.
    """
    omega_quat = np.concatenate([[0.0], np.asarray(omega_body, dtype=np.float64)])
    return 0.5 * quaternion_multiply(q, omega_quat)


def rotate_body_to_inertial(
    q: NDArray[np.float64], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rotate a body-frame vector into the inertial frame.

    Parameters
    ----------
    q:
        Unit quaternion ``[w, x, y, z]``.
    vector:
        Vector in body coordinates.

    Returns
    -------
    numpy.ndarray
        The same vector in inertial coordinates.
    """
    return quaternion_to_matrix(q) @ vector


def rotate_inertial_to_body(
    q: NDArray[np.float64], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rotate an inertial-frame vector into the body frame.

    Parameters
    ----------
    q:
        Unit quaternion ``[w, x, y, z]``.
    vector:
        Vector in inertial coordinates.

    Returns
    -------
    numpy.ndarray
        The same vector in body coordinates.
    """
    return quaternion_to_matrix(q).T @ vector


def euler_angles(q: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert a quaternion to roll, pitch and yaw [rad].

    Uses the aerospace 3-2-1 (yaw-pitch-roll) sequence [2], Sec. 1.3. Pitch is
    clamped at the +/-90 degree gimbal-lock singularity rather than allowed to
    produce a NaN.

    Parameters
    ----------
    q:
        Unit quaternion ``[w, x, y, z]``.

    Returns
    -------
    tuple of float
        ``(roll, pitch, yaw)`` [rad].
    """
    w, x, y, z = quaternion_normalise(q)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sin_roll, cos_roll))

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = float(np.arcsin(np.clip(sin_pitch, -1.0, 1.0)))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(sin_yaw, cos_yaw))

    return roll, pitch, yaw
