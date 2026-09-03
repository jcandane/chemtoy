# src/chemtoy/sampling.py

"""
Sampling utilities for ChemToy.

This module samples orientation angles only.

It does not build rotation matrices. The projection code is responsible
for converting angles into rotation matrices when needed.

Angle convention
----------------
Each sampled orientation is represented as

    (theta, phi, psi)

where

    theta : polar angle from +z, in [0, pi]
    phi   : azimuthal angle around z, in [0, 2pi)
    psi   : in-plane twist angle, in [0, 2pi)

Shapes
------
SO(3) sampling:

    angles.shape == (N, 3)

S² fixed-twist sampling:

    angles.shape == (N, 3)

but psi is fixed, usually zero.

Notes
-----
Uniform S² sampling should sample

    cos(theta) ~ Uniform(-1, 1)
    phi        ~ Uniform(0, 2pi)

not

    theta ~ Uniform(0, pi)

because uniform theta oversamples the poles.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


# ---------------------------------------------------------------------
# Random number handling
# ---------------------------------------------------------------------


def _rng(
    rng: np.random.Generator | int | None = None,
) -> np.random.Generator:
    """
    Normalize random-number input.

    Parameters
    ----------
    rng
        None, integer seed, or existing NumPy Generator.

    Returns
    -------
    np.random.Generator
    """

    if rng is None:
        return np.random.default_rng()

    if isinstance(rng, np.random.Generator):
        return rng

    return np.random.default_rng(rng)


def _validate_n(n: int) -> None:
    """
    Validate sample count.
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer.")

    if n < 1:
        raise ValueError("n must be positive.")


# ---------------------------------------------------------------------
# S² viewing directions
# ---------------------------------------------------------------------


def sample_s2_angles(
    n: int,
    psi: float = 0.0,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """
    Sample uniform viewing directions on S² with fixed in-plane twist.

    Parameters
    ----------
    n
        Number of samples.

    psi
        Fixed in-plane twist angle.

    rng
        Optional random generator or seed.

    dtype
        Output dtype.

    Returns
    -------
    angles
        Array of shape (n, 3), with columns

            theta, phi, psi

        The first two angles define a uniform direction on S².
        The third angle is fixed.
    """

    _validate_n(n)

    rng = _rng(rng)

    cos_theta = rng.uniform(
        -1.0,
        1.0,
        size=n,
    )

    theta = np.arccos(
        cos_theta,
    )

    phi = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=n,
    )

    psi_values = np.full(
        n,
        psi,
        dtype=np.float64,
    )

    angles = np.stack(
        [
            theta,
            phi,
            psi_values,
        ],
        axis=1,
    )

    return angles.astype(dtype)


def sample_s2_directions(
    n: int,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """
    Sample uniform viewing directions on S².

    Parameters
    ----------
    n
        Number of directions.

    rng
        Optional random generator or seed.

    dtype
        Output dtype.

    Returns
    -------
    directions
        Array of shape (n, 3), with columns

            x, y, z
    """

    angles = sample_s2_angles(
        n=n,
        psi=0.0,
        rng=rng,
        dtype=np.float64,
    )

    theta = angles[:, 0]
    phi = angles[:, 1]

    sin_theta = np.sin(theta)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)

    directions = np.stack(
        [
            x,
            y,
            z,
        ],
        axis=1,
    )

    return directions.astype(dtype)


# ---------------------------------------------------------------------
# SO(3) angles
# ---------------------------------------------------------------------


def sample_so3_angles(
    n: int,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """
    Sample simple SO(3)-style Euler angles.

    Parameters
    ----------
    n
        Number of orientations.

    rng
        Optional random generator or seed.

    dtype
        Output dtype.

    Returns
    -------
    angles
        Array of shape (n, 3), with columns

            theta, phi, psi

    Notes
    -----
    This samples

        cos(theta) ~ Uniform(-1, 1)
        phi        ~ Uniform(0, 2pi)
        psi        ~ Uniform(0, 2pi)

    which gives a uniform viewing direction plus uniform in-plane twist.

    The projection module will decide how to turn these angles into
    rotation matrices.
    """

    _validate_n(n)

    rng = _rng(rng)

    cos_theta = rng.uniform(
        -1.0,
        1.0,
        size=n,
    )

    theta = np.arccos(
        cos_theta,
    )

    phi = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=n,
    )

    psi = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=n,
    )

    angles = np.stack(
        [
            theta,
            phi,
            psi,
        ],
        axis=1,
    )

    return angles.astype(dtype)


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def sample_angles(
    n: int,
    mode: str = "so3",
    psi: float = 0.0,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """
    Sample orientation angles.

    Parameters
    ----------
    n
        Number of samples.

    mode
        Sampling mode.

        Options:

            "so3"
                Uniform viewing direction with random in-plane twist.

            "s2"
            "s2_fixed_twist"
                Uniform viewing direction with fixed in-plane twist.

    psi
        Fixed twist used only for S² modes.

    rng
        Optional random generator or seed.

    dtype
        Output dtype.

    Returns
    -------
    angles
        Array of shape (n, 3), with columns

            theta, phi, psi
    """

    mode = mode.lower()

    if mode == "so3":
        return sample_so3_angles(
            n=n,
            rng=rng,
            dtype=dtype,
        )

    if mode in {"s2", "s2_fixed_twist"}:
        return sample_s2_angles(
            n=n,
            psi=psi,
            rng=rng,
            dtype=dtype,
        )

    raise ValueError(
        "mode must be 'so3', 's2', or 's2_fixed_twist'."
    )


def angles_to_directions(
    angles: Array,
    dtype=np.float32,
) -> Array:
    """
    Convert sampled angles to viewing directions.

    Parameters
    ----------
    angles
        Array of shape (n, 3), with columns

            theta, phi, psi

        psi is ignored.

    Returns
    -------
    directions
        Array of shape (n, 3), with columns

            x, y, z
    """

    angles = np.asarray(
        angles,
        dtype=np.float64,
    )

    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(
            "angles must have shape (n, 3)."
        )

    theta = angles[:, 0]
    phi = angles[:, 1]

    sin_theta = np.sin(theta)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)

    directions = np.stack(
        [
            x,
            y,
            z,
        ],
        axis=1,
    )

    return directions.astype(dtype)


def wrap_angles(
    angles: Array,
    dtype=np.float32,
) -> Array:
    """
    Wrap angles into conventional ranges.

    theta is clipped into [0, pi].
    phi and psi are wrapped into [0, 2pi).
    """

    angles = np.asarray(
        angles,
        dtype=np.float64,
    ).copy()

    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(
            "angles must have shape (n, 3)."
        )

    angles[:, 0] = np.clip(
        angles[:, 0],
        0.0,
        np.pi,
    )

    angles[:, 1] = np.mod(
        angles[:, 1],
        2.0 * np.pi,
    )

    angles[:, 2] = np.mod(
        angles[:, 2],
        2.0 * np.pi,
    )

    return angles.astype(dtype)
