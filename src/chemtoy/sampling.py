# src/chemtoy/sampling.py

"""Sampling utilities for ChemToy.

This module samples orientation angles only. It does not build rotation
matrices; the projection code converts angles into rotation matrices when
needed.

Angle convention
----------------
Each sampled orientation is represented as (θ, φ, ψ) where
    θ : polar angle from +z, in [0, π]
    φ : azimuthal angle around z, in [0, 2π)
    ψ : in-plane twist angle, in [0, 2π)

Shapes
------
All samplers return angles.shape == (N, 3). For fixed-twist modes ψ is
constant (usually zero); for the S¹ mode θ is also constant.

Notes
-----
Uniform S² sampling draws cos(θ) ~ Uniform(-1, 1), φ ~ Uniform(0, 2π) —
not θ ~ Uniform(0, π), which oversamples the poles. Uniform S¹ sampling
fixes θ and draws φ ~ Uniform(0, 2π): the samples are literal random
phases on a great circle of viewing directions.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------
# Random number handling
# ---------------------------------------------------------------------


def _rng(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    """Normalize random-number input: None, integer seed, or Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _validate_n(n: int) -> None:
    """Validate sample count."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n < 1:
        raise ValueError("n must be positive.")


# ---------------------------------------------------------------------
# S² viewing directions
# ---------------------------------------------------------------------


def sample_s2_angles(
    n: int,
    ψ: float = 0.0,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """Sample uniform viewing directions on S² with fixed in-plane twist.

    Parameters
    ----------
    n : number of samples.
    ψ : fixed in-plane twist angle.
    rng : optional random generator or seed.
    dtype : output dtype.

    Returns
    -------
    angles : (n, 3) array with columns (θ, φ, ψ). The first two angles
        define a uniform direction on S²; the third is fixed.
    """
    _validate_n(n)
    rng = _rng(rng)

    cos_θ = rng.uniform(-1.0, 1.0, size=n)
    θ = np.arccos(cos_θ)
    φ = rng.uniform(0.0, 2.0 * np.pi, size=n)
    ψ_values = np.full(n, ψ, dtype=np.float64)

    angles = np.stack([θ, φ, ψ_values], axis=1)
    return angles.astype(dtype)


def sample_s2_directions(
    n: int,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """Sample uniform viewing directions on S² as unit vectors.

    Parameters
    ----------
    n : number of directions.
    rng : optional random generator or seed.
    dtype : output dtype.

    Returns
    -------
    directions : (n, 3) array with columns (x, y, z).
    """
    angles = sample_s2_angles(n=n, ψ=0.0, rng=rng, dtype=np.float64)
    θ, φ = angles[:, 0], angles[:, 1]

    sin_θ = np.sin(θ)
    directions = np.stack(
        [sin_θ * np.cos(φ), sin_θ * np.sin(φ), np.cos(θ)], axis=1
    )
    return directions.astype(dtype)


# ---------------------------------------------------------------------
# S¹ viewing directions (great circle: random phases)
# ---------------------------------------------------------------------


def sample_s1_angles(
    n: int,
    θ0: float = np.pi / 2.0,
    ψ: float = 0.0,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """Sample uniform viewing directions on a great circle S¹ ⊂ S².

    The polar angle is pinned at θ0 (default π/2, the equator) and the
    azimuth φ ~ Uniform(0, 2π) is the only random variable — the samples
    are random phases in the literal sense. In-plane twist ψ is fixed.

    Parameters
    ----------
    n : number of samples.
    θ0 : fixed polar angle of the circle (π/2 = equatorial great circle;
        other values give non-great circles of latitude, which are still
        uniform in φ but shorter — pass π/2 for the standard S¹ latent).
    ψ : fixed in-plane twist angle.
    rng : optional random generator or seed.
    dtype : output dtype.

    Returns
    -------
    angles : (n, 3) array with columns (θ, φ, ψ); θ and ψ are constant.
    """
    _validate_n(n)
    rng = _rng(rng)

    θ = np.full(n, θ0, dtype=np.float64)
    φ = rng.uniform(0.0, 2.0 * np.pi, size=n)
    ψ_values = np.full(n, ψ, dtype=np.float64)

    angles = np.stack([θ, φ, ψ_values], axis=1)
    return angles.astype(dtype)


# ---------------------------------------------------------------------
# SO(3) angles
# ---------------------------------------------------------------------


def sample_so3_angles(
    n: int,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """Sample simple SO(3)-style Euler angles.

    Draws cos(θ) ~ Uniform(-1, 1), φ ~ Uniform(0, 2π), ψ ~ Uniform(0, 2π):
    a uniform viewing direction plus uniform in-plane twist. The projection
    module decides how to turn these angles into rotation matrices.

    Parameters
    ----------
    n : number of orientations.
    rng : optional random generator or seed.
    dtype : output dtype.

    Returns
    -------
    angles : (n, 3) array with columns (θ, φ, ψ).
    """
    _validate_n(n)
    rng = _rng(rng)

    cos_θ = rng.uniform(-1.0, 1.0, size=n)
    θ = np.arccos(cos_θ)
    φ = rng.uniform(0.0, 2.0 * np.pi, size=n)
    ψ = rng.uniform(0.0, 2.0 * np.pi, size=n)

    angles = np.stack([θ, φ, ψ], axis=1)
    return angles.astype(dtype)


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def sample_angles(
    n: int,
    mode: str = "so3",
    ψ: float = 0.0,
    θ0: float = np.pi / 2.0,
    rng: np.random.Generator | int | None = None,
    dtype=np.float32,
) -> Array:
    """Sample orientation angles.

    Parameters
    ----------
    n : number of samples.
    mode : sampling mode. Options:
        "so3"
            Uniform viewing direction with random in-plane twist.
        "s2", "s2_fixed_twist"
            Uniform viewing direction with fixed in-plane twist.
        "s1", "s1_fixed_twist", "great_circle"
            Uniform azimuth on a circle of fixed polar angle θ0
            (random phases), with fixed in-plane twist.
    ψ : fixed twist used only for the S² and S¹ modes.
    θ0 : fixed polar angle used only for the S¹ modes (default π/2).
    rng : optional random generator or seed.
    dtype : output dtype.

    Returns
    -------
    angles : (n, 3) array with columns (θ, φ, ψ).
    """
    mode = mode.lower()

    if mode == "so3":
        return sample_so3_angles(n=n, rng=rng, dtype=dtype)

    if mode in {"s2", "s2_fixed_twist"}:
        return sample_s2_angles(n=n, ψ=ψ, rng=rng, dtype=dtype)

    if mode in {"s1", "s1_fixed_twist", "great_circle"}:
        return sample_s1_angles(n=n, θ0=θ0, ψ=ψ, rng=rng, dtype=dtype)

    raise ValueError(
        "mode must be 'so3', 's2', 's2_fixed_twist', 's1', "
        "'s1_fixed_twist', or 'great_circle'."
    )


def angles_to_directions(angles: Array, dtype=np.float32) -> Array:
    """Convert sampled angles to viewing directions.

    Parameters
    ----------
    angles : (n, 3) array with columns (θ, φ, ψ); ψ is ignored.

    Returns
    -------
    directions : (n, 3) array with columns (x, y, z).
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError("angles must have shape (n, 3).")

    θ, φ = angles[:, 0], angles[:, 1]
    sin_θ = np.sin(θ)
    directions = np.stack(
        [sin_θ * np.cos(φ), sin_θ * np.sin(φ), np.cos(θ)], axis=1
    )
    return directions.astype(dtype)


def wrap_angles(angles: Array, dtype=np.float32) -> Array:
    """Wrap angles into conventional ranges.

    θ is clipped into [0, π]; φ and ψ are wrapped into [0, 2π).
    """
    angles = np.asarray(angles, dtype=np.float64).copy()
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError("angles must have shape (n, 3).")

    angles[:, 0] = np.clip(angles[:, 0], 0.0, np.pi)
    angles[:, 1] = np.mod(angles[:, 1], 2.0 * np.pi)
    angles[:, 2] = np.mod(angles[:, 2], 2.0 * np.pi)
    return angles.astype(dtype)
