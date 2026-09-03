# src/chemtoy/project.py

"""
Projection routines for ChemToy.

This module converts a 3D atomic structure into simulated 2D
measurements.

For now, only toy CryoEM-style image projections are implemented.

Current model
-------------
- Each atom is represented by the same isotropic Gaussian blob.
- No element-specific scattering.
- No CTF.
- No dose model.
- No noise.
- No solvent.
- No microscope physics.

Future model
------------
The same public project(...) API can later dispatch to diffraction
simulation when given a DiffractionDetector.

Design rule
-----------
project.py performs numerical projection.

It does not:
- download structures
- sample angles
- define detector classes
- save files
- make plots
"""

from __future__ import annotations

import numpy as np

from .detector import Detector2D, ImageDetector, DiffractionDetector
from .structure import Structure


Array = np.ndarray


# ---------------------------------------------------------------------
# Angle handling
# ---------------------------------------------------------------------


def validate_angles(
    angles: Array,
) -> Array:
    """
    Validate orientation angles.

    Parameters
    ----------
    angles
        Array of shape (N, 3), with columns:

            theta, phi, psi

    Returns
    -------
    ndarray
        Float32 angle array with shape (N, 3).
    """

    angles = np.asarray(
        angles,
        dtype=np.float32,
    )

    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(
            "angles must have shape (N, 3), with columns theta, phi, psi."
        )

    return angles


def euler_matrix(
    theta: float,
    phi: float,
    psi: float,
) -> Array:
    """
    Convert Euler angles to a 3D rotation matrix.

    Convention
    ----------
    The angles are interpreted as:

        theta : polar angle from +z
        phi   : azimuth around z
        psi   : in-plane twist around z

    The matrix is:

        R = Rz(phi) @ Ry(theta) @ Rz(psi)

    This is a simple and explicit convention suitable for a toy
    simulator. The projection code applies this matrix to centered
    coordinates before dropping the z-coordinate.
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    cp = np.cos(phi)
    sp = np.sin(phi)

    cs = np.cos(psi)
    ss = np.sin(psi)

    rz_phi = np.array(
        [
            [cp, -sp, 0.0],
            [sp, cp, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    ry_theta = np.array(
        [
            [ct, 0.0, st],
            [0.0, 1.0, 0.0],
            [-st, 0.0, ct],
        ],
        dtype=np.float32,
    )

    rz_psi = np.array(
        [
            [cs, -ss, 0.0],
            [ss, cs, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return (
        rz_phi
        @ ry_theta
        @ rz_psi
    ).astype(np.float32)


def rotate_coordinates(
    coordinates: Array,
    theta: float,
    phi: float,
    psi: float,
) -> Array:
    """
    Rotate 3D coordinates using ChemToy Euler angles.

    Parameters
    ----------
    coordinates
        Array of shape (A, 3).

    theta, phi, psi
        Euler angles in radians.

    Returns
    -------
    ndarray
        Rotated coordinates with shape (A, 3).
    """

    coordinates = np.asarray(
        coordinates,
        dtype=np.float32,
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (A, 3).")

    matrix = euler_matrix(
        theta=theta,
        phi=phi,
        psi=psi,
    )

    return (
        coordinates
        @ matrix.T
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Gaussian rendering
# ---------------------------------------------------------------------


def gaussian_kernel(
    sigma: float = 1.25,
    truncate: float = 4.0,
    dtype=np.float32,
) -> Array:
    """
    Build a small normalized 2D Gaussian kernel.

    Parameters
    ----------
    sigma
        Gaussian width in pixels.

    truncate
        Kernel radius in units of sigma.

    Returns
    -------
    ndarray
        2D Gaussian kernel.
    """

    sigma = float(sigma)
    truncate = float(truncate)

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    if truncate <= 0.0:
        raise ValueError("truncate must be positive.")

    radius = int(
        np.ceil(
            truncate * sigma
        )
    )

    axis = np.arange(
        -radius,
        radius + 1,
        dtype=np.float32,
    )

    xx, yy = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )

    kernel = np.exp(
        -0.5
        * (xx**2 + yy**2)
        / np.float32(sigma**2)
    )

    total = float(kernel.sum())

    if total > 0.0:
        kernel = kernel / total

    return kernel.astype(dtype)


def deposit_kernel(
    image: Array,
    kernel: Array,
    x: float,
    y: float,
) -> None:
    """
    Add a Gaussian kernel into an image at floating-point pixel location.

    The location is rounded to the nearest pixel.

    This function modifies image in place.
    """

    height, width = image.shape

    cx = int(round(float(x)))
    cy = int(round(float(y)))

    kh, kw = kernel.shape

    ry = kh // 2
    rx = kw // 2

    x0 = cx - rx
    x1 = cx + rx + 1

    y0 = cy - ry
    y1 = cy + ry + 1

    ix0 = max(0, x0)
    ix1 = min(width, x1)

    iy0 = max(0, y0)
    iy1 = min(height, y1)

    if ix0 >= ix1 or iy0 >= iy1:
        return

    kx0 = ix0 - x0
    kx1 = kx0 + (ix1 - ix0)

    ky0 = iy0 - y0
    ky1 = ky0 + (iy1 - iy0)

    image[iy0:iy1, ix0:ix1] += kernel[ky0:ky1, kx0:kx1]


def normalize_image(
    image: Array,
    eps: float = 1e-8,
) -> Array:
    """
    Normalize an image into [0, 1].
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    lo = float(image.min())
    hi = float(image.max())

    if hi - lo < eps:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    return (
        (image - lo)
        / (hi - lo)
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Image projection
# ---------------------------------------------------------------------


def project_image_single(
    structure: Structure,
    detector: ImageDetector,
    angle: Array,
    sigma: float = 1.25,
    truncate: float = 4.0,
    normalize: bool = True,
) -> Array:
    """
    Project one structure orientation into one toy CryoEM image.

    Parameters
    ----------
    structure
        Atomic structure.

    detector
        ImageDetector describing output image geometry.

    angle
        Array-like with three values:

            theta, phi, psi

    sigma
        Atom Gaussian width in pixels.

    truncate
        Gaussian kernel radius in sigma units.

    normalize
        Normalize density before mapping foreground/background.

    Returns
    -------
    ndarray
        Image with shape detector.shape.
    """

    angle = np.asarray(
        angle,
        dtype=np.float32,
    )

    if angle.shape != (3,):
        raise ValueError("angle must have shape (3,).")

    theta, phi, psi = angle

    coords = structure.centered_coordinates

    rotated = rotate_coordinates(
        coords,
        theta=float(theta),
        phi=float(phi),
        psi=float(psi),
    )

    xy = rotated[:, :2]

    scale = detector.fit_scale(
        xy,
    )

    pixel_xy = detector.world_to_pixel(
        xy,
        scale=scale,
    )

    kernel = gaussian_kernel(
        sigma=sigma,
        truncate=truncate,
    )

    density = detector.zeros(
        dtype=np.float32,
    )

    for x, y in pixel_xy:
        deposit_kernel(
            density,
            kernel,
            x=float(x),
            y=float(y),
        )

    if normalize:
        density = normalize_image(
            density,
        )

    image = (
        detector.background
        + (
            detector.foreground
            - detector.background
        )
        * density
    )

    return image.astype(np.float32)


def project_images(
    structure: Structure,
    detector: ImageDetector,
    angles: Array,
    sigma: float = 1.25,
    truncate: float = 4.0,
    normalize: bool = True,
) -> Array:
    """
    Project many orientations into toy CryoEM images.

    Parameters
    ----------
    structure
        Atomic structure.

    detector
        ImageDetector.

    angles
        Array of shape (N, 3), with columns theta, phi, psi.

    sigma
        Atom Gaussian width in pixels.

    truncate
        Gaussian kernel radius in sigma units.

    normalize
        Normalize each image independently.

    Returns
    -------
    ndarray
        Array of shape (N, H, W).
    """

    angles = validate_angles(
        angles,
    )

    images = np.empty(
        (
            len(angles),
            detector.height,
            detector.width,
        ),
        dtype=np.float32,
    )

    for i, angle in enumerate(angles):

        images[i] = project_image_single(
            structure=structure,
            detector=detector,
            angle=angle,
            sigma=sigma,
            truncate=truncate,
            normalize=normalize,
        )

    return images


# ---------------------------------------------------------------------
# Future diffraction placeholder
# ---------------------------------------------------------------------


def project_diffraction(
    structure: Structure,
    detector: DiffractionDetector,
    angles: Array,
    *args,
    **kwargs,
) -> Array:
    """
    Future diffraction projection.

    This is intentionally not implemented yet.

    Later this will compute a toy diffraction pattern such as:

        coordinates
            -> structure factor F(q)
            -> intensity |F(q)|^2
            -> detector image
    """

    raise NotImplementedError(
        "Diffraction projection is not implemented yet."
    )


# ---------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------


def project(
    structure: Structure,
    detector: Detector2D,
    angles: Array,
    sigma: float = 1.25,
    truncate: float = 4.0,
    normalize: bool = True,
) -> Array:
    """
    Project a structure onto a detector.

    Parameters
    ----------
    structure
        Atomic structure.

    detector
        Detector object.

    angles
        Array of shape (N, 3), with columns theta, phi, psi.

    sigma
        Atom Gaussian width in pixels for image projections.

    truncate
        Gaussian kernel radius in sigma units.

    normalize
        Normalize each output image independently.

    Returns
    -------
    ndarray
        Simulated measurements.

        For ImageDetector:

            shape = (N, H, W)
    """

    if isinstance(detector, ImageDetector):

        return project_images(
            structure=structure,
            detector=detector,
            angles=angles,
            sigma=sigma,
            truncate=truncate,
            normalize=normalize,
        )

    if isinstance(detector, DiffractionDetector):

        return project_diffraction(
            structure=structure,
            detector=detector,
            angles=angles,
            sigma=sigma,
            truncate=truncate,
            normalize=normalize,
        )

    raise TypeError(
        "detector must be an ImageDetector or DiffractionDetector."
    )
