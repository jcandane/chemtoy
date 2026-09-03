# src/chemtoy/detector.py

"""
Detector definitions for ChemToy.

A detector describes the 2D grid on which a simulated measurement is
recorded.

This module does not perform projection, rendering, Fourier transforms,
or diffraction physics. It only defines detector geometry and parameters.

Current detectors
-----------------
ImageDetector
    Real-space image detector for toy CryoEM-style projections.

DiffractionDetector
    Reciprocal-space detector for future diffraction simulations.

Design rule
-----------
detector.py describes the measurement grid.

project.py uses detectors to generate measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _validate_shape(
    shape: tuple[int, int],
) -> tuple[int, int]:
    """
    Validate and normalize a detector shape.

    Parameters
    ----------
    shape
        Detector shape as (height, width).

    Returns
    -------
    tuple[int, int]
        Normalized shape.
    """

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2: (height, width).")

    height, width = int(shape[0]), int(shape[1])

    if height <= 0 or width <= 0:
        raise ValueError("detector dimensions must be positive.")

    return height, width


def _validate_fill_fraction(
    fill_fraction: float,
) -> float:
    """
    Validate how much of the detector the object should occupy.
    """

    fill_fraction = float(fill_fraction)

    if not 0.0 < fill_fraction <= 1.0:
        raise ValueError("fill_fraction must be in the interval (0, 1].")

    return fill_fraction


def _validate_positive(
    value: float,
    name: str,
) -> float:
    """
    Validate a positive scalar.
    """

    value = float(value)

    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")

    return value


def _coordinate_axis(
    size: int,
    spacing: float,
) -> Array:
    """
    Build a centered coordinate axis.

    For size 5 and spacing 1, this returns:

        [-2, -1, 0, 1, 2]

    For size 4 and spacing 1, this returns:

        [-1.5, -0.5, 0.5, 1.5]
    """

    center = (size - 1) / 2.0

    axis = (
        np.arange(size, dtype=np.float32)
        - center
    )

    return axis * np.float32(spacing)


# ---------------------------------------------------------------------
# Base detector behavior
# ---------------------------------------------------------------------


@dataclass(slots=True)
class Detector2D:
    """
    Base class for 2D detectors.

    Parameters
    ----------
    shape
        Detector shape as (height, width).
    """

    shape: tuple[int, int] = (128, 128)

    def __post_init__(self) -> None:
        Detector2D.__post_init__(self)

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def center(self) -> Array:
        """
        Pixel coordinate of the detector center.

        Returned as

            [x_center, y_center]

        because image coordinates usually use x before y.
        """

        return np.array(
            [
                (self.width - 1) / 2.0,
                (self.height - 1) / 2.0,
            ],
            dtype=np.float32,
        )

    @property
    def pixel_grid(self) -> tuple[Array, Array]:
        """
        Pixel index grid.

        Returns
        -------
        xx, yy
            Arrays of shape (height, width).
        """

        y = np.arange(
            self.height,
            dtype=np.float32,
        )

        x = np.arange(
            self.width,
            dtype=np.float32,
        )

        xx, yy = np.meshgrid(
            x,
            y,
            indexing="xy",
        )

        return xx, yy

    def zeros(
        self,
        dtype=np.float32,
    ) -> Array:
        """
        Allocate an empty detector image.
        """

        return np.zeros(
            self.shape,
            dtype=dtype,
        )

    def ones(
        self,
        dtype=np.float32,
    ) -> Array:
        """
        Allocate a detector image filled with ones.
        """

        return np.ones(
            self.shape,
            dtype=dtype,
        )

    def inside(
        self,
        xy: Array,
    ) -> Array:
        """
        Test whether pixel coordinates lie inside the detector.

        Parameters
        ----------
        xy
            Array of shape (N, 2), with columns x, y.

        Returns
        -------
        mask
            Boolean array of shape (N,).
        """

        xy = np.asarray(
            xy,
            dtype=np.float32,
        )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2).")

        x = xy[:, 0]
        y = xy[:, 1]

        return (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
        )


# ---------------------------------------------------------------------
# Real-space image detector
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ImageDetector(Detector2D):
    """
    Real-space detector for toy CryoEM-style projection images.

    Parameters
    ----------
    shape
        Output image shape as (height, width).

    fill_fraction
        Fraction of the smaller detector dimension that the projected
        object should occupy.

    background
        Background value used by image renderers.

    foreground
        Foreground value used by image renderers.

    pixel_size
        Optional real-space pixel size.

        For now this is metadata only. Projection can either use this
        later as Å/pixel, or continue using automatic fit-to-detector
        scaling for toy examples.
    """

    fill_fraction: float = 0.90

    background: float = 1.0

    foreground: float = 0.0

    pixel_size: float | None = None

    kind: str = "image"

    def __post_init__(self) -> None:
        super().__post_init__()

        self.fill_fraction = _validate_fill_fraction(
            self.fill_fraction,
        )

        self.background = float(
            self.background,
        )

        self.foreground = float(
            self.foreground,
        )

        if self.pixel_size is not None:
            self.pixel_size = _validate_positive(
                self.pixel_size,
                "pixel_size",
            )

    @property
    def image_radius(self) -> float:
        """
        Radius available for fitting a projected object.
        """

        return (
            min(self.shape)
            * 0.5
            * self.fill_fraction
        )

    def coordinate_grid(
        self,
    ) -> tuple[Array, Array]:
        """
        Centered real-space detector coordinates.

        If pixel_size is None, one pixel is treated as one coordinate
        unit.

        Returns
        -------
        x, y
            Arrays of shape (height, width).
        """

        spacing = (
            1.0
            if self.pixel_size is None
            else self.pixel_size
        )

        x = _coordinate_axis(
            self.width,
            spacing,
        )

        y = _coordinate_axis(
            self.height,
            spacing,
        )

        xx, yy = np.meshgrid(
            x,
            y,
            indexing="xy",
        )

        return xx, yy

    def world_to_pixel(
        self,
        xy: Array,
        scale: float,
    ) -> Array:
        """
        Convert centered 2D coordinates to pixel coordinates.

        Parameters
        ----------
        xy
            Array of shape (N, 2), centered around zero.

        scale
            Multiplicative scale from coordinate units to pixels.

        Returns
        -------
        pixel_xy
            Array of shape (N, 2), with columns x, y.
        """

        xy = np.asarray(
            xy,
            dtype=np.float32,
        )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2).")

        pixel_xy = (
            xy * np.float32(scale)
            + self.center
        )

        return pixel_xy.astype(np.float32)

    def fit_scale(
        self,
        xy: Array,
    ) -> float:
        """
        Compute an automatic scale so coordinates fit inside the image.

        Parameters
        ----------
        xy
            Array of shape (N, 2), centered projected coordinates.

        Returns
        -------
        float
            Scale factor from coordinate units to pixels.
        """

        xy = np.asarray(
            xy,
            dtype=np.float32,
        )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2).")

        radius = float(
            np.linalg.norm(
                xy,
                axis=1,
            ).max()
        )

        if radius <= 1e-12:
            return 1.0

        return self.image_radius / radius


# ---------------------------------------------------------------------
# Reciprocal-space diffraction detector
# ---------------------------------------------------------------------


@dataclass(slots=True)
class DiffractionDetector(Detector2D):
    """
    Reciprocal-space detector for future diffraction simulations.

    Parameters
    ----------
    shape
        Output diffraction pattern shape as (height, width).

    q_max
        Maximum reciprocal-space coordinate along the smaller detector
        dimension.

        This is a toy parameter for now. Later it can be connected to
        wavelength, detector distance, pixel size, and scattering angle.

    beam_center
        Optional beam center as (x, y). If None, the center of the
        detector is used.

    log_scale
        Whether visualization or later rendering should prefer log-scaled
        intensities.

    epsilon
        Small positive value for safe log transforms.
    """

    q_max: float = 1.0

    beam_center: tuple[float, float] | None = None

    log_scale: bool = True

    epsilon: float = 1e-8

    kind: str = "diffraction"

    def __post_init__(self) -> None:
        super().__post_init__()

        self.q_max = _validate_positive(
            self.q_max,
            "q_max",
        )

        self.epsilon = _validate_positive(
            self.epsilon,
            "epsilon",
        )

        self.log_scale = bool(
            self.log_scale,
        )

        if self.beam_center is not None:

            if len(self.beam_center) != 2:
                raise ValueError("beam_center must be (x, y).")

            self.beam_center = (
                float(self.beam_center[0]),
                float(self.beam_center[1]),
            )

    @property
    def center(self) -> Array:
        """
        Beam center in pixel coordinates.

        Returned as [x_center, y_center].
        """

        if self.beam_center is None:
            return super().center

        return np.array(
            self.beam_center,
            dtype=np.float32,
        )

    @property
    def q_grid(
        self,
    ) -> tuple[Array, Array]:
        """
        Reciprocal-space detector grid.

        Returns
        -------
        qx, qy
            Arrays of shape (height, width).

        Notes
        -----
        This is a simple centered square reciprocal-space grid.

        It is not yet tied to physical instrument parameters.
        """

        half_size = min(
            self.height,
            self.width,
        ) / 2.0

        q_spacing = (
            self.q_max
            / half_size
        )

        qx_axis = (
            np.arange(
                self.width,
                dtype=np.float32,
            )
            - self.center[0]
        ) * np.float32(q_spacing)

        qy_axis = (
            np.arange(
                self.height,
                dtype=np.float32,
            )
            - self.center[1]
        ) * np.float32(q_spacing)

        qx, qy = np.meshgrid(
            qx_axis,
            qy_axis,
            indexing="xy",
        )

        return qx, qy

    @property
    def q_radius_grid(
        self,
    ) -> Array:
        """
        Radial reciprocal-space coordinate.
        """

        qx, qy = self.q_grid

        return np.sqrt(
            qx**2
            + qy**2
        ).astype(np.float32)

    def apply_intensity_transform(
        self,
        intensity: Array,
    ) -> Array:
        """
        Apply the detector's preferred intensity transform.

        For now this is mainly useful for future diffraction visualization.
        """

        intensity = np.asarray(
            intensity,
            dtype=np.float32,
        )

        if not self.log_scale:
            return intensity

        return np.log1p(
            intensity
            + np.float32(self.epsilon)
        ).astype(np.float32)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def make_detector(
    config: dict,
) -> Detector2D:
    """
    Build a detector from a dictionary.

    Examples
    --------
    Image detector:

        {
            "kind": "image",
            "shape": [128, 128],
            "fill_fraction": 0.9
        }

    Diffraction detector:

        {
            "kind": "diffraction",
            "shape": [256, 256],
            "q_max": 2.0
        }
    """

    if "kind" not in config:
        raise KeyError("detector config must contain 'kind'.")

    kind = str(
        config["kind"]
    ).lower().strip()

    shape = tuple(
        config.get(
            "shape",
            (128, 128),
        )
    )

    if kind == "image":

        return ImageDetector(
            shape=shape,
            fill_fraction=config.get(
                "fill_fraction",
                0.90,
            ),
            background=config.get(
                "background",
                1.0,
            ),
            foreground=config.get(
                "foreground",
                0.0,
            ),
            pixel_size=config.get(
                "pixel_size",
                None,
            ),
        )

    if kind == "diffraction":

        return DiffractionDetector(
            shape=shape,
            q_max=config.get(
                "q_max",
                1.0,
            ),
            beam_center=config.get(
                "beam_center",
                None,
            ),
            log_scale=config.get(
                "log_scale",
                True,
            ),
            epsilon=config.get(
                "epsilon",
                1e-8,
            ),
        )

    raise ValueError(
        "detector kind must be 'image' or 'diffraction'."
    )
