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
# Validation helpers
# ---------------------------------------------------------------------


def _validate_shape(
    shape: tuple[int, int] | list[int],
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
        raise ValueError("shape must have length 2: (height, width).")

    height = int(shape[0])
    width = int(shape[1])

    if height <= 0 or width <= 0:
        raise ValueError("detector height and width must be positive.")

    return height, width


def _validate_fill_fraction(
    fill_fraction: float,
) -> float:
    """
    Validate object fill fraction.
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


def _centered_axis(
    size: int,
    spacing: float = 1.0,
) -> Array:
    """
    Return a centered detector coordinate axis.

    Examples
    --------
    size = 5, spacing = 1

        [-2, -1, 0, 1, 2]

    size = 4, spacing = 1

        [-1.5, -0.5, 0.5, 1.5]
    """

    center = (size - 1) / 2.0

    axis = np.arange(
        size,
        dtype=np.float32,
    )

    return (
        axis
        - np.float32(center)
    ) * np.float32(spacing)


# ---------------------------------------------------------------------
# Base detector
# ---------------------------------------------------------------------


@dataclass(slots=True)
class Detector2D:
    """
    Base class for 2D detectors.

    Parameters
    ----------
    shape
        Detector shape as (height, width).

    Notes
    -----
    This class only knows about a rectangular 2D grid.
    Subclasses add semantic meaning to the grid.
    """

    shape: tuple[int, int] | list[int] = (128, 128)

    kind: str = "detector"

    def __post_init__(self) -> None:
        """
        Validate base detector fields.

        Do not call Detector2D.__post_init__ from inside this method.
        Child classes may call this method explicitly.
        """

        self.shape = _validate_shape(
            self.shape,
        )

    @property
    def height(self) -> int:
        """
        Detector height in pixels.
        """

        return self.shape[0]

    @property
    def width(self) -> int:
        """
        Detector width in pixels.
        """

        return self.shape[1]

    @property
    def center(self) -> Array:
        """
        Detector center in pixel coordinates.

        Returns
        -------
        ndarray
            Array [x_center, y_center].
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

        x = np.arange(
            self.width,
            dtype=np.float32,
        )

        y = np.arange(
            self.height,
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
        Allocate a detector array filled with zeros.
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
        Allocate a detector array filled with ones.
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
        Test whether pixel coordinates are inside the detector.

        Parameters
        ----------
        xy
            Array of shape (N, 2), with columns x, y.

        Returns
        -------
        ndarray
            Boolean mask of shape (N,).
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
# Image detector
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
        Background image value.

    foreground
        Foreground image value.

    pixel_size
        Optional real-space pixel size.

        For now, this is metadata only. The toy projector uses automatic
        fit-to-detector scaling unless project.py later chooses otherwise.
    """

    shape: tuple[int, int] | list[int] = (128, 128)

    fill_fraction: float = 0.90

    background: float = 1.0

    foreground: float = 0.0

    pixel_size: float | None = None

    kind: str = "image"

    def __post_init__(self) -> None:
        """
        Validate image detector fields.
        """

        Detector2D.__post_init__(
            self,
        )

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
            min(self.height, self.width)
            * 0.5
            * self.fill_fraction
        )

    def coordinate_grid(self) -> tuple[Array, Array]:
        """
        Centered real-space detector coordinates.

        Returns
        -------
        xx, yy
            Arrays of shape (height, width).

        Notes
        -----
        If pixel_size is None, one pixel corresponds to one coordinate
        unit.
        """

        spacing = (
            1.0
            if self.pixel_size is None
            else self.pixel_size
        )

        x = _centered_axis(
            self.width,
            spacing,
        )

        y = _centered_axis(
            self.height,
            spacing,
        )

        xx, yy = np.meshgrid(
            x,
            y,
            indexing="xy",
        )

        return xx, yy

    def fit_scale(
        self,
        xy: Array,
    ) -> float:
        """
        Compute a scale factor so projected coordinates fit in the image.

        Parameters
        ----------
        xy
            Centered 2D projected coordinates with shape (N, 2).

        Returns
        -------
        float
            Coordinate-to-pixel scale factor.
        """

        xy = np.asarray(
            xy,
            dtype=np.float32,
        )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2).")

        if xy.shape[0] == 0:
            return 1.0

        radius = float(
            np.linalg.norm(
                xy,
                axis=1,
            ).max()
        )

        if radius <= 1e-12:
            return 1.0

        return self.image_radius / radius

    def world_to_pixel(
        self,
        xy: Array,
        scale: float,
    ) -> Array:
        """
        Convert centered 2D coordinates into pixel coordinates.

        Parameters
        ----------
        xy
            Array of shape (N, 2), centered around zero.

        scale
            Multiplicative coordinate-to-pixel scale.

        Returns
        -------
        ndarray
            Pixel coordinates of shape (N, 2), with columns x, y.
        """

        xy = np.asarray(
            xy,
            dtype=np.float32,
        )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2).")

        return (
            xy * np.float32(scale)
            + self.center[None, :]
        ).astype(np.float32)


# ---------------------------------------------------------------------
# Diffraction detector
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

    beam_center
        Optional beam center as (x, y). If None, detector center is used.

    log_scale
        Whether visualization should prefer log intensity.

    epsilon
        Small positive value used in safe log transforms.

    Notes
    -----
    This class does not compute diffraction. It only defines the
    reciprocal-space grid and associated detector metadata.
    """

    shape: tuple[int, int] | list[int] = (128, 128)

    q_max: float = 1.0

    beam_center: tuple[float, float] | list[float] | None = None

    log_scale: bool = True

    epsilon: float = 1e-8

    kind: str = "diffraction"

    def __post_init__(self) -> None:
        """
        Validate diffraction detector fields.
        """

        Detector2D.__post_init__(
            self,
        )

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
                raise ValueError("beam_center must have length 2: (x, y).")

            self.beam_center = (
                float(self.beam_center[0]),
                float(self.beam_center[1]),
            )

    @property
    def center(self) -> Array:
        """
        Beam center in pixel coordinates.

        Returns
        -------
        ndarray
            Array [x_center, y_center].
        """

        if self.beam_center is None:
            return Detector2D.center.fget(self)

        return np.array(
            self.beam_center,
            dtype=np.float32,
        )

    @property
    def q_grid(self) -> tuple[Array, Array]:
        """
        Reciprocal-space coordinate grid.

        Returns
        -------
        qx, qy
            Arrays of shape (height, width).
        """

        half_size = min(
            self.height,
            self.width,
        ) / 2.0

        q_spacing = self.q_max / half_size

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
    def q_radius_grid(self) -> Array:
        """
        Radial reciprocal-space coordinate grid.
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
        Apply detector-preferred intensity transform.

        This is mostly for future diffraction visualization.
        """

        intensity = np.asarray(
            intensity,
            dtype=np.float32,
        )

        if not self.log_scale:
            return intensity

        return np.log1p(
            intensity + np.float32(self.epsilon)
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

    if config is None:
        config = {}

    kind = str(
        config.get(
            "kind",
            "image",
        )
    ).lower().strip()

    shape = config.get(
        "shape",
        (128, 128),
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
