# src/chemtoy/visualize.py

"""
Visualization utilities for ChemToy.

This module converts simulated arrays into PNG images.

It is intentionally separate from project.py.

project.py
    performs numerical projection and returns NumPy arrays.

visualize.py
    plots arrays and saves PNG files.

Supported image shapes
----------------------
Single image:

    (H, W)

Single flattened image:

    (D,)

Image stack:

    (N, H, W)

Flattened image matrix:

    (N, D)

For flattened inputs, the original image shape must be provided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


Array = np.ndarray


# ---------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------


def as_image(
    image: Array,
    shape: tuple[int, int] | None = None,
) -> Array:
    """
    Convert an input array into a single 2D image.

    Parameters
    ----------
    image
        Either a 2D image with shape (H, W), or a flattened image
        with shape (D,).

    shape
        Required when image is flattened.

    Returns
    -------
    ndarray
        2D image with shape (H, W).
    """

    image = np.asarray(image)

    if image.ndim == 2:
        return image.astype(np.float32)

    if image.ndim == 1:

        if shape is None:
            raise ValueError(
                "shape must be provided when image is flattened."
            )

        height, width = shape

        expected = height * width

        if image.size != expected:
            raise ValueError(
                f"Flattened image has size {image.size}, "
                f"but shape {shape} requires {expected} pixels."
            )

        return image.reshape(shape).astype(np.float32)

    raise ValueError(
        "image must have shape (H, W) or (D,)."
    )


def as_image_stack(
    images: Array,
    shape: tuple[int, int] | None = None,
) -> Array:
    """
    Convert an input array into an image stack.

    Parameters
    ----------
    images
        Either:

            (N, H, W)

        or:

            (N, D)

    shape
        Required when images has shape (N, D).

    Returns
    -------
    ndarray
        Image stack with shape (N, H, W).
    """

    images = np.asarray(images)

    if images.ndim == 3:
        return images.astype(np.float32)

    if images.ndim == 2:

        if shape is None:
            raise ValueError(
                "shape must be provided when images are flattened."
            )

        height, width = shape

        expected = height * width

        if images.shape[1] != expected:
            raise ValueError(
                f"Flattened images have D={images.shape[1]}, "
                f"but shape {shape} requires D={expected}."
            )

        return images.reshape(
            images.shape[0],
            height,
            width,
        ).astype(np.float32)

    raise ValueError(
        "images must have shape (N, H, W) or (N, D)."
    )


def flatten_images(
    images: Array,
) -> Array:
    """
    Flatten an image stack from (N, H, W) to (N, D).

    This is the format wanted for manifold learning experiments.
    """

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    if images.ndim != 3:
        raise ValueError(
            "images must have shape (N, H, W)."
        )

    n = images.shape[0]

    return images.reshape(
        n,
        -1,
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------


def normalize_for_display(
    image: Array,
    eps: float = 1e-8,
) -> Array:
    """
    Normalize an image to [0, 1] for display.

    This does not modify the scientific data array.
    It is only for visualization.
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
# Plotting
# ---------------------------------------------------------------------


def plot_image(
    image: Array,
    shape: tuple[int, int] | None = None,
    ax=None,
    title: str | None = None,
    normalize: bool = False,
):
    """
    Plot one image.

    Parameters
    ----------
    image
        Image with shape (H, W), or flattened image with shape (D,).

    shape
        Original image shape, required for flattened input.

    ax
        Optional matplotlib axis.

    title
        Optional plot title.

    normalize
        Normalize image to [0, 1] before plotting.
    """

    image = as_image(
        image,
        shape=shape,
    )

    if normalize:
        image = normalize_for_display(image)

    if ax is None:

        fig, ax = plt.subplots(
            figsize=(4, 4),
        )

    else:

        fig = ax.figure

    ax.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_montage(
    images: Array,
    shape: tuple[int, int] | None = None,
    max_images: int = 36,
    columns: int = 6,
    figsize: tuple[float, float] | None = None,
    normalize: bool = False,
):
    """
    Plot a montage of images.

    Parameters
    ----------
    images
        Image stack with shape (N, H, W), or flattened matrix with
        shape (N, D).

    shape
        Original image shape, required for flattened input.

    max_images
        Maximum number of images to show.

    columns
        Number of montage columns.

    figsize
        Optional matplotlib figure size.

    normalize
        Normalize each image independently for display.
    """

    images = as_image_stack(
        images,
        shape=shape,
    )

    n = min(
        len(images),
        int(max_images),
    )

    columns = int(columns)

    if columns < 1:
        raise ValueError("columns must be positive.")

    rows = int(
        np.ceil(
            n / columns,
        )
    )

    if figsize is None:
        figsize = (
            2.0 * columns,
            2.0 * rows,
        )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=figsize,
    )

    axes = np.asarray(
        axes,
    ).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i in range(n):

        image = images[i]

        if normalize:
            image = normalize_for_display(image)

        axes[i].imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
        )

        axes[i].set_title(
            str(i),
            fontsize=8,
        )

    fig.tight_layout()

    return fig, axes


# ---------------------------------------------------------------------
# Saving PNGs
# ---------------------------------------------------------------------


def save_image(
    image: Array,
    filename: str | Path,
    shape: tuple[int, int] | None = None,
    normalize: bool = False,
) -> Path:
    """
    Save one image as a PNG.
    """

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, _ = plot_image(
        image=image,
        shape=shape,
        normalize=normalize,
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.0,
    )

    plt.close(fig)

    return filename


def save_montage(
    images: Array,
    filename: str | Path,
    shape: tuple[int, int] | None = None,
    max_images: int = 36,
    columns: int = 6,
    normalize: bool = False,
) -> Path:
    """
    Save a montage PNG from an image stack or flattened image matrix.
    """

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, _ = plot_montage(
        images=images,
        shape=shape,
        max_images=max_images,
        columns=columns,
        normalize=normalize,
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def save_image_series(
    images: Array,
    directory: str | Path,
    shape: tuple[int, int] | None = None,
    max_images: int | None = None,
    prefix: str = "image",
    normalize: bool = False,
) -> list[Path]:
    """
    Save individual image PNGs.

    Parameters
    ----------
    images
        Image stack with shape (N, H, W), or flattened image matrix
        with shape (N, D).

    directory
        Output directory.

    shape
        Original image shape, required for flattened input.

    max_images
        Optional maximum number of images to save.

    prefix
        Filename prefix.

    normalize
        Normalize each image for display.
    """

    images = as_image_stack(
        images,
        shape=shape,
    )

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if max_images is None:
        n = len(images)
    else:
        n = min(
            len(images),
            int(max_images),
        )

    paths: list[Path] = []

    for i in range(n):

        filename = directory / f"{prefix}_{i:04d}.png"

        path = save_image(
            image=images[i],
            filename=filename,
            normalize=normalize,
        )

        paths.append(path)

    return paths


# ---------------------------------------------------------------------
# Dataset saving
# ---------------------------------------------------------------------


def save_flat_dataset(
    images: Array,
    filename: str | Path,
) -> Path:
    """
    Save image stack as a flattened NumPy dataset.

    Input
    -----
    images:
        shape (N, H, W)

    Output
    ------
    npy file containing:

        shape (N, D)

    where:

        D = H * W
    """

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    flat = flatten_images(
        images,
    )

    np.save(
        filename,
        flat,
    )

    return filename


def save_image_stack(
    images: Array,
    filename: str | Path,
) -> Path:
    """
    Save image stack as a NumPy array.

    Input shape:

        (N, H, W)
    """

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    if images.ndim != 3:
        raise ValueError(
            "images must have shape (N, H, W)."
        )

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        filename,
        images,
    )

    return filename
