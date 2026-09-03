# src/chemtoy/experiment.py

"""
Experiment orchestration for ChemToy.

This module connects the core pieces of ChemToy:

    source.py
        loads a molecular structure

    sampling.py
        samples orientation angles

    detector.py
        defines the output detector grid

    project.py
        generates simulated measurements

    visualize.py
        saves PNGs and NumPy arrays

The goal of experiment.py is convenience and reproducibility.

It should not contain projection physics itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import yaml

from .source import load_structure
from .sampling import sample_angles
from .detector import make_detector, ImageDetector, DiffractionDetector
from .project import project
from .visualize import (
    save_image_stack,
    save_flat_dataset,
    save_montage,
    save_image_series,
)


Array = np.ndarray


# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load an experiment configuration from YAML.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError("Experiment config must be a dictionary.")

    return config


def save_json(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Save a dictionary as JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    return path


def _as_shape(
    value,
    default: tuple[int, int] = (128, 128),
) -> tuple[int, int]:
    """
    Normalize a shape-like value into (height, width).
    """

    if value is None:
        return default

    if len(value) != 2:
        raise ValueError("shape must have two entries: [height, width].")

    return (
        int(value[0]),
        int(value[1]),
    )


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentConfig:
    """
    Normalized ChemToy experiment configuration.

    This dataclass is intentionally small. The raw YAML dictionary is
    still saved into metadata for reproducibility.
    """

    name: str = "chemtoy_experiment"

    output_dir: str | Path = "outputs"

    source: dict[str, Any] | None = None

    sampling: dict[str, Any] | None = None

    detector: dict[str, Any] | None = None

    projection: dict[str, Any] | None = None

    visualization: dict[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
    ) -> "ExperimentConfig":
        """
        Build an ExperimentConfig from a YAML-style dictionary.
        """

        experiment = config.get(
            "experiment",
            {},
        )

        return cls(
            name=experiment.get(
                "name",
                config.get("name", "chemtoy_experiment"),
            ),
            output_dir=experiment.get(
                "output_dir",
                config.get("output_dir", "outputs"),
            ),
            source=config.get("source"),
            sampling=config.get("sampling", {}),
            detector=config.get("detector", {}),
            projection=config.get("projection", {}),
            visualization=config.get("visualization", {}),
        )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.name


# ---------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentResult:
    """
    Result of a ChemToy experiment.
    """

    images: Array

    images_flat: Array

    angles: Array

    output_dir: Path

    metadata: dict[str, Any]


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


@dataclass(slots=True)
class Experiment:
    """
    YAML-driven ChemToy experiment.

    Example
    -------
    experiment = Experiment.from_yaml("configs/quick_pdb.yml")

    result = experiment.run()
    """

    config: ExperimentConfig

    raw_config: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
    ) -> "Experiment":
        """
        Create an Experiment from a dictionary.
        """

        return cls(
            config=ExperimentConfig.from_dict(config),
            raw_config=config,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "Experiment":
        """
        Create an Experiment from a YAML file.
        """

        config = load_config(path)

        return cls.from_dict(config)

    def run(
        self,
        force_download: bool = False,
    ) -> ExperimentResult:
        """
        Run the experiment.

        Parameters
        ----------
        force_download
            Redownload remote sources even if cached.

        Returns
        -------
        ExperimentResult
        """

        cfg = self.config

        output_dir = cfg.output_path

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if cfg.source is None:
            raise ValueError("Experiment config must contain a source section.")

        # ------------------------------------------------------------
        # Load molecular structure
        # ------------------------------------------------------------

        structure = load_structure(
            cfg.source,
            force=force_download,
        )

        # ------------------------------------------------------------
        # Sample orientations
        # ------------------------------------------------------------

        sampling_cfg = cfg.sampling or {}

        n = int(
            sampling_cfg.get(
                "n",
                sampling_cfg.get("samples", 16),
            )
        )

        mode = sampling_cfg.get(
            "mode",
            "so3",
        )

        psi = float(
            sampling_cfg.get(
                "psi",
                0.0,
            )
        )

        seed = sampling_cfg.get(
            "seed",
            None,
        )

        angles = sample_angles(
            n=n,
            mode=mode,
            psi=psi,
            rng=seed,
        )

        # ------------------------------------------------------------
        # Detector
        # ------------------------------------------------------------

        detector_cfg = cfg.detector or {
            "kind": "image",
            "shape": [128, 128],
        }

        detector = make_detector(
            detector_cfg,
        )

        # ------------------------------------------------------------
        # Projection
        # ------------------------------------------------------------

        projection_cfg = cfg.projection or {}

        sigma = float(
            projection_cfg.get(
                "sigma",
                1.25,
            )
        )

        truncate = float(
            projection_cfg.get(
                "truncate",
                4.0,
            )
        )

        normalize = bool(
            projection_cfg.get(
                "normalize",
                True,
            )
        )

        images = project(
            structure=structure,
            detector=detector,
            angles=angles,
            sigma=sigma,
            truncate=truncate,
            normalize=normalize,
        )

        if images.ndim != 3:
            raise ValueError(
                "This first experiment implementation expects image output "
                "with shape (N, H, W)."
            )

        images_flat = images.reshape(
            images.shape[0],
            -1,
        ).astype(np.float32)

        # ------------------------------------------------------------
        # Save arrays
        # ------------------------------------------------------------

        save_image_stack(
            images,
            output_dir / "images.npy",
        )

        save_flat_dataset(
            images,
            output_dir / "images_flat.npy",
        )

        np.save(
            output_dir / "angles.npy",
            angles.astype(np.float32),
        )

        # ------------------------------------------------------------
        # Visualization
        # ------------------------------------------------------------

        visualization_cfg = cfg.visualization or {}

        montage_max_images = int(
            visualization_cfg.get(
                "montage_max_images",
                36,
            )
        )

        montage_columns = int(
            visualization_cfg.get(
                "montage_columns",
                6,
            )
        )

        save_individual = bool(
            visualization_cfg.get(
                "save_individual",
                True,
            )
        )

        individual_max_images = int(
            visualization_cfg.get(
                "individual_max_images",
                min(16, len(images)),
            )
        )

        display_normalize = bool(
            visualization_cfg.get(
                "normalize",
                False,
            )
        )

        save_montage(
            images=images,
            filename=output_dir / "montage.png",
            max_images=montage_max_images,
            columns=montage_columns,
            normalize=display_normalize,
        )

        if save_individual:
            save_image_series(
                images=images,
                directory=output_dir / "pngs",
                max_images=individual_max_images,
                prefix="image",
                normalize=display_normalize,
            )

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        metadata = self.metadata(
            structure=structure,
            detector=detector,
            angles=angles,
            images=images,
            images_flat=images_flat,
        )

        save_json(
            metadata,
            output_dir / "metadata.json",
        )

        return ExperimentResult(
            images=images,
            images_flat=images_flat,
            angles=angles,
            output_dir=output_dir,
            metadata=metadata,
        )

    def metadata(
        self,
        structure,
        detector,
        angles: Array,
        images: Array,
        images_flat: Array,
    ) -> dict[str, Any]:
        """
        Build metadata for a completed run.
        """

        detector_metadata: dict[str, Any] = {
            "kind": getattr(detector, "kind", type(detector).__name__),
            "shape": list(detector.shape),
        }

        if isinstance(detector, ImageDetector):
            detector_metadata.update(
                {
                    "fill_fraction": detector.fill_fraction,
                    "background": detector.background,
                    "foreground": detector.foreground,
                    "pixel_size": detector.pixel_size,
                }
            )

        if isinstance(detector, DiffractionDetector):
            detector_metadata.update(
                {
                    "q_max": detector.q_max,
                    "beam_center": detector.beam_center,
                    "log_scale": detector.log_scale,
                    "epsilon": detector.epsilon,
                }
            )

        return {
            "experiment": {
                "name": self.config.name,
                "output_dir": str(self.config.output_path),
            },
            "source": self.raw_config.get("source"),
            "structure": structure.summary(),
            "sampling": {
                "n": int(len(angles)),
                "mode": (self.config.sampling or {}).get("mode", "so3"),
                "angles_shape": list(angles.shape),
                "angle_columns": ["theta", "phi", "psi"],
            },
            "detector": detector_metadata,
            "projection": {
                "sigma": float((self.config.projection or {}).get("sigma", 1.25)),
                "truncate": float((self.config.projection or {}).get("truncate", 4.0)),
                "normalize": bool((self.config.projection or {}).get("normalize", True)),
            },
            "outputs": {
                "images_shape": list(images.shape),
                "images_flat_shape": list(images_flat.shape),
                "images_npy": "images.npy",
                "images_flat_npy": "images_flat.npy",
                "angles_npy": "angles.npy",
                "montage_png": "montage.png",
                "png_directory": "pngs",
            },
            "raw_config": self.raw_config,
        }


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------


def run_experiment(
    config: str | Path | dict[str, Any],
    force_download: bool = False,
) -> ExperimentResult:
    """
    Run an experiment from a YAML path or dictionary.
    """

    if isinstance(config, (str, Path)):
        experiment = Experiment.from_yaml(config)
    elif isinstance(config, dict):
        experiment = Experiment.from_dict(config)
    else:
        raise TypeError(
            "config must be a YAML path or dictionary."
        )

    return experiment.run(
        force_download=force_download,
    )
