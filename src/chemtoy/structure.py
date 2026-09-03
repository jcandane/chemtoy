# src/chemtoy/structure.py

"""
Atomic structure representation for ChemToy.

This module defines the Structure object used by ChemToy experiments.

A Structure is a lightweight wrapper around an MDAnalysis Universe.
It represents a 3D collection of atoms, independent of any particular
experiment type.

Important design rule
---------------------
structure.py knows about atoms.

It does not know about:
- PDB downloads
- PubChem downloads
- detectors
- projection images
- diffraction patterns
- experiments
- visualization

Those responsibilities belong to other modules.

Typical flow
------------
source.py
    resolves PDB / PubChem / local files

structure.py
    loads the local molecular file using MDAnalysis

project.py
    uses the coordinates to generate toy images or diffraction patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import MDAnalysis as mda


Array = np.ndarray


@dataclass(slots=True)
class Structure:
    """
    Atomic structure backed by MDAnalysis.

    Parameters
    ----------
    path
        Local molecular structure file.

    name
        Optional human-readable name. If omitted, the filename stem is used.

    Notes
    -----
    This object is deliberately small. It provides a uniform interface
    for coordinates and atom metadata, regardless of whether the original
    source was PDB, PubChem, or an XYZ file.
    """

    path: str | Path

    name: str | None = None

    _universe: mda.Universe = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.path = Path(self.path)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        if not self.path.is_file():
            raise ValueError(f"Expected a file, got: {self.path}")

        if self.name is None:
            self.name = self.path.stem

        self._universe = mda.Universe(
            str(self.path),
        )

    # -----------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        name: str | None = None,
    ) -> "Structure":
        """
        Load a Structure from a local molecular file.

        Examples
        --------
        Structure.from_file("data/sources/pdb/1TIM.pdb")

        Structure.from_file("data/molecules/caffeine.xyz")
        """

        return cls(
            path=path,
            name=name,
        )

    # -----------------------------------------------------------------
    # MDAnalysis access
    # -----------------------------------------------------------------

    @property
    def universe(self) -> mda.Universe:
        """
        Underlying MDAnalysis Universe.
        """

        return self._universe

    @property
    def atoms(self):
        """
        MDAnalysis AtomGroup containing all atoms.
        """

        return self.universe.atoms

    # -----------------------------------------------------------------
    # Basic metadata
    # -----------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        """
        Number of atoms.
        """

        return len(self.atoms)

    @property
    def n_residues(self) -> int:
        """
        Number of residues, if available.

        For small molecules or XYZ files, this may simply be 1.
        """

        try:
            return len(self.universe.residues)
        except Exception:
            return 0

    @property
    def n_segments(self) -> int:
        """
        Number of segments/chains, if available.
        """

        try:
            return len(self.universe.segments)
        except Exception:
            return 0

    @property
    def dimensions(self) -> Array | None:
        """
        Unit cell or box dimensions, if present.

        Returns None if unavailable.
        """

        dims = self.universe.dimensions

        if dims is None:
            return None

        return np.asarray(
            dims,
            dtype=np.float32,
        )

    # -----------------------------------------------------------------
    # Coordinates
    # -----------------------------------------------------------------

    @property
    def coordinates(self) -> Array:
        """
        Atomic coordinates.

        Returns
        -------
        ndarray
            Array of shape (N, 3), dtype float32.
        """

        coords = np.asarray(
            self.atoms.positions,
            dtype=np.float32,
        )

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "Expected coordinates with shape (N, 3)."
            )

        return coords.copy()

    @property
    def center_of_geometry(self) -> Array:
        """
        Geometric center of all atoms.
        """

        if self.n_atoms == 0:
            raise ValueError("Cannot center an empty structure.")

        return np.asarray(
            self.coordinates.mean(axis=0),
            dtype=np.float32,
        )

    @property
    def center_of_mass(self) -> Array:
        """
        Mass-weighted center of all atoms.

        Falls back to center_of_geometry if masses are unavailable.
        """

        masses = self.masses

        if masses is None:
            return self.center_of_geometry

        masses = np.asarray(
            masses,
            dtype=np.float32,
        )

        if masses.shape[0] != self.n_atoms:
            return self.center_of_geometry

        total_mass = float(masses.sum())

        if total_mass <= 0.0:
            return self.center_of_geometry

        coords = self.coordinates

        return np.asarray(
            (coords * masses[:, None]).sum(axis=0) / total_mass,
            dtype=np.float32,
        )

    @property
    def centered_coordinates(self) -> Array:
        """
        Coordinates centered at the geometric center.

        This is the coordinate array most projection code should use.
        """

        return (
            self.coordinates
            - self.center_of_geometry[None, :]
        ).astype(np.float32)

    @property
    def com_coordinates(self) -> Array:
        """
        Coordinates centered at the center of mass.
        """

        return (
            self.coordinates
            - self.center_of_mass[None, :]
        ).astype(np.float32)

    # -----------------------------------------------------------------
    # Atom metadata
    # -----------------------------------------------------------------

    @property
    def atom_names(self) -> Array | None:
        """
        Atom names, if available.
        """

        try:
            return np.asarray(self.atoms.names)
        except Exception:
            return None

    @property
    def elements(self) -> Array | None:
        """
        Atomic element symbols, if available.

        Some file formats, especially simple XYZ files, may expose
        elements directly. Some PDB files may not.
        """

        try:
            elements = np.asarray(self.atoms.elements)

            if len(elements) == self.n_atoms:
                return elements

        except Exception:
            pass

        return self._infer_elements_from_names()

    @property
    def masses(self) -> Array | None:
        """
        Atomic masses, if available.
        """

        try:
            masses = np.asarray(
                self.atoms.masses,
                dtype=np.float32,
            )

            if len(masses) == self.n_atoms:
                return masses

        except Exception:
            pass

        return None

    @property
    def residue_names(self) -> Array | None:
        """
        Residue names, if available.
        """

        try:
            return np.asarray(self.atoms.resnames)
        except Exception:
            return None

    @property
    def residue_ids(self) -> Array | None:
        """
        Residue IDs, if available.
        """

        try:
            return np.asarray(self.atoms.resids)
        except Exception:
            return None

    @property
    def chain_ids(self) -> Array | None:
        """
        Chain identifiers, if available.
        """

        try:
            return np.asarray(self.atoms.chainIDs)
        except Exception:
            pass

        try:
            return np.asarray(self.atoms.segids)
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------

    @property
    def bounding_box(self) -> Array:
        """
        Axis-aligned bounding box.

        Returns
        -------
        ndarray
            Array with shape (2, 3):

                [[xmin, ymin, zmin],
                 [xmax, ymax, zmax]]
        """

        coords = self.coordinates

        return np.stack(
            [
                coords.min(axis=0),
                coords.max(axis=0),
            ],
            axis=0,
        ).astype(np.float32)

    @property
    def radius(self) -> float:
        """
        Maximum distance from the geometric center.
        """

        coords = self.centered_coordinates

        return float(
            np.linalg.norm(
                coords,
                axis=1,
            ).max()
        )

    @property
    def extent(self) -> Array:
        """
        Width of the structure along x, y, z.
        """

        box = self.bounding_box

        return (
            box[1]
            - box[0]
        ).astype(np.float32)

    # -----------------------------------------------------------------
    # Simple operations
    # -----------------------------------------------------------------

    def select(
        self,
        selection: str,
        name: str | None = None,
    ) -> "Structure":
        """
        Return a new Structure from an MDAnalysis selection.

        Examples
        --------
        protein = structure.select("protein")

        backbone = structure.select("backbone")

        ligand = structure.select("resname ATP")

        Notes
        -----
        This writes no files. It creates a new in-memory MDAnalysis
        Universe using the selected atoms.
        """

        atoms = self.atoms.select_atoms(selection)

        if len(atoms) == 0:
            raise ValueError(
                f"No atoms matched selection: {selection!r}"
            )

        universe = mda.Merge(atoms)

        obj = object.__new__(Structure)

        obj.path = self.path
        obj.name = name or f"{self.name}:{selection}"
        obj._universe = universe

        return obj

    def copy(
        self,
        name: str | None = None,
    ) -> "Structure":
        """
        Return an in-memory copy of the Structure.
        """

        universe = mda.Merge(self.atoms)

        obj = object.__new__(Structure)

        obj.path = self.path
        obj.name = name or self.name
        obj._universe = universe

        return obj

    # -----------------------------------------------------------------
    # Conversion
    # -----------------------------------------------------------------

    def to_numpy(
        self,
        centered: bool = True,
        center: str = "geometry",
        dtype=np.float32,
    ) -> Array:
        """
        Return coordinates as a NumPy array.

        Parameters
        ----------
        centered
            If True, return centered coordinates.

        center
            Centering method. Either "geometry" or "mass".

        dtype
            Output dtype.
        """

        if not centered:
            return self.coordinates.astype(dtype)

        center = center.lower().strip()

        if center == "geometry":
            return self.centered_coordinates.astype(dtype)

        if center == "mass":
            return self.com_coordinates.astype(dtype)

        raise ValueError(
            "center must be 'geometry' or 'mass'."
        )

    # -----------------------------------------------------------------
    # Saving and summaries
    # -----------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save atoms to a local structure file.

        MDAnalysis infers the output format from the extension.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.atoms.write(str(path))

        return path

    def summary(self) -> dict:
        """
        Return a lightweight summary dictionary.
        """

        return {
            "name": self.name,
            "path": str(self.path),
            "n_atoms": self.n_atoms,
            "n_residues": self.n_residues,
            "n_segments": self.n_segments,
            "radius": self.radius,
            "extent": self.extent.tolist(),
            "elements_available": self.elements is not None,
            "masses_available": self.masses is not None,
        }

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _infer_elements_from_names(self) -> Array | None:
        """
        Best-effort element inference from atom names.

        This is intentionally conservative. If atom names are unavailable,
        returns None.
        """

        names = self.atom_names

        if names is None:
            return None

        inferred: list[str] = []

        for name in names:

            text = str(name).strip()

            if not text:
                inferred.append("")
                continue

            # Remove leading digits, common in some atom-name fields.
            text = text.lstrip("0123456789")

            if not text:
                inferred.append("")
                continue

            # First letter uppercase, optional second lowercase.
            if len(text) >= 2 and text[1].islower():
                element = text[:2]
            else:
                element = text[0]

            inferred.append(
                element.capitalize()
            )

        return np.asarray(inferred)

    # -----------------------------------------------------------------
    # Python protocol
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_atoms

    def __repr__(self) -> str:
        return (
            "Structure("
            f"name={self.name!r}, "
            f"n_atoms={self.n_atoms}, "
            f"path={str(self.path)!r}"
            ")"
        )

    __str__ = __repr__
