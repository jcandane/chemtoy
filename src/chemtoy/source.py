# src/chemtoy/source.py

"""
Molecular source resolution for ChemToy.

This module handles where atomic structures come from.

Supported sources
-----------------
1. PDB / RCSB

    source:
      kind: pdb
      id: 1TIM

2. PubChem

    source:
      kind: pubchem
      cid: 2244
      record_type: 3d

3. Local file

    source:
      kind: file
      path: data/molecules/caffeine.xyz

The output of this module is always a chemtoy.structure.Structure.

Important design rule
---------------------
source.py owns downloading and caching.

structure.py owns parsing and atom access through MDAnalysis.

experiment.py owns orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .structure import Structure


PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

PUBCHEM_SDF_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
    "{cid}/SDF?record_type={record_type}"
)


@dataclass(slots=True)
class SourceSpec:
    """
    Description of a molecular source.

    Parameters
    ----------
    kind
        Source kind.

        Supported values:

            "pdb"
            "pubchem"
            "file"

    id
        PDB identifier, for example "1TIM".

    cid
        PubChem compound identifier.

    path
        Local molecular file path.

    record_type
        PubChem record type.

        Usually:

            "3d"
            "2d"

    cache_dir
        Root cache directory for downloaded structures.
    """

    kind: str

    id: str | None = None

    cid: str | int | None = None

    path: str | Path | None = None

    record_type: str = "3d"

    cache_dir: str | Path = "data/sources"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "SourceSpec":
        """
        Create a SourceSpec from a dictionary.

        This is intended for YAML-loaded configuration.
        """

        if "kind" not in data:
            raise KeyError("Source config must contain 'kind'.")

        return cls(
            kind=data["kind"],
            id=data.get("id"),
            cid=data.get("cid"),
            path=data.get("path"),
            record_type=data.get("record_type", "3d"),
            cache_dir=data.get("cache_dir", "data/sources"),
        )

    @property
    def cache_root(self) -> Path:
        return Path(self.cache_dir)

    def normalized_kind(self) -> str:
        return self.kind.lower().strip()


# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------


def _download(
    url: str,
    path: Path,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """
    Download a URL into a local file.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists() and not force:
        return path

    response = requests.get(
        url,
        timeout=timeout,
    )

    response.raise_for_status()

    path.write_bytes(
        response.content,
    )

    return path


# ---------------------------------------------------------------------
# PDB / RCSB
# ---------------------------------------------------------------------


def download_pdb(
    pdb_id: str,
    cache_dir: str | Path = "data/sources/pdb",
    force: bool = False,
) -> Path:
    """
    Download a PDB file from RCSB.

    Parameters
    ----------
    pdb_id
        Four-character PDB identifier.

    cache_dir
        Directory where downloaded PDB files are cached.

    force
        Redownload even if the file already exists.

    Returns
    -------
    Path
        Local path to the cached PDB file.
    """

    pdb_id = pdb_id.upper().strip()

    if not pdb_id:
        raise ValueError("pdb_id must be non-empty.")

    cache_dir = Path(cache_dir)

    path = cache_dir / f"{pdb_id}.pdb"

    url = PDB_DOWNLOAD_URL.format(
        pdb_id=pdb_id,
    )

    return _download(
        url=url,
        path=path,
        force=force,
    )


# ---------------------------------------------------------------------
# PubChem
# ---------------------------------------------------------------------


def download_pubchem_sdf(
    cid: str | int,
    cache_dir: str | Path = "data/sources/pubchem",
    record_type: str = "3d",
    force: bool = False,
) -> Path:
    """
    Download an SDF file from PubChem.

    Parameters
    ----------
    cid
        PubChem compound identifier.

    cache_dir
        Directory where downloaded PubChem structures are cached.

    record_type
        PubChem record type.

        Common values:

            "3d"
            "2d"

    force
        Redownload even if the file already exists.

    Returns
    -------
    Path
        Local path to the cached SDF file.
    """

    cid = str(cid).strip()

    if not cid:
        raise ValueError("cid must be non-empty.")

    record_type = record_type.lower().strip()

    if record_type not in {"2d", "3d"}:
        raise ValueError("record_type must be '2d' or '3d'.")

    cache_dir = Path(cache_dir)

    path = cache_dir / f"CID_{cid}_{record_type}.sdf"

    url = PUBCHEM_SDF_URL.format(
        cid=cid,
        record_type=record_type,
    )

    return _download(
        url=url,
        path=path,
        force=force,
    )


# ---------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------


def resolve_file(
    path: str | Path,
) -> Path:
    """
    Validate and resolve a local molecular file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    return path.resolve()


# ---------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------


def resolve_source(
    spec: SourceSpec | dict[str, Any],
    force: bool = False,
) -> Path:
    """
    Resolve a source specification into a local molecular file.

    Parameters
    ----------
    spec
        SourceSpec or dictionary.

    force
        Redownload remote sources even if cached.

    Returns
    -------
    Path
        Local molecular file.
    """

    if isinstance(spec, dict):
        spec = SourceSpec.from_dict(spec)

    kind = spec.normalized_kind()

    cache_root = spec.cache_root

    if kind == "pdb":

        if spec.id is None:
            raise ValueError("PDB source requires 'id'.")

        return download_pdb(
            pdb_id=spec.id,
            cache_dir=cache_root / "pdb",
            force=force,
        )

    if kind == "pubchem":

        if spec.cid is None:
            raise ValueError("PubChem source requires 'cid'.")

        return download_pubchem_sdf(
            cid=spec.cid,
            cache_dir=cache_root / "pubchem",
            record_type=spec.record_type,
            force=force,
        )

    if kind == "file":

        if spec.path is None:
            raise ValueError("File source requires 'path'.")

        return resolve_file(
            spec.path,
        )

    raise ValueError(
        "source kind must be one of: 'pdb', 'pubchem', 'file'."
    )


def load_structure(
    spec: SourceSpec | dict[str, Any],
    force: bool = False,
) -> Structure:
    """
    Resolve a source and load it as a Structure.

    Parameters
    ----------
    spec
        SourceSpec or dictionary.

    force
        Redownload remote sources even if cached.

    Returns
    -------
    Structure
        MDAnalysis-backed atomic structure.
    """

    path = resolve_source(
        spec,
        force=force,
    )

    return Structure.from_file(path)


# ---------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------


def load_pdb(
    pdb_id: str,
    cache_dir: str | Path = "data/sources",
    force: bool = False,
) -> Structure:
    """
    Download and load a PDB structure.
    """

    return load_structure(
        SourceSpec(
            kind="pdb",
            id=pdb_id,
            cache_dir=cache_dir,
        ),
        force=force,
    )


def load_pubchem(
    cid: str | int,
    cache_dir: str | Path = "data/sources",
    record_type: str = "3d",
    force: bool = False,
) -> Structure:
    """
    Download and load a PubChem compound.
    """

    return load_structure(
        SourceSpec(
            kind="pubchem",
            cid=cid,
            record_type=record_type,
            cache_dir=cache_dir,
        ),
        force=force,
    )


def load_file(
    path: str | Path,
) -> Structure:
    """
    Load a local molecular file.
    """

    return load_structure(
        SourceSpec(
            kind="file",
            path=path,
        )
    )
