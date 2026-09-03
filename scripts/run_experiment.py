# scripts/run_experiment.py

"""
Command-line entry point for running ChemToy experiments.

This script is intended for both local use and GitHub Actions.

Example
-------
python scripts/run_experiment.py \
    --source-kind pdb \
    --pdb-id 1TIM \
    --n 64 \
    --pixels 128 \
    --seed 0 \
    --sigma 1.25 \
    --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chemtoy.experiment import run_experiment


def build_config(args: argparse.Namespace) -> dict:
    """
    Build a ChemToy experiment config dictionary from CLI arguments.
    """

    source_kind = args.source_kind.lower().strip()

    if source_kind == "pdb":
        source = {
            "kind": "pdb",
            "id": args.pdb_id,
        }

        name = f"quick_pdb_{args.pdb_id.upper()}"

    elif source_kind == "pubchem":
        source = {
            "kind": "pubchem",
            "cid": args.pubchem_cid,
            "record_type": args.pubchem_record_type,
        }

        name = f"quick_pubchem_CID_{args.pubchem_cid}"

    elif source_kind == "file":
        source = {
            "kind": "file",
            "path": args.file_path,
        }

        name = f"quick_file_{Path(args.file_path).stem}"

    else:
        raise ValueError(
            "--source-kind must be one of: pdb, pubchem, file"
        )

    return {
        "experiment": {
            "name": name,
            "output_dir": args.output_dir,
        },
        "source": source,
        "sampling": {
            "n": args.n,
            "mode": args.sampling_mode,
            "seed": args.seed,
            "psi": args.psi,
        },
        "detector": {
            "kind": "image",
            "shape": [
                args.pixels,
                args.pixels,
            ],
            "fill_fraction": args.fill_fraction,
        },
        "projection": {
            "sigma": args.sigma,
            "truncate": args.truncate,
            "normalize": True,
        },
        "visualization": {
            "montage_max_images": args.montage_max_images,
            "montage_columns": args.montage_columns,
            "save_individual": True,
            "individual_max_images": args.individual_max_images,
            "normalize": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run a quick ChemToy toy CryoEM experiment."
    )

    parser.add_argument(
        "--source-kind",
        choices=["pdb", "pubchem", "file"],
        default="pdb",
        help="Structure source kind.",
    )

    parser.add_argument(
        "--pdb-id",
        default="1TIM",
        help="PDB ID used when --source-kind pdb.",
    )

    parser.add_argument(
        "--pubchem-cid",
        default="2244",
        help="PubChem CID used when --source-kind pubchem.",
    )

    parser.add_argument(
        "--pubchem-record-type",
        choices=["2d", "3d"],
        default="3d",
        help="PubChem record type.",
    )

    parser.add_argument(
        "--file-path",
        default="",
        help="Local molecular file used when --source-kind file.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=64,
        help="Number of projection images.",
    )

    parser.add_argument(
        "--pixels",
        type=int,
        default=128,
        help="Image width and height in pixels.",
    )

    parser.add_argument(
        "--sampling-mode",
        choices=["so3", "s2", "s2_fixed_twist"],
        default="so3",
        help="Orientation sampling mode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--psi",
        type=float,
        default=0.0,
        help="Fixed in-plane angle for S2 sampling.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.25,
        help="Gaussian atom width in pixels.",
    )

    parser.add_argument(
        "--truncate",
        type=float,
        default=4.0,
        help="Gaussian kernel truncation radius in sigma units.",
    )

    parser.add_argument(
        "--fill-fraction",
        type=float,
        default=0.90,
        help="Fraction of detector occupied by the projected molecule.",
    )

    parser.add_argument(
        "--montage-max-images",
        type=int,
        default=36,
        help="Maximum number of images in montage.",
    )

    parser.add_argument(
        "--montage-columns",
        type=int,
        default=6,
        help="Number of columns in montage.",
    )

    parser.add_argument(
        "--individual-max-images",
        type=int,
        default=16,
        help="Maximum number of individual PNGs to save.",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the CLI.
    """

    args = parse_args()

    config = build_config(args)

    result = run_experiment(config)

    print("ChemToy experiment complete.")
    print(f"Output directory: {result.output_dir}")
    print(f"Images shape: {result.images.shape}")
    print(f"Flat dataset shape: {result.images_flat.shape}")
    print(f"Montage: {result.output_dir / 'montage.png'}")
    print(f"Flat NumPy dataset: {result.output_dir / 'images_flat.npy'}")


if __name__ == "__main__":
    main()
