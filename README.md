# ChemToy

ChemToy is a small Python package for generating toy structural-biology datasets from molecular structures.

The initial goal is simple:

> Take a molecular structure, sample many 3D orientations, project each orientation into a 2D toy CryoEM-like image, and save the result as a NumPy array for manifold learning experiments.

The current output is designed for downstream machine learning:

```text
images.npy        # shape (N, H, W)
images_flat.npy   # shape (N, D), where D = H * W
angles.npy        # shape (N, 3), columns are theta, phi, psi
montage.png       # quick visual sanity check
metadata.json     # run metadata
```

This is not yet a physically accurate CryoEM simulator. It is intentionally a lightweight forward model for generating clean synthetic projection images.

---

## What ChemToy does right now

ChemToy currently supports a basic toy CryoEM-style image simulation:

```text
molecular source
    ↓
Structure
    ↓
sample N orientations
    ↓
rotate 3D atoms
    ↓
orthographic projection to 2D
    ↓
draw one Gaussian blob per atom
    ↓
save image stack and flattened dataset
```

Each atom is currently rendered as the same generic Gaussian. There is no CTF, no noise, no element-specific scattering, no solvent, and no detector realism yet.

This is deliberate. The first milestone is simply to see whether the geometry and dataset generation work.

---

## Repository layout

```text
chemtoy/
  pyproject.toml

  src/
    chemtoy/
      __init__.py
      source.py
      structure.py
      sampling.py
      detector.py
      project.py
      visualize.py
      experiment.py

  scripts/
    run_experiment.py

  .github/
    workflows/
      quickpdb.yml

  results/
    ...
```

---

## Core files

### `source.py`

`source.py` is responsible for getting molecular structure files.

It supports three kinds of sources:

```yaml
source:
  kind: pdb
  id: 1TIM
```

```yaml
source:
  kind: pubchem
  cid: 2244
  record_type: 3d
```

```yaml
source:
  kind: file
  path: data/molecules/caffeine.xyz
```

The job of `source.py` is:

```text
PDB ID / PubChem CID / local path
    ↓
download or validate file
    ↓
local molecular file
    ↓
Structure.from_file(...)
```

For PDB structures, it downloads a `.pdb` file from RCSB.

For PubChem molecules, it downloads an `.sdf` file from PubChem.

For local files, such as `.xyz`, it simply checks that the file exists.

`source.py` does not parse molecular structures itself. It only resolves where the file comes from.

---

### `structure.py`

`structure.py` defines the `Structure` object.

A `Structure` is a lightweight wrapper around an MDAnalysis `Universe`.

It represents atoms in 3D:

```python
structure.coordinates
structure.centered_coordinates
structure.elements
structure.n_atoms
structure.radius
structure.summary()
```

The important design rule is:

> `structure.py` knows about atoms, but not experiments.

It does not know about CryoEM, diffraction, detectors, sampling, PNG files, or GitHub Actions.

Everything eventually becomes a `Structure`, regardless of whether the original source came from PDB, PubChem, or a local `.xyz` file.

---

### `sampling.py`

`sampling.py` samples orientation angles.

Each orientation is represented as:

```text
theta, phi, psi
```

where:

```text
theta = polar angle from +z
phi   = azimuth around z
psi   = in-plane twist angle
```

The output is always:

```python
angles.shape == (N, 3)
```

For example:

```python
from chemtoy.sampling import sample_angles

angles = sample_angles(
    n=1500,
    mode="so3",
    rng=0,
)
```

For full SO(3)-style sampling:

```text
theta: sampled through uniform cos(theta)
phi:   uniform in [0, 2π)
psi:   uniform in [0, 2π)
```

For S² viewing-direction sampling with fixed twist:

```python
angles = sample_angles(
    n=1500,
    mode="s2",
    psi=0.0,
    rng=0,
)
```

In this case, `theta` and `phi` define a uniform direction on the sphere, while `psi` is fixed.

---

### `detector.py`

`detector.py` defines the 2D measurement grid.

There are currently two detector objects:

```python
ImageDetector
DiffractionDetector
```

#### `ImageDetector`

`ImageDetector` is used for toy CryoEM-like images.

Example:

```python
from chemtoy.detector import ImageDetector

detector = ImageDetector(
    shape=(128, 128),
    fill_fraction=0.9,
)
```

It stores parameters such as:

```text
shape
height
width
center
fill_fraction
background
foreground
pixel_size
```

It does not render images itself. It only describes the image grid.

#### `DiffractionDetector`

`DiffractionDetector` is included as a placeholder for future diffraction simulations.

Example:

```python
from chemtoy.detector import DiffractionDetector

detector = DiffractionDetector(
    shape=(256, 256),
    q_max=2.0,
)
```

It defines a reciprocal-space grid, but diffraction physics is not implemented yet.

---

### `project.py`

`project.py` performs the numerical forward projection.

For now, it supports toy CryoEM-style image projections.

The main function is:

```python
from chemtoy.project import project

images = project(
    structure=structure,
    detector=detector,
    angles=angles,
)
```

For an image detector, the output is:

```python
images.shape == (N, H, W)
```

Internally, `project.py` does:

```text
Structure.centered_coordinates
    ↓
convert theta, phi, psi to rotation matrix
    ↓
rotate atoms
    ↓
drop z-coordinate
    ↓
fit projected x/y coordinates to detector
    ↓
deposit Gaussian blobs
    ↓
return image stack
```

`project.py` does not save files and does not make plots.

---

### `visualize.py`

`visualize.py` converts image arrays into PNGs.

It accepts either image stacks:

```python
images.shape == (N, H, W)
```

or flattened datasets:

```python
images_flat.shape == (N, D)
```

Main utilities include:

```python
save_montage(...)
save_image(...)
save_image_series(...)
save_image_stack(...)
save_flat_dataset(...)
```

Example:

```python
from chemtoy.visualize import save_montage, save_flat_dataset

save_montage(
    images,
    "results/example/montage.png",
)

save_flat_dataset(
    images,
    "results/example/images_flat.npy",
)
```

The flattened `.npy` file is the main output for manifold learning experiments.

---

### `experiment.py`

`experiment.py` is the orchestration layer.

It connects:

```text
source.py
sampling.py
detector.py
project.py
visualize.py
```

A complete run does:

```text
load structure
sample angles
make detector
project images
save arrays
save montage
save metadata
```

Example:

```python
from chemtoy.experiment import run_experiment

result = run_experiment(
    {
        "experiment": {
            "name": "quick_1tim",
            "output_dir": "results",
        },
        "source": {
            "kind": "pdb",
            "id": "1TIM",
        },
        "sampling": {
            "n": 64,
            "mode": "so3",
            "seed": 0,
        },
        "detector": {
            "kind": "image",
            "shape": [128, 128],
            "fill_fraction": 0.9,
        },
        "projection": {
            "sigma": 1.25,
            "truncate": 4.0,
            "normalize": True,
        },
    }
)

print(result.images.shape)
print(result.images_flat.shape)
print(result.output_dir)
```

---

## Command-line use

The script:

```text
scripts/run_experiment.py
```

runs a quick experiment from command-line arguments.

Example:

```bash
python scripts/run_experiment.py \
  --source-kind pdb \
  --pdb-id 1TIM \
  --n 64 \
  --pixels 128 \
  --seed 0 \
  --sigma 1.25 \
  --output-dir results
```

This produces:

```text
results/quick_pdb_1TIM/
  images.npy
  images_flat.npy
  angles.npy
  montage.png
  metadata.json
  pngs/
    image_0000.png
    image_0001.png
    ...
```

---

## GitHub Actions

The workflow:

```text
.github/workflows/quickpdb.yml
```

runs a quick PDB experiment directly on GitHub Actions.

It can be launched manually from the GitHub Actions tab.

The workflow:

1. checks out the repository
2. installs ChemToy
3. downloads a PDB structure
4. generates toy CryoEM-like projection images
5. saves results into `results/`
6. commits the results back to the repository

This is enough for simple toy samples. No Modal or GPU server is needed yet.

---

## Example output

A successful run creates:

```text
results/quick_pdb_1TIM/images.npy
```

with shape:

```text
(N, H, W)
```

and:

```text
results/quick_pdb_1TIM/images_flat.npy
```

with shape:

```text
(N, D)
```

where:

```text
D = H * W
```

For example, if:

```text
N = 1500
H = 128
W = 128
```

then:

```text
images.npy       shape = (1500, 128, 128)
images_flat.npy  shape = (1500, 16384)
```

The flattened array is the main dataset for manifold learning.

---

## Current limitations

ChemToy is currently a toy model.

The present CryoEM-style image model does not include:

```text
CTF
noise
dose effects
ice/background
solvent
atomic form factors
electron scattering factors
element-dependent intensities
real microscope parameters
motion blur
defocus variation
```

Every atom is currently drawn as the same Gaussian blob.

This is useful for geometry, orientation, projection, and manifold-learning experiments, but not for realistic CryoEM simulation.

---

## Design philosophy

ChemToy separates responsibilities into small files:

```text
source.py
    where atoms come from

structure.py
    what the atoms are

sampling.py
    which orientations to use

detector.py
    what 2D grid records the measurement

project.py
    how atoms become simulated measurements

visualize.py
    how arrays become PNGs

experiment.py
    how a full run is orchestrated
```

This keeps the code easy to extend.

The same `Structure` object should eventually support several experiment types:

```text
CryoEM projection images
X-ray diffraction patterns
electron diffraction patterns
SAXS-like curves
tomography-style tilt series
```

---

## Future additions

Important future features include:

### Better structure handling

Add richer support for:

```text
mmCIF files
SDF files from PubChem
XYZ files
MOL2 files
multi-model structures
hydrogen filtering
atom selections
centering options
mass weighting
element metadata
```

### More realistic CryoEM images

Add:

```text
contrast transfer function, CTF
defocus
amplitude contrast
noise
dose
ice/background
particle translations
random in-plane shifts
variable pixel size
electron scattering factors
```

### Diffraction simulation

Add a real implementation for `DiffractionDetector`.

Future diffraction flow:

```text
Structure
    ↓
sample orientation
    ↓
compute structure factor F(q)
    ↓
intensity = |F(q)|²
    ↓
detector image
```

Eventually this could support:

```text
X-ray diffraction
electron diffraction
powder-like patterns
single-particle diffraction
reciprocal-space masks
log-intensity visualization
```

### Manifold learning utilities

Since the main scientific output is:

```text
images_flat.npy
```

future analysis utilities could include:

```text
PCA
diffusion maps
UMAP
t-SNE
nearest-neighbor graphs
orientation coloring
latent-space visualization
```

### Better experiment configs

Eventually support full YAML experiment files, for example:

```yaml
experiment:
  name: pdb_1tim_1500
  output_dir: results

source:
  kind: pdb
  id: 1TIM

sampling:
  n: 1500
  mode: so3
  seed: 0

detector:
  kind: image
  shape: [128, 128]
  fill_fraction: 0.9

projection:
  sigma: 1.25
  normalize: true

visualization:
  montage_max_images: 36
  save_individual: true
```

### Modal or GPU backend

GitHub Actions is enough for small CPU toy examples.

Later, larger runs could move to Modal or another compute backend for:

```text
large N
larger images
GPU projection
diffraction simulations
batch generation
automatic result commits
```

---

## Project status

ChemToy is currently in the first milestone:

> Generate simple toy CryoEM-like projection images from molecular structures and save them as NumPy arrays for manifold learning.

The code is intentionally simple right now. The priority is to make the end-to-end pipeline work before adding physical realism.
