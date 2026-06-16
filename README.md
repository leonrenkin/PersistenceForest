# Persistence Forests and Generalized Landscapes

Implementation accompanying the manuscript on persistent cycle progressions and generalized persistence landscapes available at https://doi.org/10.48550/arXiv.2512.09668. The code is meant to be usable by anyone interested in experimenting with the algorithm and reproducing figures/benchmarks from the manuscript.

## What this repo provides
- `PersistenceForest` (primary entry point) builds the forest of optimal cycles for an alpha complex, together with barcodes and cycle representatives over the filtration.
- Plotting and animation methods for cycle representatives and barcodes in codimension 1.
- Generalized persistence landscapes including vectorisation functions cycles like length, enclosed area and excess curvature.
- End-to-end example in `pers_forest_example.py` showing forest construction, plotting, landscape computation, and simple ML vectorisation.
- Example for generating animations in `animation_tutorial.ipynb`.
- Benchmark tooling in `benchmark.py` to reproduce runtime plots reported in the paper.
- Paper figure notebook `paper-examples.ipy` was used to generate manuscript graphics.

Legacy note: `LoopForest.py` is an older version kept for reference and is no longer part of the workflow.

## Installation
Tested with Python 3.13.3.
```bash
git clone https://github.com/leonrenkin/PersistenceForest.git
cd LoopForest
pip install .
```
Optional extras:
```bash
# Plotly-based 2D/3D interactive plotting
pip install ".[plotly]"

# Notebook inline rendering (MIME/widget renderers)
pip install ".[notebook]"

# GIF export via Pillow
pip install ".[animation]"
```

## Quickstart
```python
import numpy as np
import matplotlib.pyplot as plt
from loopforest import PersistenceForest
from loopforest.cycle_rep_vectorisation import signed_chain_edge_length

# 1) Create a point cloud
rng = np.random.default_rng(0)
pts = rng.random((300, 2))

# 2) Build the persistence forest (alpha complex)
forest = PersistenceForest(pts, print_info=True)

# 3) Visualize
forest.plot_barcode(min_bar_length=0.01, coloring="forest")
forest.plot_at_filtration(0.1)

# 4) Generalized landscapes
forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_edge_length,
    max_k=5,
    num_grid_points=512,
    label="edge-length",
)
forest.plot_landscape_family(label="edge-length")
plt.show()
```
Run the richer demo (plots + simple ML vectorisation) with:
```bash
python pers_forest_example.py
```

## Generalized landscapes and vectorisation
- Define cycle functionals in `cycle_rep_vectorisations.py` (examples: edge length, area, connected components, signed/unsigned variants).
- `forest.compute_generalized_landscape_family(...)` builds families for one functional; `plot_landscape_comparison_between_functionals` contrasts multiple labels.
- `MultiLandscapeVectorizer` turns multiple forests into fixed-length feature vectors (optionally with basic stats) for downstream ML models.

## Repository guide
- `PersistenceForest.py` – forest construction, barcodes, plotting wrappers, generalized landscapes.
- `forest_plotting.py` – shared plotting/animation utilities.
- `forest_landscapes.py` – landscape computation and visualisation.
- `cycle_rep_vectorisations.py` – cycle functionals and vectoriser.
- `color_scheme.py` – consistent color palettes across plots.
- `pers_forest_example.py` – main usage example.
- `animation_tutorial.ipynb` – animation example. 
- `benchmark.py` – runtime benchmarks.
- `paper-examples.ipy`, `generalized_landscape_plots/`, `paper_figures/` – scripts/notebooks for paper figures.
- `point_cloud_sampling.py`, `point_cloud_generator.py` – synthetic data utilities.
- `LoopForest.py` – deprecated predecessor, kept only for historical reference.

## Notes
- Animations require a working Matplotlib animation backend (Pillow or ffmpeg).
