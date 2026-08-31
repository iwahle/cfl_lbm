<h1 align="center">Unsupervised learning of mapping between brain lesions and behavior</h1>
<p align="center">
<img src=resources/cover.png width="600" />
</p>


Causal Feature Learning (CFL) is an unsupervised algorithm designed to construct macro-variables from low-level data, preserving the causal relationships present in the data. In this repository, CFL is benchmarked against Canonical Correlations Analysis (CCA) using a synthetic dataset, and is applied to human brain lesion data and corresponding responses to language, visuospatial, and depression assessments (as described in the associated preprint). This code depends on the [CFL software package](https://github.com/eberharf/cfl) which can be installed via pip.


Each `src/` subdirectory is a self-contained analysis that produces one or more figures for the paper. `figure_reference.md` maps every figure/panel to the script that generates it.

## Setup

```bash
conda env create -f cfl-lbm-env.yml
conda activate cfl-lbm
pip install -e .
```

Requires Python 3.8. The core dependency is [`cfl`](https://pypi.org/project/cfl/), which provides the `Experiment` class and CFL pipeline blocks (`CondDensityEstimator`, `CauseClusterer`) used throughout.

## Running an analysis

Every analysis module is run as a script from the repo root:

```bash
python -m src.lv_cfl.run_cfl
python -m src.dep_cfl.run_cfl --exp_id 0 --plot_order 1 3 4 0 2
```

- `--exp_id -1` (default) trains a fresh CFL `Experiment`. A non-negative `--exp_id` reloads a previously trained experiment's saved results instead of retraining — use this to iterate on plots without rerunning training.
- `--plot_order` remaps cluster indices for figure display only.
- Data-processing scripts (`src/data_processing/**/format_*.py`) must be run first to produce the `.npy` files each analysis depends on.

## Data layout

`data/`, `results/`, and `figures/` are gitignored and generated locally:

- `data/<dataset>/` — preprocessed lesion masks (`X.npy`), deficit scores (`Y.npy`), demographics, etiology, and train/test splits. Datasets: `cohort1` (language/visuospatial deficits), `cohort2` (depression questionnaire deficits), `simulated`/`simulated_schaefer200` (synthetic lesions with known ground-truth parcels).
- `results/<module>/cfl_results/` — trained CFL experiment state, reloadable via `--exp_id`.
- `figures/<module>/` — output plots for each analysis module.

Please see the manuscript for data sharing detials. 

## Module naming

- `<name>_cfl/` — primary CFL analysis for a dataset (`cfl_config.py` + `run_cfl.py`).
- `dep_q_vs_*/` — depression-cohort comparisons of CFL clustering against alternative deficit-aggregation strategies.
- `dep_dems/`, `lv_dems/` — cluster-vs-demographics/etiology analyses.
- `cca_comparison/`, `dep_vs_kmeans/`, `loc_comparison_schaefer200/` — baseline-method comparisons against CFL.
- `dep_*` = cohort2 (depression), `lv_*` = cohort1 (language/visuospatial).

Shared code lives in `src/util/` (paths, constants, data loading, significance testing, spatial eval metrics) and `src/vis/` (brain overlays, deficit plots, sankey diagrams).
