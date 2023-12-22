
<h1 align="center">Lesion-Behavior Mapping using Causal Feature Learning</h1>
<p align="center">
<img src=readme_graphic.png width="600" />
</p>

### Overview

Causal Feature Learning (CFL) is an unsupervised algorithm designed to construct
macro-variables from low-level data, preserving the causal relationships present
in the data. In this repository, CFL is applied to human brain lesion data and
corresponding responses to language, visuospatial, and depression assessments in
order to identify 1) categories of lesions that are unique in their effects on
test responses and 2) categories of test responses that are unique in their
likelihoods of occurring given any lesion. This code depends on the [CFL
software package](https://github.com/eberharf/cfl) which can be installed via
[pip](https://cfl.readthedocs.io/en/latest/getting_started/SETUP.html).

### Running the code

1. To use this code with your own data, add a new directory within `data` that 
contains the following files:

    - `X.npy`: an n_samples x n_voxels array of vectorized lesion masks
    - `Y.npy`: an n_samples x n_items array of behavioral test responses
    - `dems.npy` : optional, an n_samples x n_demographics array of demographic
      measures to include when running CFL. 

2. Parameters to run CFL with can be modified in `cfl_params.py`. Consult the
   CFL software package
   [documentation](https://cfl.readthedocs.io/en/latest/index.html#) for details
   on setting parameters. Examples of how to set parameters for hyperparameter
   tuning are included in this file as well.

3. Modify `util.load_scale_data` to properly preprocess your specific dataset.

4. `run_cfl.py` and most files in `extended_analyses` take the following arguments:

    - `analysis`: the name of the directory within `data` where your data is 
      stored
    - `include_dem`: if 1, will include the demographic quantities specified
        in `dems.npy`; if 0, will not.

  Set these with flags as needed when running scripts from the command line.

### Included analyses

- `source/run_cfl.py`: fits a CFL model provided cause and effect variable data
- `source/extended_analyses/cluster_questions.py`: clusters question-wise
  responses based on their contributions to defining the effect partition found
  by CFL 
- `source/extended_analyses/compare_aggregates.py`: evaluates candidate
  aggregate BDI quantities based on ability to predict categories found by CFL
- `source/extended_analyses/compare_cca.py`: compares CFL results to CCA results
- `source/extended_analyses/compare_mbdi.py` compares CFL results found when the
  effect is given as responses to the 21 BDI questions versus the mean score
  across questions
- `source/extended_analyses/compare_naive.py`: compares CFL results to those
  found when lesion masks are clustered without regard to the effect


### License and Citations

If you use cfl_lbm in published research work, we encourage you to cite this
repository:

```
Lesion-Behavior Mapping using Causal Feature Learning (2023). https://github.com/iwahle/cfl_lbm
```

or use the BibTex reference:

```
@misc{cfl_lbm2023,
    title     = "Lesion-Behavior Mapping using Causal Feature Learning",
    year      = "2023",
    publisher = "GitHub",
    url       = "https://github.com/iwahle/cfl_lbm"}
```

### Contact

Please reach out to Iman Wahle (imanwahle@gmail.com) with any questions.