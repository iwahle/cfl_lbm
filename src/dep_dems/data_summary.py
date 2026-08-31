import os
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *
from src.vis.brain_vis import lesion_heatmap

fig_path = os.path.join(FIG_PATH, 'cohort2_summary')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main():

    dataset = 'cohort2'

    # histograms of demographics
    dems = np.load(os.path.join(DATA_PATH, f'{dataset}/dems_raw.npy'))
    dem_names = np.load(os.path.join(DATA_PATH, f'{dataset}/dem_names.npy'))
    deficit_histograms(dems, dem_names, os.path.join(fig_path, 
                       'dem_histograms.png'), dems=True)


if __name__ == '__main__':
    main()