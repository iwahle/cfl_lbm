import os
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *

fig_path = os.path.join(FIG_PATH, 'cohort1_summary')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main():

    dataset = 'cohort1'

    # histograms of deficits
    Y = np.load(os.path.join(DATA_PATH, f'{dataset}/Yraw.npy'))
    deficit_names = ['Language', 'Visuospatial']
    deficit_histograms(Y, deficit_names, os.path.join(fig_path, 
                                                      'deficit_histograms.png'))
    deficit_names = ['L', 'V']
    deficit_corr(Y, deficit_names, 
                 save_path=os.path.join(fig_path, 'deficit_corr.png'))

    

    # histograms of demographics
    dems = np.load(os.path.join(DATA_PATH, f'{dataset}/dems_raw.npy'))
    dem_names = np.load(os.path.join(DATA_PATH, f'{dataset}/dem_names.npy'))
    deficit_histograms(dems, dem_names, os.path.join(fig_path, 
                       'dem_histograms.png'), dems=True)


if __name__ == '__main__':
    main()