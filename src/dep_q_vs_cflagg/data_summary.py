import os
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *

fig_path = os.path.join(FIG_PATH, 'cohort2_summary')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main():

    dataset = 'cohort2'

    # histograms of demographics
    Y = np.load(os.path.join(DATA_PATH, f'{dataset}/Y_raw.npy'))
    Y_names = ['Mean BDI']
    deficit_histograms(Y, Y_names, os.path.join(fig_path, 
                       'meannBDI_histograms.png'), dems=True)


if __name__ == '__main__':
    main()