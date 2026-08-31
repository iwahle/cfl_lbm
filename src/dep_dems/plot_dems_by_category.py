import os
import argparse
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *
from cfl.util.experiment_loading import exp_load
from scipy.stats import mannwhitneyu
from src.lv_dems.plot_dems_by_category import plot_dems_by_category

fig_path = os.path.join(FIG_PATH, 'dep_dems')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=0, plot_order=None, density=False, do_bonf=True):

    dataset = 'cohort2'
    results_path = os.path.join(RESULTS_PATH, 'dep_cfl/cfl_results')

    dems = np.load(os.path.join(DATA_PATH, f'{dataset}/dems_raw.npy'))
    dem_names = np.load(os.path.join(DATA_PATH, f'{dataset}/dem_names.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'))
    dems = dems[test_idx]

    xlbls = exp_load(results_path, exp_id, 'test', 'CauseClusterer', 'x_lbls')
    plot_dems_by_category(dems, xlbls, dem_names, plot_order, exp_id, fig_path, density, do_bonf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--plot_order', type=int, nargs='+', default=None)
    args = parser.parse_args()
    for do_bonf in [True, False]:
        for density in [True, False]:
            main(exp_id=args.exp_id, plot_order=args.plot_order, density=density, do_bonf=do_bonf)