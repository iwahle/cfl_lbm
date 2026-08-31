import os
import argparse
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *
from cfl.util.experiment_loading import exp_load
from scipy.stats import mannwhitneyu

fig_path = os.path.join(FIG_PATH, 'lv_dems')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def plot_dems_by_category(dems, xlbls, dem_names, plot_order, exp_id, fig_path, density=False, do_bonf=True):
    n_cats = len(np.unique(xlbls))
    n_dems = dems.shape[1]
    if plot_order is None:
        plot_order = range(n_cats)
    
    fig,ax = plt.subplots(n_cats, n_dems, figsize=(PW//2,n_cats), sharex='col')
    for i in range(n_cats):
        for j in range(n_dems):
            if dem_names[j]=='Sex':
                bins = [-0.5,0.5,1.5]
            else:
                bins = np.linspace(np.min(dems[:,j])-1, np.max(dems[:,j])+1, 20)
            ax[i,j].hist(dems[:,j], color='gray', bins=bins, density=density,
                         label='All')
            ax[i,j].hist(dems[xlbls==plot_order[i],j], color=BLUE, bins=bins, 
                         density=density, alpha=0.7, label=f'Category {i+1}')
            ax[i,j].axvline(np.mean(dems[:,j]), color='black', linestyle='--', lw=1)
            ax[i,j].axvline(np.mean(dems[xlbls==plot_order[i],j]), color=DARK_BLUE, linestyle='--', lw=1)
            # turn off top and right spines
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            # mannwhitneyu two sided
            stat, pval = mannwhitneyu(dems[:,j], dems[xlbls==plot_order[i],j], alternative='two-sided')
            # adjust by bonferroni
            if do_bonf:
                n_comparisons = n_cats * n_dems
                pval = pval * n_comparisons
            ax[i,j].set_title(f'p={pval:.2f}', fontsize=6, loc='left')
            if j==0:
                ax[i,j].set_ylabel(f'Category {i+1}')
            if i==n_cats-1:
                ax[i,j].set_xlabel(dem_names[j])
            if dem_names[j]=='Sex':
                ax[i,j].set_xticks([0,1])
                ax[i,j].set_xticklabels(['M', 'F'])

    fig.align_ylabels()


    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'dems_by_category_{exp_id}_density_{density}_bonf_{do_bonf}.svg'), 
                format='svg', transparent=True, dpi=300)
    print(f"Saved figure to {os.path.join(fig_path, f'dems_by_category_{exp_id}_density_{density}_bonf_{do_bonf}.svg')}")
    plt.close()



def main(exp_id=0, plot_order=None, density=False, do_bonf=True):

    dataset = 'cohort1'
    results_path = os.path.join(RESULTS_PATH, 'lv_cfl/cfl_results')

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