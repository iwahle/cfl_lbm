import os
import numpy as np
from src.util.constants import *
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import argparse

def load_lesymap_evals(series=''):
    '''
    Returns a'''
    evals = np.load(os.path.join(RESULTS_PATH, 
        f'simulated_schaefer200/lesymap_results{series}/lesymap_evals{series}.npy'))
    return evals


def load_cfl_evals(cluster_thresh):
    '''
    Returns a n_rois x n_metrics array of CFL evaluation metrics.
    '''

    fig_path = os.path.join(FIG_PATH, 'loc_comparison_schaefer200', f'cfl_results_{cluster_thresh}')
    evals = [] # will be n_rois x n_metrics
    for ri in range(100): # go through each ground truth roi
        evals.append(np.load(os.path.join(fig_path, f'cfl_eval_{ri}.npy')))
    evals = np.array(evals)
    return evals

# FIG 3
def plot_evals(evals, methods, eval_names, save_path, 
               ylims=[0.8,45,35,42]):
    '''
    Arguments:
        evals: n_methods x n_points x n_metrics array of evaluation metrics
        methods: list of method names
        eval_names: list of evaluation metric names
    '''
    print(evals.shape)
    n_points = evals.shape[1]
    fig,axs = plt.subplots(2,2,figsize=(CW1, CW1*3/4), sharex='col')
    fig.subplots_adjust(hspace=0.2)
    axs_flat = axs.flatten()

    for ei in range(len(eval_names)): # one panel for each metric
        ax = axs_flat[ei]
        sig_bar_cnt = 0
        for mi,method in enumerate(methods): # one bar for each method
            jitter = np.random.normal(0, 0.05, size=n_points)
            ax.bar(mi, np.nanmean(evals[mi,:,ei]), color='gray', 
                        yerr=np.nanstd(evals[mi,:,ei]))
            ax.scatter(np.ones(n_points)*mi+jitter, evals[mi,:,ei],  
                edgecolors='black', s=4, facecolors='black', alpha=0.5)
            if ylims is not None:
                ax.set_ylim(-ylims[ei]/20,ylims[ei])
            
            # add significance test
            if mi>0:
                stat, p_val = wilcoxon(evals[0,:,ei], evals[mi,:,ei], nan_policy='omit',
                                       alternative='less' if ei==0 else 'greater')
                if p_val < 0.05:
                    y_max = ax.get_ylim()[1]*(0.9 - 0.02*sig_bar_cnt)
                    ax.plot([0, mi], [y_max, y_max], color='black', lw=1)
                    sig_bar_cnt += 1
                print(f'{eval_names[ei]}: {method} vs. benchmark: p={p_val}')

        # connect points of same ROI across method bars
        ax.plot(np.arange(len(methods)), 
                    evals[:,:,ei], c='gray', alpha=0.4, linewidth=1)
    

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_ylabel(eval_names[ei])
    fig.align_ylabels(axs[:,0])

    plt.tight_layout()
    print('saving to:', save_path)
    plt.savefig(save_path, dpi=300)

def main(series=''):

    # load data
    evals = {}
    evals['LESYMAP'] = load_lesymap_evals(series)
    cluster_threshs = [0.8,0.9,0.95]
    for ct in cluster_threshs:
        evals[f'CFL ({int(ct*100)}%)'] = load_cfl_evals(ct)

    # only plot locs where there are sig results for all methods
    valid_locs = np.zeros((100, len(evals.keys())))
    print('Number of valid locs for each method:')
    for mi,method in enumerate(evals.keys()):
        valid_locs[:,mi] = ~np.isnan(evals[method][:,0])
        print(method, valid_locs[:,mi].sum())
    valid_locs = np.all(valid_locs, axis=1)
    evals = {k: v[valid_locs] for k,v in evals.items()}

    evals_arr = np.concatenate([evals[k][None,...] for k in evals.keys()], axis=0)
    methods = list(evals.keys())
    eval_names = ['Dice score (A.U.)', 'Peak disp. (voxels)', 'Centroid disp. (voxels)', 'Contour disp. (voxels)']

    save_path = os.path.join(FIG_PATH, 'loc_comparison_schaefer200', 
                             f'evals_cluster_threshs.png')
    plot_evals(evals_arr, methods, eval_names, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', type=str, default='')
    args = parser.parse_args()
    main(args.series)
