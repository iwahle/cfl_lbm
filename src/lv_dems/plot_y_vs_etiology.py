import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.util.constants import *
from scipy.stats import ttest_ind
from cfl.util.experiment_loading import exp_load
from matplotlib import cm
from matplotlib.colors import ListedColormap

def plot_y_vs_etiology(Y_test, xlbls, etiology_test, etiology_mapping, fig_path,
                       plot_order=None):
    ''' Should be n_xlbls x n_Y subpanels. Each subpanel
        should have etiology categories along x axis and
        scatter of Y values for that xlbl and etiology along
        y axis.
    '''
    if plot_order is None:
        plot_order = range(n_xlbls)
    else:
        assert len(plot_order)==n_xlbls
    assert len(xlbls)==len(etiology_test)
    n_xlbls = len(np.unique(xlbls))
    n_Y = Y_test.shape[1]
    xticklabels = [k.lower().capitalize() for k in etiology_mapping.keys()]
    fig,ax = plt.subplots(n_xlbls, n_Y, figsize=(PW/2.35,n_xlbls*1.46), sharex=True, sharey='col')
    for i in range(n_xlbls): # cause category
        for j in range(n_Y): # behavioral score
            for k, etiology in enumerate(etiology_mapping.keys()): # etiology
                subject_mask = (xlbls==plot_order[i]) & (etiology_test==etiology_mapping[etiology])
                Y_vals = Y_test[subject_mask,j]
                ax[i,j].scatter(np.ones(len(Y_vals))*k, Y_vals, color='black', s=1, alpha=0.8)
            ax[i,j].set_xticks(range(len(etiology_mapping)))
            ax[i,j].set_xticklabels(xticklabels, rotation=90, fontsize=6)
            # remove top and right spines
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)

        # ax[i,0].set_ylabel(f'Category {i+1}\nTest score')
        ax[i,0].set_ylabel('Test score')
    ax[0,0].set_title('Language')
    ax[0,1].set_title('Visuospatial')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'y_vs_etiology.svg'), dpi=300, transparent=True)
    print(f"Saved figure to {os.path.join(fig_path, 'y_vs_etiology.svg')}")
    plt.close()

def calc_sig(Y_test, xlbls, etiology_test, etiology_mapping, min_subjects=5):
    ''' For each xlbl and y, test whether etiology distributions are
        signficiantly different for each pair of etiologies. Only 
        include pairs that have at least min_subjects subjects each.
    '''
    u_etio = np.array(list(etiology_mapping.values()))
    print('u_etio:', u_etio)
    print(etiology_mapping)
    n_xlbls = len(np.unique(xlbls))
    n_Y = Y_test.shape[1]
    sigs_all = {}
    for i in range(n_xlbls):
        sigs_all[i] = {}
        xlbl_mask = xlbls==i
        for j in range(n_Y):
            sigs = np.zeros((len(u_etio), len(u_etio)))
            for e1i,e1 in enumerate(u_etio):
                for e2i,e2 in enumerate(u_etio):
                    if e1i==e2i:
                        sigs[e1i,e2i] = np.nan
                        continue
                    e1_mask = etiology_test==e1
                    e2_mask = etiology_test==e2
                    if (np.sum(xlbl_mask & e1_mask) < min_subjects) or \
                        (np.sum(xlbl_mask & e2_mask) < min_subjects):
                        sigs[e1i,e2i] = np.nan
                        continue
                    e1_Y = Y_test[xlbl_mask & e1_mask,j]
                    e2_Y = Y_test[xlbl_mask & e2_mask,j]
                    _, pval = ttest_ind(e1_Y, e2_Y, alternative='two-sided')
                    sigs[e1i,e2i] = pval
            sigs_all[i][j] = sigs
            print(f'Minimum sig for cause category {i} and Ydim {j}:', 
                  np.nanmin(sigs))
    return sigs_all

def plot_sig(sigs_all, etiology_mapping, fig_path, plot_order=None):
    ''' Plot the significance matrix for each xlbl and y. There should
        be one subpanel for each xlbl and y. Each subpanel should have
        etiology categories along x and y axis, and the significance
        matrix should be plotted as a color map.
    '''
    n_xlbls = len(sigs_all)
    n_Y = len(sigs_all[0])
    cbound = np.nanmax([np.nanmax(sigs_all[i][j]) for i in range(n_xlbls) for j in range(n_Y)])
    print('cbound:', cbound)
    ticklabels = [k.lower().capitalize() for k in etiology_mapping.keys()]
    fig,ax = plt.subplots(n_xlbls, n_Y, figsize=(PW/2,n_xlbls*1.5), sharex=True, sharey=True)
    for i in range(n_xlbls):
        for j in range(n_Y):
            sigs = sigs_all[plot_order[i]][j]
            blues_half = cm.get_cmap('Blues', 256)(np.linspace(0.2, 1, 128))
            cmap_half = ListedColormap(blues_half)
            # Set nan values to gray by specifying the 'nan' color in the colormap
            cmap_half.set_bad(color='lightgray')
            im = ax[i,j].imshow(sigs, cmap=cmap_half, vmin=0, vmax=cbound)
       
            ax[i,j].set_xticks(range(len(etiology_mapping)))
            ax[i,j].set_xticklabels(ticklabels, rotation=90, fontsize=6)
            if j==0:
                ax[i,j].set_yticks(range(len(etiology_mapping)))
                ax[i,j].set_yticklabels(ticklabels, fontsize=6)
        # ax[i,0].set_ylabel(f'Category {i+1}')
    ax[0,0].set_title('Language')
    ax[0,1].set_title('Visuospatial')
    # plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'sig.svg'), dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved figure to {os.path.join(fig_path, 'sig.svg')}")
    plt.close()
    # make separate figure with colorbar
    fig,ax = plt.subplots(1,1,figsize=(1,1))
    cbar = fig.colorbar(im, ax=ax, location='right', pad=0.02)
    # add more ticks
    cbar.ax.set_yticks(np.arange(0, cbound+0.05, 0.25))
    cbar.set_label('p-value', fontsize=6)
    ax.axis('off')
    plt.savefig(os.path.join(fig_path, 'sig_cbar.svg'), dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved figure to {os.path.join(fig_path, 'sig_cbar.svg')}")
    plt.close()

def main(exp_id=0, plot_order=None):
    dataset = 'cohort1'
    results_path = os.path.join(RESULTS_PATH, 'lv_cfl/cfl_results')
    fig_path = os.path.join(FIG_PATH, 'lv_dems')

    etiology_mapping = np.load(os.path.join(DATA_PATH, 'cohort1/etiology_mapping.npy'), 
                            allow_pickle=True).item()

    etiology_test = np.load(os.path.join(DATA_PATH, 'cohort1/etiology_test.npy'))

    xlbls = exp_load(results_path, exp_id, 'test', 'CauseClusterer', 'x_lbls')
    Y = np.load(os.path.join(DATA_PATH, f'{dataset}/Yraw.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'))
    Y_test = Y[test_idx]
    plot_y_vs_etiology(Y_test, xlbls, etiology_test, etiology_mapping, fig_path, plot_order=plot_order)
    sigs_all = calc_sig(Y_test, xlbls, etiology_test, etiology_mapping, min_subjects=5)
    plot_sig(sigs_all, etiology_mapping, fig_path, plot_order=plot_order)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--plot_order', type=int, nargs='+', default=None)
    args = parser.parse_args()
    main(exp_id=args.exp_id, plot_order=args.plot_order)    


# TODO:
# - correct for multiple comparisons
# - use plot_order