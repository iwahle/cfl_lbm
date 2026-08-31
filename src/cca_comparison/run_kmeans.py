import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.util.constants import *
from src.vis.sankey import plot_sankey
from src.vis.brain_vis import lesion_heatmap
from src.util.data_util import load_data
from cfl.util.experiment_loading import exp_load
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import KFold
from src.util.sig_test import sig_test_voxels

fig_path = os.path.join(FIG_PATH, 'cca_comparison/kmeans_results')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, ref_exp_id=0, plot_order=None, parcel_series='schaefer200'):

    # load data
    X,X_test,_,_,_,_ = load_data(f'simulated_{parcel_series}', dems=False)

    results_path = os.path.join(RESULTS_PATH, 'cca_comparison/kmeans_results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # grid seach n_clusters kmeans kfold
    n_clusters = range(2,21)
    try:
        all_scores = np.load(os.path.join(fig_path, 'kmeans_scores.npy'))
    except:
        all_scores = []
        grid = KFold(n_splits=5, shuffle=True, random_state=RS)
        for n in n_clusters:
            print(n)
            kmeans = KMeans(n_clusters=n, init='random', n_init=50,
                            random_state=RS)
            scores = []
            for train_idx, test_idx in grid.split(X):
                print(X[train_idx].shape, X[test_idx].shape)
                kmeans.fit(X[train_idx])
                xlbls = kmeans.predict(X[test_idx])
                scores.append(davies_bouldin_score(X[test_idx], xlbls))
            all_scores.append(scores)

        all_scores = np.array(all_scores)
        np.save(os.path.join(fig_path, 'n_clusters.npy'), n_clusters)
        np.save(os.path.join(fig_path, 'kmeans_scores.npy'), all_scores)
    
    fig,ax = plt.subplots(figsize=(2,2))
    ax.plot(n_clusters, np.mean(all_scores, axis=1), color='black')
    ax.fill_between(n_clusters, np.mean(all_scores, axis=1)-np.std(all_scores, axis=1),
                    np.mean(all_scores, axis=1)+np.std(all_scores, axis=1), alpha=0.3, color='black')
    ax.set_xlabel('# clusters')
    ax.set_ylabel('Davies Bouldin')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'kmeans_kfold.png'), dpi=300, bbox_inches='tight')
    # plt.show()

    # # get user input
    n_clusters = int(input('Enter number of clusters: '))
    # n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=50,
                    random_state=RS)
    kmeans.fit(X)
    xlbls_test = kmeans.predict(X_test)

    # load results from cfl
    rp_cfl = os.path.join(RESULTS_PATH, 'cca_comparison/cfl_results')
    xlbls_test_q = exp_load(rp_cfl, ref_exp_id, 'test', 'CauseClusterer',
                                 'x_lbls')
    
    plot_sankey([xlbls_test_q, xlbls_test], os.path.join(fig_path, 'sankey.png'))

    n_clusters = len(np.unique(xlbls_test))
    if plot_order is None:
        plot_order = range(n_clusters)
    else:
        assert len(plot_order)==n_clusters

    # plot lesions
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW,1*n_clusters))
    vmax = np.max([np.mean(X_test[xlbls_test==plot_order[i]],axis=0) for i in range(n_clusters)])
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order[i]]
        lesion_heatmap(masks, mode='mean', ax=ax[i], cut_coords=None, vmax=0.8*vmax,
                       threshold=1e-6)
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, 'kmeans_lesion_means.png'), dpi=300)

    # plot lesions single z slice for sankey kmeans
    for i in range(n_clusters):
        try:
            sig_map_raw = np.load(os.path.join(fig_path, f'kmeans_sig_map_raw_{plot_order[i]}.npy'))
        except:
            print(f'kmeans_sig_map_raw_{plot_order[i]} not found, running sig test')
            sig_map_raw = sig_test_voxels(X_test, xlbls_test, xi=plot_order[i], n_resample=10000)
            np.save(os.path.join(fig_path, f'kmeans_sig_map_raw_{plot_order[i]}.npy'), sig_map_raw)

        # filter by significance
        pthresh = 0.05
        sig_map = sig_map_raw > (1 - pthresh)
        # filter by internal agreement
        cluster_thresh = 0.8
        masks = X_test[xlbls_test==plot_order[i]]
        print(f'cluster {i} has {np.sum(np.sum(masks,axis=0)>cluster_thresh)}')
        sig_map[np.mean(masks,axis=0)<cluster_thresh] = 0
        # sig_map should now be a binary map of voxels that are significantly lesioned

        fig,ax = plt.subplots(figsize=(1,1))
        masks = X_test[xlbls_test==plot_order[i]]
        lesion_heatmap(masks, mode='mean', ax=ax, cut_coords=None, 
                       vmax=1, threshold=1e-6, display_mode='z_ex',
                       colorbar=False, ylabel=False, contour_map=sig_map)
        plt.savefig(os.path.join(fig_path, f'kmeans_lesion_means_oc{i}_nc{plot_order[i]}.png'), dpi=300)

    # plot unraveled lesion means for supp
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW,n_clusters))
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order[i]]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax[i], display_mode='z',
                                     cut_coords=None, vmax=1, threshold=1e-6)
        # set fewer ticks cbar
        orthoslicer._colorbar_ax.locator_params(nbins=3)

        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'kmeans_lesion_means_unraveled.png'), dpi=300,
                bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--ref_exp_id', type=int, default=0)
    parser.add_argument('--plot_order', type=int, nargs='+', default=None)

    main(**vars(parser.parse_args()))