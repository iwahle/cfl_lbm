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

fig_path = os.path.join(FIG_PATH, 'dep_vs_kmeans')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, ref_exp_id=0, plot_order=None):

    # load data
    X,X_test,Y,Y_test,_,_ = load_data('cohort2', dems=False)

    results_path = os.path.join(RESULTS_PATH, 'dep_q_vs_mean/cfl_results')
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

    # # get user input
    # n_clusters = int(input('Enter number of clusters: '))
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=50,
                    random_state=RS)
    kmeans.fit(X)
    xlbls_test = kmeans.predict(X_test)

    # load results from cfl
    rp_cfl = os.path.join(RESULTS_PATH, 'dep_cfl/cfl_results')
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

    # plot lesions single z slice for sankey
    for i in range(n_clusters):
        fig,ax = plt.subplots(figsize=(1,1))
        masks = X_test[xlbls_test==plot_order[i]]
        lesion_heatmap(masks, mode='mean', ax=ax, cut_coords=None, 
                       vmax=1, threshold=1e-6, display_mode='z_ex',
                       colorbar=False, ylabel=False)
        plt.savefig(os.path.join(fig_path, f'kmeans_lesion_means_oc{i}_nc{plot_order[i]}.png'), dpi=300)

    # plot y variance for each cluster
    mbdi_test = np.mean(Y_test,axis=1)
    fig,ax = plt.subplots(1,2,figsize=(4,2,))
    for i in range(np.unique(xlbls_test_q).shape[0]):
        ax[0].bar(i, np.var(mbdi_test[xlbls_test_q==i]))
    for i in range(n_clusters):
        ax[1].bar(i, np.var(mbdi_test[xlbls_test==plot_order[i]]))
    ax[0].set_xticks(range(np.unique(xlbls_test_q).shape[0]))
    ax[1].set_xticks(range(n_clusters))
    ax[0].set_title('CFL categories')
    ax[1].set_title('K-means clusters')
    ax[0].set_ylabel('Variance')
    ax[1].set_ylabel('Variance')
    ax[0].set_xlabel('Category')
    ax[1].set_xlabel('Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'kmeans_y_var.png'), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--ref_exp_id', type=int, default=0)
    parser.add_argument('--plot_order', type=int, nargs='+', default=None)

    main(**vars(parser.parse_args()))