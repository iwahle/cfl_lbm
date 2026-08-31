import os
import argparse
import numpy as np
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from src.lv_dems.cfl_config import *
from src.vis.sankey import plot_sankey
from src.util.data_util import load_data
from src.vis.brain_vis import lesion_heatmap
from cfl.util.experiment_loading import exp_load

fig_path = os.path.join(FIG_PATH, 'lv_dems')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, ref_exp_id=0, plot_order_nd=None, plot_order_yd=None):

    # load data
    dataset = 'cohort1'
    X, X_test, Y, Y_test, deficit_names, dem_names = load_data(dataset, dems=True)
    results_path = os.path.join(RESULTS_PATH, 'lv_dems/cfl_results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if exp_id==-1:
        # define cfl experiment
        data_info = {'X_dims': X.shape, 'Y_dims': Y.shape, 'Y_type' : 'continuous'}
        my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                            block_names=block_names, block_params=block_params, 
                            blocks=None, verbose=0, results_path=results_path)

        # fit model, predict
        train_results = my_exp.train()
        my_exp.add_dataset(X_test, Y_test, dataset_name='test')
        test_results = my_exp.predict(dataset='test')
        xlbls_test = test_results['CauseClusterer']['x_lbls']

    else:
        xlbls_test = exp_load(results_path, exp_id, 'test', 'CauseClusterer', 
                              'x_lbls')
        
    # load results from no dems
    rp_nodem = os.path.join(RESULTS_PATH, 'lv_cfl/cfl_results')
    xlbls_test_nodems = exp_load(rp_nodem, ref_exp_id, 'test', 'CauseClusterer',
                                 'x_lbls')
    
    plot_sankey([xlbls_test_nodems, xlbls_test], os.path.join(fig_path, 'sankey.png'))

    n_clusters = len(np.unique(xlbls_test))
    if plot_order_yd is None:
        plot_order_yd = range(n_clusters)
    else:
        assert len(plot_order_yd)==n_clusters
    if plot_order_nd is None:
        plot_order_nd = range(n_clusters)
    else:
        assert len(plot_order_nd)==n_clusters

    # reload lesion data without dems for plotting
    _, X_test, _,_,_,_ = load_data(dataset, dems=False)

    # plot lesions
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW//2,1*n_clusters))
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order_yd[i]]
        lesion_heatmap(masks, mode='mean', ax=ax[i],
                       vmax=1, threshold=1e-6)
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means.png'), dpi=300)

    # plot lesions single z slice for sankey
    for i in range(n_clusters):
        fig,ax = plt.subplots(figsize=(1,1))
        masks = X_test[xlbls_test==plot_order_yd[i]]
        lesion_heatmap(masks, mode='mean', ax=ax, cut_coords=None, 
                       vmax=1, threshold=1e-6, display_mode='z_ex',
                       colorbar=False, ylabel=False)
        plt.savefig(os.path.join(fig_path, f'cfl_lesion_means_oc{i}_nc{plot_order_yd[i]}.png'), dpi=300)


    # reload raw dems unscaled for plotting
    dems = np.load(os.path.join(DATA_PATH, f'{dataset}/dems_raw.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'))
    dems_test = dems[test_idx];del(dems,test_idx)
    dem_names = np.load(os.path.join(DATA_PATH, f'{dataset}/dem_names.npy'))

    # plot demographic distributions by cluster
    for lbls,fn,plot_order in zip([xlbls_test_nodems, xlbls_test],['nodem','dem'],
                                  [plot_order_nd, plot_order_yd]):
        fig,ax = plt.subplots(n_clusters, dems_test.shape[1], figsize=(PW//2,1*n_clusters),
                            sharex='col', sharey='row')
        avgs = np.array([np.mean(dems_test[lbls==i], axis=0) for i in range(n_clusters)]) 
        for i in range(n_clusters):
            for j in range(dems_test.shape[1]):
                bins = np.linspace(np.min(dems_test[:,j]), np.max(dems_test[:,j]), 15)
                ax[i,j].hist(dems_test[lbls==plot_order[i],j], color=GRAY, bins=bins)
                ax[i,j].axvline(avgs[plot_order[i],j], color=GRAY, linestyle='--', lw=2)
                ax[i,j].axvline(dems_test[:,j].mean(), color='black', linestyle='--')
                if i==n_clusters-1:
                    ax[i,j].set_xlabel(dem_names[j])
                if dem_names[j]=='Sex':
                    ax[i,j].set_xticks([0,1])
                    ax[i,j].set_xticklabels(['M', 'F'])
        plt.tight_layout()
        plt.savefig(os.path.join(fig_path, f'dems_cluster_dists_{fn}.png'), dpi=300, transparent=True)

    # plot all dems
    fig,ax = plt.subplots(1,dems_test.shape[1], figsize=(PW//2,1.4), sharey=True)
    for j in range(dems_test.shape[1]):
        bins = np.linspace(np.min(dems_test[:,j]), np.max(dems_test[:,j]), 15)
        ax[j].hist(dems_test[:,j], color='black', bins=bins)
        ax[j].axvline(dems_test[:,j].mean(), color='black', linestyle='--')
        ax[j].set_xlabel(dem_names[j])
        if dem_names[j]=='Sex':
            ax[j].set_xticks([0,1])
            ax[j].set_xticklabels(['M', 'F'])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'dems_total_dists.png'), dpi=300, transparent=True)
    
                          


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--ref_exp_id', type=int, default=0)
    parser.add_argument('--plot_order_nd', type=int, nargs='+', default=None) # no dem
    parser.add_argument('--plot_order_yd', type=int, nargs='+', default=None) # yes dem

    main(**vars(parser.parse_args()))

    # plot_order_nd: 0 1 2 3 4
    # plot_order_yd: 0 1 2 3 4