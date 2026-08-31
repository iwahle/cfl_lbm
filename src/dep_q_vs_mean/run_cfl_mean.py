import os
import argparse
import numpy as np
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from src.vis.sankey import plot_sankey
from src.util.data_util import load_data
from src.dep_q_vs_mean.cfl_config import *
from src.vis.brain_vis import lesion_heatmap
from cfl.util.experiment_loading import exp_load
plt.rcParams["font.size"] = FS-1

fig_path = os.path.join(FIG_PATH, 'dep_q_vs_mean')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, ref_exp_id=0, plot_order_m=None, plot_order_q=None):

    # load data
    X, X_test, _,_, deficit_names,_ = load_data('cohort2', dems=False)
    Yraw = np.load(os.path.join(DATA_PATH, 'cohort2', 'Yraw.npy'))
    train_idx = np.load(os.path.join(DATA_PATH, 'cohort2', 'train_idx.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, 'cohort2', 'test_idx.npy'))
    Y,Y_test = Yraw[train_idx], Yraw[test_idx]
    # compute mean bdi
    Y = np.mean(Y, axis=1, keepdims=True)
    Y_test = np.mean(Y_test, axis=1, keepdims=True)
    Ymean,Ystd = np.mean(Y),np.std(Y)
    Y = (Y-Ymean)/Ystd
    Y_test = (Y_test-Ymean)/Ystd

    results_path = os.path.join(RESULTS_PATH, 'dep_q_vs_mean/cfl_results')
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
        
    # load results from Y = 21 questions
    rp_q = os.path.join(RESULTS_PATH, 'dep_cfl/cfl_results')
    xlbls_test_q = exp_load(rp_q, ref_exp_id, 'test', 'CauseClusterer', 'x_lbls')

    plot_sankey([xlbls_test_q, xlbls_test], os.path.join(fig_path, 'sankey.png'))
    
    n_clusters = len(np.unique(xlbls_test))
    if plot_order_m is None:
        plot_order_m = range(n_clusters)
    else:
        assert len(plot_order_m)==n_clusters
    if plot_order_q is None:
        plot_order_q = range(n_clusters)
    else:
        assert len(plot_order_q)==n_clusters

    # plot confusion matrix
    # FIG 7B
    cmat = np.zeros((n_clusters,n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cmat[i,j] = np.sum((xlbls_test==i) & (xlbls_test_q==j))
    cmat = cmat/np.sum(cmat, axis=1)[:,np.newaxis]
    cmat = cmat[plot_order_m][:,plot_order_q]
    fig,ax = plt.subplots(1,1,figsize=(CW1/2.7,CW1/2.7))
    im = ax.imshow(cmat, cmap='viridis', origin='upper', vmin=0, vmax=1)
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels(np.array(['C1', 'C2', 'C3'])[plot_order_q])
    ax.set_yticklabels(['C1\'', 'C2\'', 'C3\''])
    ax.set_xlabel('21 questions')
    ax.set_ylabel('Mean BDI')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'confusion_matrix.png'), dpi=300, transparent=True)

    # plot lesions
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW,2*n_clusters))
    vmax = np.max([np.mean(X_test[xlbls_test==plot_order_m[i]],axis=0) for i in range(n_clusters)])
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order_m[i]]
        lesion_heatmap(masks, mode='mean', ax=ax[i], cut_coords=None, vmax=0.8*vmax,
                       threshold=1e-6)
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means.png'), dpi=300)

    # plot lesions single z slice for sankey
    for i in range(n_clusters):
        fig,ax = plt.subplots(figsize=(1,1))
        masks = X_test[xlbls_test==plot_order_m[i]]
        lesion_heatmap(masks, mode='mean', ax=ax, cut_coords=None, 
                       vmax=1, threshold=1e-6, display_mode='z_ex',
                       colorbar=False, ylabel=False)
        plt.savefig(os.path.join(fig_path, f'cfl_lesion_means_oc{i}_nc{plot_order_m[i]}.png'), dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--ref_exp_id', type=int, default=0)
    parser.add_argument('--plot_order_m', type=int, nargs='+', default=None)
    parser.add_argument('--plot_order_q', type=int, nargs='+', default=None)

    main(**vars(parser.parse_args()))

    # python src/dep_q_vs_mean/run_cfl_mean.py --exp_id 0 --ref_exp_id 0 --plot_order_m 1 0 2