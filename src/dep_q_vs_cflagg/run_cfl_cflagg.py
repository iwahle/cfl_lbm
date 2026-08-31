import os
import argparse
import numpy as np
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from src.vis.sankey import plot_sankey
from src.util.data_util import load_data
from src.dep_q_vs_cflagg.cfl_config import *
from src.vis.brain_vis import lesion_heatmap
from cfl.util.experiment_loading import exp_load
plt.rcParams["font.size"] = FS-1

fig_path = os.path.join(FIG_PATH, 'dep_q_vs_cflagg')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, plot_order_ca=None, plot_order_q=None, n_q_clusters=None):

    # load data
    X, X_test, _,_,_,_ = load_data('cohort2', dems=False)
    Yraw = np.load(os.path.join(DATA_PATH, 'cohort2', 'Yraw.npy'))
    train_idx = np.load(os.path.join(DATA_PATH, 'cohort2', 'train_idx.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, 'cohort2', 'test_idx.npy'))
    cflc = np.load(os.path.join(FIG_PATH, 'dep_q_vs_mean', 
                                f'q_dendrogram_C_{n_q_clusters}.npy'))-1
    print(np.unique(cflc))
    Ycfl = np.zeros((Yraw.shape[0], len(np.unique(cflc))))
    for i in range(len(np.unique(cflc))):
        Ycfl[:,i] = np.mean(Yraw[:,cflc==i], axis=1)
    Y,Y_test = Ycfl[train_idx], Ycfl[test_idx]
    Ymean,Ystd = np.mean(Y),np.std(Y)
    Y = (Y-Ymean)/Ystd
    Y_test = (Y_test-Ymean)/Ystd
    print(Y_test.shape)

    results_path = os.path.join(RESULTS_PATH, 'dep_q_vs_cflagg/cfl_results')
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
    xlbls_test_q = exp_load(rp_q, 0, 'test', 'CauseClusterer',
                                 'x_lbls')
    
    plot_sankey([xlbls_test_q, xlbls_test], os.path.join(fig_path, 'sankey.png'))

    n_clusters = len(np.unique(xlbls_test))
    if plot_order_ca is None:
        plot_order_ca = range(n_clusters)
    else:
        assert len(plot_order_ca)==n_clusters
    if plot_order_q is None:
        plot_order_q = range(n_clusters)
    else:
        assert len(plot_order_q)==n_clusters    

    # plot confusion matrix
    # FIG 7D, 7E
    cmat = np.zeros((n_clusters,n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cmat[i,j] = np.sum((xlbls_test==i) & (xlbls_test_q==j))
    cmat = cmat/np.sum(cmat, axis=1)[:,np.newaxis]
    cmat = cmat[plot_order_ca][:,plot_order_q]
    fig,ax = plt.subplots(1,1,figsize=(CW1/2.7,CW1/2.7))
    im = ax.imshow(cmat, cmap='viridis', origin='upper', vmin=0, vmax=1)
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels(np.array(['C1', 'C2', 'C3'])[plot_order_q])
    ax.set_yticklabels(['C1\'', 'C2\'', 'C3\'']) #[plot_order_ca])
    ax.set_xlabel('21 questions')
    ax.set_ylabel(f'cfl agg. (k={n_q_clusters})')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'confusion_matrix_{n_q_clusters}.png'), 
                dpi=300, transparent=True)

    # plot lesions
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW,2*n_clusters))
    vmax = np.max([np.mean(X_test[xlbls_test==plot_order_ca[i]],axis=0) for i in range(n_clusters)])
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order_ca[i]]
        lesion_heatmap(masks, mode='mean', ax=ax[i], cut_coords=None, vmax=0.8*vmax,
                       threshold=1e-6)
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, f'cfl_lesion_means_{n_q_clusters}.png'), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--plot_order_ca', type=int, nargs='+', default=None)
    parser.add_argument('--plot_order_q', type=int, nargs='+', default=None)
    parser.add_argument('--n_q_clusters', type=int)

    main(**vars(parser.parse_args()))

    # python src/dep_q_vs_cflagg/run_cfl_cflagg.py --exp_id 0 --n_q_clusters 3 --plot_order_ca 2 0 1
    # python src/dep_q_vs_cflagg/run_cfl_cflagg.py --exp_id 1 --n_q_clusters 5 --plot_order_ca 1 2 0