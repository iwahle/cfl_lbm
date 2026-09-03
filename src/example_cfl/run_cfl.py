import os
import argparse
import numpy as np
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from src.example_cfl.cfl_config import *
from src.util.data_util import load_data
from src.vis.brain_vis import lesion_heatmap
from cfl.util.experiment_loading import exp_load
from src.vis.draw_relation import draw_relation, draw_relation_sankey

'''
To just reproduce the example figures, run:
python src/example_cfl/run_cfl.py --exp_id 0 --plot_order_c 0 1 2 --plot_order_e 0 1 2
'''


fig_path = os.path.join(FIG_PATH, 'example_data')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, plot_order_c=None, plot_order_e=None):

    if plot_order_c is None:
        plot_order_c = range(block_params[1]['model_params']['n_clusters'])
    if plot_order_e is None:
        plot_order_e = range(block_params[2]['model_params']['n_clusters'])

    # load data
    X, X_test, Y, Y_test, deficit_names,_ = load_data('example_data')
    results_path = os.path.join(RESULTS_PATH, 'example_data/cfl_results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if exp_id==-1:
        # define cfl experiment
        data_info = {'X_dims': X.shape, 'Y_dims': Y.shape, 'Y_type' : 'continuous'}
        my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                            block_names=block_names, block_params=block_params, 
                            blocks=None, verbose=0, results_path=results_path)

        print(block_params)
        # fit model, predict
        train_results = my_exp.train()
        my_exp.add_dataset(X_test, Y_test, dataset_name='test')
        test_results = my_exp.predict(dataset='test')
        xlbls_test = test_results['CauseClusterer']['x_lbls']
        ylbls_test = test_results['EffectClusterer']['y_lbls']
        print(np.unique(xlbls_test))
        print(np.unique(ylbls_test))

    else:
        # load results
        xlbls_test = exp_load(results_path, exp_id, 'test', 'CauseClusterer', 
                              'x_lbls')
        ylbls_test = exp_load(results_path, exp_id, 'test', 'EffectClusterer',
                              'y_lbls')
        
    n_cclusters = len(np.unique(xlbls_test))
    n_eclusters = len(np.unique(ylbls_test))

    # combined summary figure: lesion means | relation | deficit hists, side by side
    w_lesions, w_relation, w_spacer, w_hists = PW//2, PW//4, 0.1, PW//2
    relation_y_offset = -0.06  # knob: shift relation plot up (+) or down (-), 
                               # as a fraction of figure height (small values, e.g. +/-0.02 to 0.1)
    height = max(n_cclusters, n_eclusters)*1
    fig = plt.figure(figsize=(w_lesions+w_spacer+w_relation+w_spacer+w_hists, height))
    gs = fig.add_gridspec(1, 5, width_ratios=[w_lesions, w_spacer, w_relation, w_spacer, w_hists])

    gs_lesions = gs[0].subgridspec(n_cclusters, 1)
    for i in range(n_cclusters):
        ax = fig.add_subplot(gs_lesions[i, 0])
        masks = X_test[xlbls_test==plot_order_c[i]]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax,
                                    cut_coords=None, vmax=1,
                                    threshold=1e-6, ylabel=False,
                                    cbar_label='# lesioned / # total')
        orthoslicer.title(f'Cause category {i+1}', size=FS, rotation='vertical',
                        x=-0.1, y=1, color='black', bgcolor='white', alpha=0)                                    
        orthoslicer._colorbar_ax.locator_params(nbins=3)

    ax_relation = fig.add_subplot(gs[2])
    draw_relation(cms=xlbls_test, ems=ylbls_test, plot_order_c=plot_order_c,
                  plot_order_e=plot_order_e, fontsize=FS, ax=ax_relation,
                  y_offset=relation_y_offset)

    n_deficits = Y.shape[1]
    gs_hists = gs[4].subgridspec(n_eclusters, n_deficits)
    hist_axes = np.empty((n_eclusters, n_deficits), dtype=object)
    for i in range(n_eclusters):
        for j in range(n_deficits):
            ax = fig.add_subplot(gs_hists[i, j], sharex=hist_axes[0,0],
                                 sharey=hist_axes[i,0])
            hist_axes[i,j] = ax
            ax.hist(Y_test[ylbls_test==plot_order_e[i],j], bins=np.linspace(0,1,20))
            ax.set_xlim(0,1)
            if j==0:
                ax.set_ylabel(f'Effect category {i+1}')
            if i==n_eclusters-1:
                ax.set_xlabel('Deficit')
            if i==0:
                ax.set_title(f'Deficit {deficit_names[j]}')
            ax.label_outer()

    plt.savefig(os.path.join(fig_path, 'cfl_summary.png'), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--plot_order_c', type=int, nargs='+', default=None)
    parser.add_argument('--plot_order_e', type=int, nargs='+', default=None)

    main(**vars(parser.parse_args()))
