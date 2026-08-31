import os
import argparse
from cfl import Experiment
from src.util.constants import *
from src.dep_cfl.cfl_config import *
from src.vis.sankey import plot_sankey
from src.util.data_util import load_data
from cfl.util.experiment_loading import exp_load

fig_path = os.path.join(FIG_PATH, 'dep_cfl')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(n_clusters_tune, exp_ids=[-1]):

    if exp_ids!=[-1]:
        assert len(exp_ids)==len(n_clusters_tune)

    # load data
    X, X_test, Y, Y_test, _, _ = load_data('cohort2')
    results_path = os.path.join(RESULTS_PATH, 'dep_cfl/cfl_results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if exp_ids==[-1]:

        lblss = []
        for n_clusters in n_clusters_tune:
            # modify what's in config file
            my_cde_params = cde_params.copy()
            my_cc_params = cc_params.copy()
            my_cc_params['model_params']['n_clusters'] = n_clusters
            my_block_params = [my_cde_params, my_cc_params]

            # define cfl experiment
            data_info = {'X_dims': X.shape, 'Y_dims': Y.shape, 'Y_type' : 'continuous'}
            my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                                block_names=block_names, block_params=my_block_params, 
                                blocks=None, verbose=0, results_path=results_path)

            # fit model, predict
            train_results = my_exp.train()
            my_exp.add_dataset(X_test, Y_test, dataset_name='test')
            test_results = my_exp.predict(dataset='test')
            xlbls_test = test_results['CauseClusterer']['x_lbls']
            lblss.append(xlbls_test)

    else:
        lblss = [exp_load(results_path, exp_id, 'test', 'CauseClusterer', 
                              'x_lbls') for exp_id in exp_ids]

    plot_sankey(lblss, os.path.join(fig_path, 'var_clusters_sankey.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters_tune', type=int, nargs='+')
    parser.add_argument('--exp_ids', type=int, nargs='+', default=[-1])

    main(**vars(parser.parse_args()))

# python src/dep_cfl/vary_clusters_sankey.py --n_clusters_tune 2 3 4 --exp_ids -1