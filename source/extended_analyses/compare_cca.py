import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cfl.experiment import Experiment
from cfl.util.data_processing import one_hot_encode

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression as LR

from source.util import *
from source.paths import *
from source.cfl_params import *

def evaluate_cca(X, Y, in_idx, out_idx, with_pca=False):
    ''' 
    Train a CCA model on a subset of X,Y and evaluate model prediction of Y
    from X on withheld subset of X,Y. 
    Arguments:
        - X (np.ndarray): n_samples x n_voxels array of lesion masks
        - Y (np.ndarray): n_samples x n_features array of question-wise or total
                          BDI responses
        - in_idx (np.ndarray): indices of training data
        - out_idx (np.ndarray): indices of withheld data
        - with_pca (bool): whether to first reduce dimensionality of X with PCA
    Returns:
        - cca_err (float): mean squared error of CCA prediction
    '''
    if with_pca:
        # train CCA
        pca = PCA(n_components=21)
        Xred_in = pca.fit_transform(X[in_idx,:])
        cca = CCA(n_components=Y.shape[1])
        cca.fit(Xred_in,Y[in_idx,:])

        # predict cca
        Xred_out = pca.transform(X[out_idx,:])
        Y_pred_cca = cca.predict(Xred_out)

    else:
        # train CCA
        cca = CCA(n_components=Y.shape[1])
        cca.fit(X[in_idx,:],Y[in_idx,:])

        # predict cca
        Y_pred_cca = cca.predict(X[out_idx,:])

    # MSE
    cca_err = mse(Y[out_idx,:], Y_pred_cca)
    return cca_err

def evaluate_cfl(X, Y, in_idx, out_idx, data_info):
    '''
    Train a CFL model on a subset of X,Y and evaluate model prediction of Y
    from X macro-state assignments on withheld subset of X,Y. 
    Arguments:
        - X (np.ndarray): n_samples x n_voxels array of lesion masks
        - Y (np.ndarray): n_samples x n_features array of question-wise or total
                          BDI responses
        - in_idx (np.ndarray): indices of training data
        - out_idx (np.ndarray): indices of withheld data
        - data_info (dict): dictionary of data information required by CFL
    '''
    
    # run cfl on training data
    block_names = ['CondDensityEstimator', 'CauseClusterer']
    save_path = os.path.join(RESULTS_PATH, 'bdi')

    cfl_exp = Experiment(X_train=X[in_idx,:], Y_train=Y[in_idx,:], 
                        data_info=data_info, block_names=block_names, 
                        block_params=bdi_params_fixed, blocks=None, verbose=0, 
                        results_path=None)
    train_results = cfl_exp.train()
    n_clusters = bdi_params_fixed[1]['model_params']['n_clusters']
    cfl_lbls_in_idx = train_results['CauseClusterer']['x_lbls']
    cfl_lbls_in_idx = one_hot_encode(cfl_lbls_in_idx, range(n_clusters))

    # predict on withheld data
    cfl_exp.add_dataset(X[out_idx,:],Y[out_idx,:], 'withheld_data')
    cfl_lbls_out_idx = cfl_exp.predict('withheld_data')['CauseClusterer']['x_lbls']
    cfl_lbls_out_idx = one_hot_encode(cfl_lbls_out_idx, range(n_clusters))

    # build linear regression model
    cfl_model = LR()
    cfl_model.fit(cfl_lbls_in_idx, Y[in_idx,:])
    Y_pred_cfl = cfl_model.predict(cfl_lbls_out_idx)

    # compute prediction error
    cfl_err = mse(Y[out_idx,:], Y_pred_cfl)
    return cfl_err


def plot(errss, names, total_bdi, save_path):
    '''
    Plot prediction errors for different models Arguments:
        - errss (list): list of lists of prediction errors
        - names (list): list of model names
        - total_bdi (bool): whether to plot results from total BDI or
                            question-wise BDI analysis
        - save_path (str): path to save plot
    '''

    n_errs = len(errss)
    means = np.zeros((n_errs,))
    stds = np.zeros((n_errs,))
    for ei in range(n_errs):
        means[ei] = np.mean(errss[ei])
        stds[ei] = np.std(errss[ei])
    
    fig,ax = plt.subplots()
    ax.bar(range(n_errs), means, yerr=stds)
    ax.set_xticks(range(n_errs))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('mean squared error')
    if total_bdi:
        ax.set_title('Mean BDI prediction')
    else:
        ax.set_title('21 question prediction')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def main(total_bdi):
    '''
    Evaluate pca, pca+cca, and cfl on 10 different withheld data folds and plot
    the results.
    Arguments:
        - total_bdi (bool): whether to use total BDI or question-wise BDI
    Returns: None
    '''

    save_dir = os.path.join(RESULTS_PATH, 'bdi')

    # load data
    X,Y,data_info = load_scale_data(analysis='bdi')
    if total_bdi:
        Y = np.mean(Y,axis=1, keepdims=True)

    # run CCA and CFL on 10 folds
    n_split = 10
    cca_errs = np.zeros((n_split,))
    pca_cca_errs = np.zeros((n_split,))
    cfl_errs = np.zeros((n_split,))
    kf = KFold(n_splits=n_split, shuffle=True, random_state=0)
    for i,(in_idx,out_idx) in tqdm(enumerate(kf.split(range(X.shape[0])))):
        cca_errs[i] = evaluate_cca(np.copy(X), np.copy(Y), in_idx, out_idx, 
                                   with_pca=False)
        pca_cca_errs[i] = evaluate_cca(np.copy(X), np.copy(Y), in_idx, out_idx, 
                                       with_pca=True)
        cfl_errs[i] = evaluate_cfl(np.copy(X), np.copy(Y), in_idx, out_idx,
                                   data_info)

    errss = [cca_errs, pca_cca_errs, cfl_errs]
    names = ['CCA', 'PCA+CCA', 'CFL']

    for i in range(3):
        print(names[i], np.mean(errss[i]))

    subdir = 'mean_bdi' if total_bdi else 'questions'

    if not os.path.exists(os.path.join(save_dir, 'cca_comparison', subdir)):
        os.makedirs(os.path.join(save_dir, 'cca_comparison', subdir))

    plot(errss, names, total_bdi=total_bdi, save_path=os.path.join(save_dir, 
        'cca_comparison', subdir, 'cca_pcacca_cfl_mse.png'))
    plot([cca_errs, cfl_errs], ['CCA', 'CFL'], total_bdi=total_bdi, 
        save_path=os.path.join(save_dir, 'cca_comparison', subdir, 
        'cca_cfl_mse.png'))
    plot([pca_cca_errs,cfl_errs], ['PCA+CCA','CFL'], total_bdi=total_bdi, 
        save_path=os.path.join(save_dir, 'cca_comparison', subdir, 
        'pcacca_cfl_mse.png'))
    plot([cca_errs,pca_cca_errs], ['CCA','PCA+CCA'], total_bdi=total_bdi, 
        save_path=os.path.join(save_dir, 'cca_comparison', subdir, 
        'cca_pcacca_mse.png'))

    print('CCA')
    print(cca_errs)
    print('PCACCA')
    print(pca_cca_errs)
    print('CFL')
    print(cfl_errs)
    
    if not os.path.exists(os.path.join(save_dir, 'cca_comparison', subdir)):
        os.mkdir(os.path.join(save_dir, 'cca_comparison', subdir))
    np.save(os.path.join(save_dir, 'cca_comparison', subdir, 'cca_errs'), 
            cca_errs)
    np.save(os.path.join(save_dir, 'cca_comparison', subdir, 'pca_cca_errs'), 
            pca_cca_errs)
    np.save(os.path.join(save_dir, 'cca_comparison', subdir, 'cfl_errs'), 
            cfl_errs)


if __name__=='__main__':
    '''
    Example usage:
        python source/extended_analyses/compare_cca.py --total_bdi 0
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_bdi', type=int)
    main(**vars(parser.parse_args()))