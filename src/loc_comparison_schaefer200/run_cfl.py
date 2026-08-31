import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from src.util.data_util import *
from src.vis.brain_vis import lesion_heatmap
from src.util.sig_test import sig_test_voxels
from src.loc_comparison_schaefer200.cfl_config import block_names as BLOCK_NAMES
from src.loc_comparison_schaefer200.cfl_config import block_params as BLOCK_PARAMS
from cfl.util.experiment_loading import exp_load

def get_cfl_results(X,Y,X_test,Y_test,results_path, block_names=BLOCK_NAMES, 
                    block_params=BLOCK_PARAMS):
    ''' Run CFL for a given behavioral score (Yidx) generated from 
        a single parcel
        Returns:
            xlbls_test: cluster labels for test set [n_voxels,]
    '''
    # define cfl experiment
    data_info = {'X_dims': X.shape, 'Y_dims': Y.shape, 'Y_type' : 'continuous'}
    my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                        block_names=deepcopy(block_names), 
                        block_params=deepcopy(block_params), 
                        blocks=None, verbose=0, results_path=results_path)

    # fit model, predict
    _ = my_exp.train()
    my_exp.add_dataset(X_test, Y_test, dataset_name='test')
    test_results = my_exp.predict(dataset='test')
    xlbls_test = test_results['CauseClusterer']['x_lbls']
    return xlbls_test, my_exp

def get_null_dist_lbls(X, Y, X_test, Y_test, block_names, block_params, n_iter=100):
    ''' Get null distribution of CFL results by shuffling Y labels
        Returns:
            null_dist: null samples of CFL lesion categories [n_iter, n_voxels]
    '''
    null_dist_lbls = np.zeros((n_iter, X_test.shape[0]))
    for i in tqdm(range(n_iter)):
        Y_shuffle = Y[np.random.permutation(Y.shape[0])]
        Y_test_shuffle = Y_test[np.random.permutation(Y_test.shape[0])]
        null_dist_lbls[i],_ = get_cfl_results(X,Y_shuffle,X_test,Y_test_shuffle,
                                       results_path=None, # don't save shuffle full results
                                       block_names=block_names, 
                                       block_params=block_params) 
    return null_dist_lbls

def get_deficit_category(xlbls_test, Y_test):
    ''' Returns which x category corresponds to the highest mean Y_test value '''
    n_clusters = np.max(xlbls_test)+1
    Ymeans = np.zeros(n_clusters)
    for i in range(n_clusters):
        Ymeans[i] = np.mean(Y_test[xlbls_test==i])
    print('SELECTED CLUSTER: ', np.argmax(Ymeans))
    return np.argmax(Ymeans)

def get_sig_map(X_test, Y_test, null_dist_lbls, map_pred, cluster_size):
    
    deficit_category_means = np.zeros((null_dist_lbls.shape[0], X_test.shape[1]))
    for i in range(null_dist_lbls.shape[0]): # loop over shuffle iterations
        deficit_category = get_deficit_category(null_dist_lbls[i].astype(int), Y_test)
        sample_idx = null_dist_lbls[i]==deficit_category
        if np.sum(sample_idx) >= cluster_size: # match to observed cluster size
            # assuming will always be >= cluster_size because shuffle will
            # lead to two generic clusters
            # Randomly select indices to keep to match cluster_size
            keep_idx = np.random.choice(np.where(sample_idx)[0], 
                                      size=cluster_size, 
                                      replace=False)
            sample_idx[:] = False
            sample_idx[keep_idx] = True
        deficit_category_means[i] = np.mean(X_test[sample_idx], axis=0)
    
    sig_map = np.mean(map_pred > deficit_category_means,axis=0)
    return sig_map

def main(exp_id=-1, Yidx=None, cluster_thresh=0.8, plot_thresh=0.8, 
         fig_path=None, results_path=None):
    ''' Run CFL for a given behavioral score (Yidx) generated from 
        a single parcel '''
    
    # load data
    X, X_test, Y, Y_test, deficit_names,_ = load_data('simulated_schaefer200')
    if Yidx is not None:
        Y = Y[:,Yidx][:,None]
        Y_test = Y_test[:,Yidx][:,None]
        deficit_names = deficit_names[Yidx]
    
    fig_path = os.path.join(FIG_PATH, fig_path)
    if not os.path.exists(fig_path):
        print(f'Creating fig_path: {fig_path}')
        os.makedirs(fig_path)

    results_path = os.path.join(RESULTS_PATH, results_path)
    if not os.path.exists(results_path):
        print(f'Creating results_path: {results_path}')
        os.makedirs(results_path)


    if exp_id==-1:
        xlbls_test,main_exp = get_cfl_results(X,Y,X_test,Y_test,results_path)
        final_block_params = main_exp.block_params
        final_block_params[1]['tune'] = False
    else:
        xlbls_test = exp_load(results_path, exp_id, 'test', 'Causeclusterer', 
                              'x_lbls')

    # plot lesions
    n_clusters = len(np.unique(xlbls_test))
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW//2,1*n_clusters))
    for i in range(n_clusters):
        masks = X_test[xlbls_test==i]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax[i], 
                                     cut_coords=None, vmax=1, threshold=plot_thresh)
        # set fewer ticks cbar
        orthoslicer._colorbar_ax.locator_params(nbins=3)

        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, f'cfl_lesion_means_{Yidx}.png'), dpi=300)

    # plot deficit hists
    fig,ax = plt.subplots(n_clusters, figsize=(PW//2,1*n_clusters),
                          sharex=True)
    global_avgs = np.mean(Y_test, axis=0)
    for i in range(n_clusters):
        ax[i].hist(Y_test[xlbls_test==i], 
                        bins=np.arange(-0.1,1.2,0.1), color=GREEN)
        avg = np.mean(Y_test[xlbls_test==i])
        ax[i].axvline(avg, color='green', linestyle='--')
        ax[i].axvline(global_avgs, color='black', linestyle='--')
        ax[0].set_title(deficit_names)
        ax[-1].set_xlabel('Score')
        ax[i].set_ylabel('Cluster {}'.format(i))
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'cfl_deficit_hists_{Yidx}.png'), dpi=300)
    
    ## evaluate
    # automatically pick cluster with higher max of mean
    lesion_idx = get_deficit_category(xlbls_test, Y_test)
    map_pred = np.mean(X_test[xlbls_test==lesion_idx], axis=0)
    sig_map_raw = sig_test_voxels(X_test, xlbls_test, lesion_idx,
                              n_resample=10000)
    print(sig_map_raw.shape)
    print(map_pred.shape)
    
    pthresh = 0.05
    sig_map = sig_map_raw > (1 - pthresh)
    # because CFL relies on lesion map clusters to define regions
    # as opposed to model weights, large (i.e. full hemispheric) lesions
    # have the effect of always generating significant voxels due to
    # coverage of regions that are otherwise poorly sampled in the dataset.
    # to avoid this, we also include a check for half of cluster members
    # to be lesioned at a given voxel to be considered significant.
    sig_map[map_pred<cluster_thresh] = 0 # filter by internal agreement
    print(np.sum(sig_map)/len(sig_map))
    map_pred[sig_map==0] = 0 # filter by significance
    
    # load ground truth
    map_gt = load_brain(os.path.join(DATA_PATH, 
        f'simulated_schaefer200/{ATLAS_FN}'),
        mask=None, to_flatten=False)
    # zero out right hemisphere of ground truth
    map_gt[map_gt.shape[0]//2:] = 0
    map_gt = map_gt.reshape(np.product(map_gt.shape))
    map_gt = map_gt==Yidx+1 # parcel ids are 1-n
    flat_t = np.reshape(TEMPLATE, -1)
    map_gt = map_gt[flat_t.astype(bool)]

    # plot gt
    fig,ax = plt.subplots(1,1,figsize=(PW//2,1))
    orthoslicer = lesion_heatmap(map_gt[None,:], mode='mean', ax=ax, 
        cut_coords=None, vmax=1, threshold=plot_thresh)
    # set fewer ticks cbar
    orthoslicer._colorbar_ax.locator_params(nbins=3)
    plt.savefig(os.path.join(fig_path, f'cfl_lesion_gt_{Yidx}.png'), dpi=300)

    # reshape to 3D
    map_pred_3d = np.zeros_like(flat_t)
    map_pred_3d[flat_t.astype(bool)] = map_pred
    map_pred_3d = map_pred_3d.reshape(TEMPLATE.shape)

    map_gt_3d = np.zeros_like(flat_t)
    map_gt_3d[flat_t.astype(bool)] = map_gt
    map_gt_3d = map_gt_3d.reshape(TEMPLATE.shape)

    print('Dice')
    dice = dice_eval(map_pred>0, map_gt)
    print(dice)

    print('Peak Disp')
    peak_disp = peak_disp_eval(map_pred_3d, map_gt_3d)
    print(peak_disp)

    print('Centroid Disp')
    cent_disp = centroid_disp_eval(map_pred_3d, map_gt_3d)
    print(cent_disp)

    print('Contour Disp')
    contour_disp = contour_disp_eval(map_pred_3d>0, map_gt_3d)
    print(contour_disp)

    np.save(os.path.join(fig_path, f'cfl_eval_{Yidx}.npy'), 
            [dice, peak_disp, cent_disp, contour_disp])

    sig_map_3d = np.zeros_like(flat_t)
    sig_map_3d[flat_t.astype(bool)] = sig_map
    sig_map_3d = sig_map_3d.reshape(TEMPLATE.shape)

    fig,ax = plt.subplots(1,3,figsize=(8,4))
    ax[0].imshow(map_pred_3d.sum(axis=2), cmap='hot')
    ax[0].set_title('Predicted')
    ax[1].imshow(map_gt_3d.sum(axis=2), cmap='hot')
    ax[1].set_title('Ground Truth')
    ax[2].imshow(sig_map_3d.sum(axis=2), cmap='hot')
    ax[2].set_title('Sig Voxels')
    plt.savefig(os.path.join(fig_path, f'cfl_lesion_3d_{Yidx}.png'), dpi=300)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_id', type=int, default=-1)
    # parser.add_argument('--Yidx', type=int, default=None)
    # parser.add_argument('--plot_order', type=int, nargs='+', default=None)
    # parser.add_argument('--thresh', type=float, default=0.8)
    # args = parser.parse_args()
    # main(exp_id=args.exp_id, Yidx=args.Yidx, plot_order=args.plot_order,
    #      thresh=args.thresh)
    
    exp_id = -1
    for cluster_thresh in [0.75,0.8,0.85,0.9,0.95,1]:
        print('########################################################')
        print(f'Running for cluster_thresh: {cluster_thresh}')
        print('########################################################')
        fig_path = f'loc_comparison_schaefer200/cfl_results_{cluster_thresh}'
        results_path = f'loc_comparison_schaefer200/cfl_results_{cluster_thresh}'
        for Yidx in range(100):
            print('Yidx: ', Yidx)
            main(exp_id=exp_id, Yidx=Yidx, cluster_thresh=cluster_thresh, 
                 fig_path=fig_path, results_path=results_path)
