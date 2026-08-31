import os
import argparse
import numpy as np
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from src.util.data_util import *
from src.cca_comparison.cfl_config import *
from src.vis.brain_vis import lesion_heatmap
from cfl.util.experiment_loading import exp_load
from src.cca_comparison.standard_cut_coords import *
from src.util.sig_test import sig_test, sig_test_voxels

fig_path = os.path.join(FIG_PATH, 'cca_comparison')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, plot_order=None, parcel_series='schaefer200', roi_idxs=None, 
atlas_fn=ATLAS_FN):

    # load data
    X, X_test, Y, Y_test, deficit_names,_ = load_data(f'simulated_{parcel_series}')
    Y = Y[:,roi_idxs]
    Y_test = Y_test[:,roi_idxs]
    results_path = os.path.join(RESULTS_PATH, 'cca_comparison/cfl_results')
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
        xlbls_test = exp_load(results_path, exp_id, 'test', 'Causeclusterer', 
                              'x_lbls')

    # plot lesions
    n_clusters = len(np.unique(xlbls_test))
    if plot_order is None:
        plot_order = range(n_clusters)
    else:
        assert len(plot_order)==n_clusters

    fig,ax = plt.subplots(n_clusters,1,figsize=(PW//2,1*n_clusters))
    cut_coords = [A_coords, B_coords, C_coords]
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order[i]]
        cut_coord = None
        if i<3:
            cut_coord = cut_coords[i]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax[i], 
                                     cut_coords=cut_coord, vmax=1, threshold=0.75)
        # set fewer ticks cbar
        orthoslicer._colorbar_ax.locator_params(nbins=3)

        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means.png'), dpi=300)

    # plot tiled version
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW,n_clusters))
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order[i]]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax[i], display_mode='z',
                                     cut_coords=None, vmax=1, threshold=1e-6, 
                                     cbar_fancy_scaling=False, 
                                     cbar_label='Prop. lesioned', cbar_width_factor=1.5)
        # set fewer ticks cbar
        orthoslicer._colorbar_ax.locator_params(nbins=3)

        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means_unraveled.png'), dpi=300,
                bbox_inches='tight')


    ###########################################################################
    # load ground truth rois
    parcel_path = os.path.join(DATA_PATH, f'simulated_{parcel_series}/{atlas_fn}')
    parcels = load_brain(parcel_path, to_flatten=True, mask=TEMPLATE)
    template_flat = TEMPLATE.flatten()
    print('parcels.shape:', parcels.shape)

    rois = []
    for roi_idx in roi_idxs:
        roi = parcels == roi_idx
        roi_in_brain_space = np.zeros(template_flat.shape)
        roi_in_brain_space[template_flat==1] = roi
        roi_in_brain_space = np.reshape(roi_in_brain_space, TEMPLATE.shape)
        roi_img = nib.Nifti1Image(roi_in_brain_space, AFFINE)
        rois.append(roi_img)
    rois = np.array(rois)
    rois_to_plot = [[0],[1],[2],[0,2],[0,1,2],[]]
    ###########################################################################
    # sig test X and plot on lesions
    # FIG 4C
    fig,ax = plt.subplots(n_clusters,1,figsize=(CW1*0.75,0.8*n_clusters))
    for i in range(n_clusters): 
        try:
            sig_map_raw = np.load(os.path.join(fig_path, f'cfl_sig_map_raw_{plot_order[i]}.npy'))
        except:
            print(f'sig_map_raw_{plot_order[i]} not found, running sig test')
            sig_map_raw = sig_test_voxels(X_test, xlbls_test, xi=plot_order[i], n_resample=10000)
            np.save(os.path.join(fig_path, f'cfl_sig_map_raw_{plot_order[i]}.npy'), sig_map_raw)

        # filter by significance
        pthresh = 0.05
        sig_map = sig_map_raw > (1 - pthresh)
        # filter by internal agreement
        cluster_thresh = 0.8
        masks = X_test[xlbls_test==plot_order[i]]
        sig_map[np.mean(masks,axis=0)<cluster_thresh] = 0
        # sig_map should now be a binary map of voxels that are significantly lesioned
        cut_coord = None
        if i<3:
            cut_coord = cut_coords[i]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax[i], cut_coords=cut_coord, 
                       vmax=1.01, threshold=1e-6, display_mode='ortho', contour_map=sig_map,
                       cbar_ticks_increment=0.5, cbarticks_right=False,
                       cbar_label='Prop. lesioned')

        for roi_idx in rois_to_plot[i]:
            orthoslicer.add_contours(rois[roi_idx], colors='darkblue', linewidths=0.25)
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means_sig.png'), dpi=300)

    ###########################################################################
    # plot single z slice with significance contour
    for i in range(n_clusters):
        try:
            sig_map_raw = np.load(os.path.join(fig_path, f'cfl_sig_map_raw_{plot_order[i]}.npy'))
        except:
            print(f'sig_map_raw_{plot_order[i]} not found, running sig test')
            sig_map_raw = sig_test_voxels(X_test, xlbls_test, xi=plot_order[i], n_resample=10000)
            np.save(os.path.join(fig_path, f'cfl_sig_map_raw_{plot_order[i]}.npy'), sig_map_raw)

        # filter by significance
        pthresh = 0.05
        sig_map = sig_map_raw > (1 - pthresh)
        # filter by internal agreement
        cluster_thresh = 0.8
        masks = X_test[xlbls_test==plot_order[i]]
        sig_map[np.mean(masks,axis=0)<cluster_thresh] = 0
        # sig_map should now be a binary map of voxels that are significantly lesioned
        fig,ax = plt.subplots(figsize=(1,1))
        lesion_heatmap(masks, mode='mean', ax=ax, cut_coords=None, colorbar=False,
                       vmax=1, threshold=1e-6, display_mode='z_ex', 
                       contour_map=sig_map, ylabel=False)
        # ax.set_ylabel('Cluster {}'.format(i))
        plt.savefig(os.path.join(fig_path, f'cfl_lesion_means_oc{i}.png'), dpi=300)

    ###########################################################################

    # sig test
    pvals_less,pvals_greater,_,smg_means = sig_test(Y_test, xlbls_test, n_resample=100000)
    fig,ax = plt.subplots(1,2,figsize=(PW,4))
    pvals = [pvals_less, pvals_greater]
    for i in range(2):
        im = ax[i].imshow(pvals[i][plot_order], aspect='auto', cmap='binary', vmin=0, vmax=1)
        ax[i].set_title(['Less', 'Greater'][i])
        ax[i].set_xlabel('Deficit')
        ax[i].set_ylabel('Cluster')
        cbar = plt.colorbar(im, ax=ax[i])
        # put star on p<0.05
        p_corrected = 0.05 / Y_test.shape[1]
        for j in range(n_clusters):
            for k in range(Y.shape[1]):
                if pvals[i][plot_order][j,k] < p_corrected:
                    ax[i].text(k,j,'*',ha='center',va='center',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_sig_test.png'), dpi=300)
    np.save(os.path.join(fig_path, 'cfl_sig_test.npy'), pvals)


    # plot deficit hists
    # FIG 4D
    fontsize = 6
    plt.rcParams.update({'font.size': fontsize,
                        'axes.labelsize': fontsize,
                        'axes.titlesize': 8,
                        'xtick.labelsize': fontsize,
                        'ytick.labelsize': fontsize,
                        'legend.fontsize': fontsize})
    fig,ax = plt.subplots(n_clusters, Y.shape[1], figsize=(0.7*CW1,0.72*n_clusters),
                          sharex=True, sharey='row')
    # global_avgs = np.mean(Y_test, axis=0)
    deficit_names_long = ['Deficit A caused\nby region A lesion', 
                          'Deficit B caused\nby region B lesion', 
                          'Deficit C caused\nby region C lesion']
    avgs = np.array([np.mean(Y_test[xlbls_test==i], axis=0) for i in range(n_clusters)])
    for i in range(n_clusters):
        for j in range(Y.shape[1]):
            ax[i,j].hist(Y_test[xlbls_test==plot_order[i],j], 
                         bins=np.arange(-0.1,1.2,0.1), color=GREEN)
            ax[i,j].axvline(avgs[plot_order[i],j], color=DARK_GREEN, linestyle=':')
            # ax[i,j].axvline(global_avgs[j], color='black', linestyle='--')
            ax[i,j].axvline(smg_means[plot_order[i],j], color='black', linestyle='--')
            ax[0,j].set_title(deficit_names_long[j], fontsize=6)
            ax[-1,j].set_xlabel('Score', labelpad=-0.5, fontsize=fontsize)
            ax[i,0].set_ylabel('Count', labelpad=-0.5, fontsize=fontsize)
    # signficance bars, have to do this separately bc y axis will change
    for i in range(n_clusters):
        for j in range(Y.shape[1]):
            if pvals_greater[plot_order[i],j] < p_corrected:
                sig_bar_min = min(avgs[plot_order[i],j], smg_means[plot_order[i],j])+0.05
                sig_bar_max = max(avgs[plot_order[i],j], smg_means[plot_order[i],j])-0.05
                ycoord = ax[i,j].get_ylim()[1]*0.86
                ax[i,j].hlines(ycoord, xmin=sig_bar_min, xmax=sig_bar_max, color='black',
                                linestyle='-', linewidth=1)
                ax[i,j].text((sig_bar_min+sig_bar_max)/2, ycoord-0.09, '*', ha='center', 
                            va='center', fontsize=10, color='black')
                
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_deficit_hists.png'), dpi=300, bbox_inches='tight',
                transparent=True)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--plot_order', type=int, nargs='+', default=None)
    parser.add_argument('--roi_idxs', type=int, nargs='+', default=EX_ROIS)
    main(**vars(parser.parse_args()))
    # plot_order: 1 3 0 2 4 5