import os
import argparse
import numpy as np
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from scipy.stats import ttest_ind
from src.dep_cfl.cfl_config import *
from src.util.data_util import load_data
from src.vis.deficit_vis import bdi_dists
from src.vis.brain_vis import lesion_heatmap
from cfl.util.experiment_loading import exp_load
from src.vis.draw_relation import draw_relation, draw_relation_sankey
from src.util.sig_test import sig_test, sig_test_relation, sig_test_voxels


fig_path = os.path.join(FIG_PATH, 'dep_cfl')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, plot_order_c=None, plot_order_e=None):

    # load data
    X, X_test, Y, Y_test, deficit_names,_ = load_data('cohort2')
    results_path = os.path.join(RESULTS_PATH, 'dep_cfl/cfl_results')
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
        ylbls_test = test_results['EffectClusterer']['y_lbls']

    else:
        # load results
        xlbls_test = exp_load(results_path, exp_id, 'test', 'CauseClusterer', 
                              'x_lbls')
        ylbls_test = exp_load(results_path, exp_id, 'test', 'EffectClusterer',
                              'y_lbls')
        

    n_cclusters = len(np.unique(xlbls_test))
    n_eclusters = len(np.unique(ylbls_test))
    if plot_order_c is None:
        plot_order_c = range(n_cclusters)
    else:
        assert len(plot_order_c)==n_cclusters

    # plot lesions
    fig,ax = plt.subplots(n_cclusters,1,figsize=(PW//2,1*n_cclusters))
    for i in range(n_cclusters):
        masks = X_test[xlbls_test==plot_order_c[i]]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax[i], 
                                    cut_coords=None, vmax=1,
                                    threshold=1e-6)
        orthoslicer._colorbar_ax.locator_params(nbins=3)
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means.png'), dpi=300)

    fig,ax = plt.subplots(n_cclusters,1,figsize=(PW,n_cclusters))
    for i in range(n_cclusters):
        masks = X_test[xlbls_test==plot_order_c[i]]
        orthoslicer = lesion_heatmap(masks, mode='mean', ax=ax[i], 
                                    cut_coords=None, vmax=1,
                                    threshold=1e-6, display_mode='z', 
                                    cbar_fancy_scaling=False,
                                    cbar_width_factor=1.5,
                                    cbar_label='Prop. lesioned')
        orthoslicer._colorbar_ax.locator_params(nbins=3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means_unraveled.png'), dpi=300,
                bbox_inches='tight')

    # plot lesions single z slice for sankey
    for i in range(n_cclusters):
        fig,ax = plt.subplots(figsize=(1,1))
        masks = X_test[xlbls_test==plot_order_c[i]]
        lesion_heatmap(masks, mode='mean', ax=ax, cut_coords=None, 
                       vmax=1, threshold=1e-6, display_mode='z_ex',
                       colorbar=False, ylabel=False)
        plt.savefig(os.path.join(fig_path, f'cfl_lesion_means_oc{i}_nc{plot_order_c[i]}.png'), dpi=300)


    ###########################################################################################
    # sig test X and plot on lesions
    # FIG 6A
    print('Starting sig test on X')
    # fig,ax = plt.subplots(n_cclusters,1,figsize=(CW1*0.7,0.8*n_cclusters))
    fig,ax = plt.subplots(n_cclusters,1,figsize=(CW1*0.8,0.9*n_cclusters))
    for i in range(n_cclusters):
        try:
            sig_map_raw = np.load(os.path.join(fig_path, f'cfl_sig_map_raw_{plot_order_c[i]}.npy'))
        except:
            print(f'sig_map_raw_{plot_order_c[i]} not found, running sig test')
            sig_map_raw = sig_test_voxels(X_test, xlbls_test, xi=plot_order_c[i], n_resample=10000)
            np.save(os.path.join(fig_path, f'cfl_sig_map_raw_{plot_order_c[i]}.npy'), sig_map_raw)
        pthresh = 0.05
        sig_map = sig_map_raw > (1 - pthresh)
        # filter by internal agreement
        cluster_thresh = 0.8
        masks = X_test[xlbls_test==plot_order_c[i]]
        print(f'cluster {i} has {np.sum(np.sum(masks,axis=0)>cluster_thresh)}')
        sig_map[np.mean(masks,axis=0)<cluster_thresh] = 0
        print(f'cluster {i} has {np.sum(sig_map)}')
            
        # sig_map should now be a binary map of voxels that are significantly lesioned
        lesion_heatmap(masks, mode='mean', ax=ax[i], cut_coords=None, 
                       vmax=1, threshold=1e-6, display_mode='ortho', contour_map=sig_map,
                       cbar_ticks_increment=0.5, cbar_label='Prop. lesioned')
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means_sig.png'), dpi=300)

    ############################################################################################

    # plot lesion sizes
    lesion_sizes = np.sum(np.reshape(X_test,(X_test.shape[0],-1)),axis=1)
    bins = np.linspace(0,np.max(lesion_sizes),20)
    fig,ax = plt.subplots(n_cclusters,1,figsize=(2,1*n_cclusters))
    for i in range(n_cclusters):
        ax[i].hist(lesion_sizes[xlbls_test==plot_order_c[i]], bins=bins, color=BLUE)
        ax[i].set_ylabel('Cluster {}\nCount'.format(i+1))
    ax[-1].set_xlabel('Lesion size')
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_sizes.png'), dpi=300)

    if plot_order_e is None:
        plot_order_e = range(n_eclusters)
    else:
        assert len(plot_order_e)==n_eclusters

    # plot deficit hists
    # FIG 6C
    Yraw = np.load(os.path.join(DATA_PATH, 'cohort2/Yraw.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, 'cohort2/test_idx.npy'))
    Yraw_test = Yraw[test_idx]

    bdi_dists(Yraw_test, ylbls_test, deficit_names, 
              os.path.join(fig_path, 'cfl_deficit_hists.png'),
              figsize=(CW1*0.8,0.65*n_eclusters),
              plot_order=plot_order_e, labeled=False)
    bdi_dists(Yraw_test, ylbls_test, deficit_names,
              os.path.join(fig_path, 'cfl_deficit_hists_labeled.png'),
              plot_order=plot_order_e, labeled=True)


    # plot mean bdi dists by cause cluster
    Yraw_test_mean = np.mean(Yraw_test, axis=1)[:,None]
    fig,ax = plt.subplots(n_cclusters, figsize=(PW//6,1*n_cclusters), 
                          sharex='col')
    global_mean = np.mean(Yraw_test_mean)
    for i in range(n_cclusters):
        bins = np.linspace(np.min(Yraw_test_mean), np.max(Yraw_test_mean),5)
        ax[i].hist(Yraw_test_mean[xlbls_test==plot_order_c[i]], color=GREEN, 
                     bins=bins)
        avg = np.mean(Yraw_test_mean[ylbls_test==plot_order_c[i]])
        ax[i].axvline(avg, color='green', linestyle='--', lw=2)
        ax[i].axvline(global_mean, color='black', linestyle='--')
        ax[-1].set_xlabel('Mean BDI')
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_mbdi_hists_by_cc.png'), dpi=300)

    # plot mean bdi dists
    Yraw_test_mean = np.mean(Yraw_test, axis=1)[:,None]
    fig,ax = plt.subplots(n_eclusters, figsize=(PW//6,1*n_eclusters), 
                          sharex='col')
    global_mean = np.mean(Yraw_test_mean)
    for i in range(n_eclusters):
        bins = np.linspace(np.min(Yraw_test_mean), np.max(Yraw_test_mean),5)
        ax[i].hist(Yraw_test_mean[ylbls_test==plot_order_e[i]], color=GREEN, 
                     bins=bins)
        avg = np.mean(Yraw_test_mean[ylbls_test==plot_order_e[i]])
        ax[i].axvline(avg, color='green', linestyle='--', lw=2)
        ax[i].axvline(global_mean, color='black', linestyle='--')
        ax[-1].set_xlabel('Mean BDI')
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_mbdi_hists_by_ec.png'), dpi=300)
    
    # sig test relation
    pvalsl_rel,pvalsg_rel = sig_test_relation(xlbls_test, ylbls_test, 
                                              n_resample=100000)
    print('pvalsl_rel:', pvalsl_rel)
    print('pvalsg_rel:', pvalsg_rel)
    fig,ax = plt.subplots(1,2,figsize=(PW,4))
    pvals = [pvalsl_rel, pvalsg_rel]
    p_thresh = 0.05 / (n_cclusters * n_eclusters)
    for i in range(2):
        im = ax[i].imshow(pvals[i][plot_order_c][:,plot_order_e], aspect='equal', 
                          cmap='binary', vmin=0, vmax=1)
        ax[i].set_title(['Less', 'Greater'][i])
        ax[i].set_xlabel('Effect')
        ax[i].set_ylabel('Cause')
        ax[i].set_xticks(range(n_eclusters))
        ax[i].set_yticks(range(n_cclusters))
        cbar = plt.colorbar(im, ax=ax[i])
        # put star on p<0.05
        for j in range(n_cclusters):
            for k in range(n_eclusters):
                if pvals[i][plot_order_c[j],plot_order_e[k]] < p_thresh:
                    ax[i].text(k,j,'*',ha='center',va='center',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_sig_test_relation.png'), dpi=300)

    # draw weights between macrostates
    # FIG 6B
    draw_relation(cms=xlbls_test, ems=ylbls_test, plot_order_c=plot_order_c,
                  plot_order_e=plot_order_e,
                  sigs_lt=pvalsl_rel, sigs_gt=pvalsg_rel, sig_thresh=p_thresh,
                  save_path=os.path.join(fig_path, 'cfl_relation.png'))
    draw_relation_sankey(xlbls_test, ylbls_test,
                         save_path=os.path.join(fig_path, 'cfl_relation_sankey.png'))

    # sig test total bdi
    pvals_less,pvals_greater,_,_ = sig_test(Yraw_test_mean, ylbls_test, n_resample=100000)
    print('pvals_less:', pvals_less)
    print('pvals_greater:', pvals_greater)
    fig,ax = plt.subplots(1,2,figsize=(PW,4))
    pvals = [pvals_less, pvals_greater]
    for i in range(2):
        im = ax[i].imshow(pvals[i][plot_order_e], aspect='auto', cmap='binary', vmin=0, vmax=1)
        ax[i].set_title(['Less', 'Greater'][i])
        ax[i].set_xlabel('Deficit')
        ax[i].set_ylabel('Cluster')
        cbar = plt.colorbar(im, ax=ax[i])
        # put star on p<0.05
        for j in range(n_eclusters):
            for k in range(Yraw_test_mean.shape[1]):
                if pvals[i][plot_order_e][j,k] < 0.05:
                    ax[i].text(k,j,'*',ha='center',va='center',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_sig_test.png'), dpi=300)
    np.save(os.path.join(fig_path, 'cfl_sig_test.npy'), pvals)


    # t-test between mean bdi of cause clusters
    print('ttest mbdi of vmpfc < non-localized:')
    print(ttest_ind(Yraw_test_mean[xlbls_test==plot_order_c[0]], 
                    Yraw_test_mean[xlbls_test==plot_order_c[2]], 
                    alternative='less'))
    print('ttest mbdi of dlpfc > non-localized:')
    print(ttest_ind(Yraw_test_mean[xlbls_test==plot_order_c[1]],
                    Yraw_test_mean[xlbls_test==plot_order_c[2]], alternative='greater'))


    # demographics per cluster (even though not included in CFL inputs,
    # for comparison to dep_dems)
    dems = np.load(os.path.join(DATA_PATH, 'cohort2/dems_raw.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, 'cohort2/test_idx.npy'))
    dems = dems[test_idx]
    _,_,_,_,_,dem_names = load_data('cohort2', dems=True)
    fig,ax = plt.subplots(n_cclusters, dems.shape[1], figsize=(PW//6*3.5,1*n_cclusters),
                          sharex='col')
    global_means = np.mean(dems, axis=0) 
    for i in range(n_cclusters):
        for j in range(dems.shape[1]):
            bins = np.linspace(np.min(dems[:,j]), np.max(dems[:,j]), 20)
            if dem_names[j]=='Sex':
                bins = [-0.2,0.2,0.8,1.2]
            ax[i,j].hist(dems[xlbls_test==plot_order_c[i],j], color=GREEN, bins=bins)
            avg = np.mean(dems[xlbls_test==plot_order_c[i],j])
            ax[i,j].axvline(avg, color='green', linestyle='--', lw=2)
            ax[i,j].axvline(global_means[j], color='black', linestyle='--')
            if i==n_cclusters-1:
                ax[i,j].set_xlabel(dem_names[j])
            if dem_names[j]=='Sex':
                ax[i,j].set_xticks([0,1])
                ax[i,j].set_xticklabels(['M', 'F'])
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_dems_by_cluster.png'), dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--plot_order_c', type=int, nargs='+', default=None)
    parser.add_argument('--plot_order_e', type=int, nargs='+', default=None)

    main(**vars(parser.parse_args()))
    # plot_order_c: 0 2 1
    # plot_order_e: 3 0 1 2