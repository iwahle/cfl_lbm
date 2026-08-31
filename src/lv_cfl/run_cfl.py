import os
import argparse
import numpy as np
from cfl import Experiment
import matplotlib.pyplot as plt
from src.util.constants import *
from src.lv_cfl.cfl_config import *
from matplotlib.gridspec import GridSpec
from src.vis.brain_vis import lesion_heatmap
from src.util.data_util import load_data
from cfl.util.experiment_loading import exp_load
from src.util.sig_test import sig_test, sig_test_voxels
plt.rcParams["font.size"] = FS-1


fig_path = os.path.join(FIG_PATH, 'lv_cfl')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=-1, plot_order=None):

    # load data
    X, X_test, Y, Y_test, deficit_names,_ = load_data('cohort1')
    results_path = os.path.join(RESULTS_PATH, 'lv_cfl/cfl_results')
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

    n_clusters = len(np.unique(xlbls_test))
    if plot_order is None:
        plot_order = range(n_clusters)
    else:
        assert len(plot_order)==n_clusters

    # plot lesions unraveled
    fig,ax = plt.subplots(n_clusters,1,figsize=(PW*0.9,0.9*n_clusters))
    for i in range(n_clusters):
        masks = X_test[xlbls_test==plot_order[i]]
        lesion_heatmap(masks, mode='mean', ax=ax[i], cut_coords=None, vmax=1,
                       threshold=1e-6, display_mode='z', cbar_label='Prop. lesioned',
                       cbar_width_factor=1.5)
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means_unraveled.png'), dpi=300,
                bbox_inches='tight')
    # ###########################################################################################
    # sig test X and plot on lesions
    # FIG 5A
    fig = plt.figure(figsize=(CW2/2-0.5,0.8*n_clusters))
    gs = GridSpec(n_clusters, 1, figure=fig, top=0.95, bottom=0.1, hspace=0.2)
    ax = [fig.add_subplot(gs[i,:]) for i in range(n_clusters)]
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
        print(f'cluster {i} has {np.sum(np.sum(masks,axis=0)>cluster_thresh)}')
        sig_map[np.mean(masks,axis=0)<cluster_thresh] = 0
        # sig_map should now be a binary map of voxels that are significantly lesioned
        lesion_heatmap(masks, mode='mean', ax=ax[i], cut_coords=None, 
                       vmax=1, threshold=1e-6, display_mode='ortho', contour_map=sig_map,
                       cbar_ticks_increment=0.5, cbarticks_right=False,
                       cbar_label='Prop. lesioned')
        ax[i].set_ylabel('Cluster {}'.format(i))
    plt.savefig(os.path.join(fig_path, 'cfl_lesion_means_sig.png'), dpi=300,
                transparent=True, bbox_inches='tight')

    ############################################################################################


    Yraw = np.load(os.path.join(DATA_PATH, 'cohort1/Yraw.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, 'cohort1/test_idx.npy'))
    Yraw_test = Yraw[test_idx]

    # sig test Y in original space
    pvals_less,pvals_greater,sml_means,smg_means = sig_test(Yraw_test, xlbls_test, n_resample=100000)
    print('pvals_less:', pvals_less[plot_order])
    print('pvals_greater:', pvals_greater[plot_order])
    fig,ax = plt.subplots(1,2,figsize=(PW,4))
    pvals = [pvals_less, pvals_greater]
    for i in range(2):
        im = ax[i].imshow(pvals[i][plot_order], aspect='auto', cmap='binary', vmin=0, vmax=1)
        ax[i].set_title(['Less', 'Greater'][i])
        ax[i].set_xlabel('Deficit')
        ax[i].set_ylabel('Cluster')
        cbar = plt.colorbar(im, ax=ax[i])
        # put star on p<0.05
        p_corrected = 0.05 / Y.shape[1]
        for j in range(n_clusters):
            for k in range(Y.shape[1]):
                if pvals[i][plot_order][j,k] < p_corrected:
                    ax[i].text(k,j,'*',ha='center',va='center',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_sig_test.png'), dpi=300)
    np.save(os.path.join(fig_path, 'cfl_sig_test.npy'), pvals)


    # plot lesion sizes on same figure as deficit hists
    # FIG 5B
    lesion_sizes = np.sum(np.reshape(X_test,(X_test.shape[0],-1)),axis=1)
    print(np.min(lesion_sizes), np.max(lesion_sizes))
    fig,ax = plt.subplots(n_clusters,3,figsize=(CW2/2,0.76*n_clusters), # add offset bc nilearn fussy
                          sharex='col', sharey='row')
    bins = np.linspace(0,np.max(lesion_sizes),15)
    for i in range(n_clusters):
        ax[i,0].hist(lesion_sizes[xlbls_test==plot_order[i]], bins=bins, color=BLUE)
        ax[i,0].set_ylabel('Cluster {}\nCount'.format(i+1))
        ax[i,0].set_ylabel('Count', labelpad=0)
       
    ax[-1,0].set_xlabel('Lesion size')
    fig.align_ylabels(ax)

    # FIG 5C
    avgs = np.array([np.mean(Yraw_test[xlbls_test==i], axis=0) for i in range(n_clusters)]) 
    for i in range(n_clusters):
        for j in range(Y.shape[1]):
            bins = np.linspace(np.min(Yraw_test[:,j]), np.max(Yraw_test[:,j]), 15)
            ax[i,j+1].hist(Yraw_test[xlbls_test==plot_order[i],j], color=GREEN, bins=bins)
            ax[i,j+1].axvline(avgs[plot_order[i],j], color=DARK_GREEN, linestyle=':', lw=1)
            ax[i,j+1].axvline(smg_means[plot_order[i],j], color='black', linestyle='--', lw=1)
            if i==n_clusters-1:
                ax[i,j+1].set_xlabel(deficit_names[j])
            # Set only two x axis ticks: min and max
            min_val = 0
            max_val = np.max(Yraw_test[:,j])
            ax[i,j+1].set_xticks([min_val, max_val])
            ax[i,j+1].set_xlim((0,np.max(Yraw_test[:,j])+1))

    # signficance bars, have to do this separately bc y axis will change
    for i in range(n_clusters):
        for j in range(Y.shape[1]):
            star_offset = 2 if j==1 else 5
            if i==n_clusters-1:
                continue
            elif pvals_greater[plot_order[i],j] < p_corrected:
                print('pvals_greater[plot_order[i],j]:', pvals_greater[plot_order[i],j])
                sig_bar_min = min(avgs[plot_order[i],j], smg_means[plot_order[i],j])+0.05
                sig_bar_max = max(avgs[plot_order[i],j], smg_means[plot_order[i],j])-0.05
                ax[i,j+1].set_ylim(ax[i,j+1].get_ylim()[0], ax[i,j+1].get_ylim()[1]*1.3)
                ycoord = ax[i,j+1].get_ylim()[1]*0.88
                # ycoordstar = ax[i,j+1].get_ylim()[1]*0.75
                ax[i,j+1].hlines(ycoord, xmin=sig_bar_min, xmax=sig_bar_max, color='black',
                                linestyle='-', linewidth=1)
                ax[i,j+1].text((sig_bar_min+sig_bar_max)/2, ycoord*1, '*', ha='center', 
                            va='center', fontsize=FS+2, color='black')
            elif pvals_less[plot_order[i],j] < p_corrected:
                print('pvals_less[plot_order[i],j]:', pvals_less[plot_order[i],j])
                # can still use smg_means here, sampled same way in both tests
                sig_bar_min = min(avgs[plot_order[i],j], smg_means[plot_order[i],j])+0.05
                sig_bar_max = max(avgs[plot_order[i],j], smg_means[plot_order[i],j])-0.05
                ax[i,j+1].set_ylim(ax[i,j+1].get_ylim()[0], ax[i,j+1].get_ylim()[1]*1.3)
                ycoord = ax[i,j+1].get_ylim()[1]*0.88
                ax[i,j+1].hlines(ycoord, xmin=sig_bar_min, xmax=sig_bar_max, color='black',
                                linestyle='-', linewidth=1)
                ax[i,j+1].text((sig_bar_min+sig_bar_max)/2, ycoord*1, '*', ha='center', 
                            va='center', fontsize=FS+2, color='black')   


    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(os.path.join(fig_path, 'cfl_deficit_hists.png'), dpi=300,
                transparent=True, bbox_inches='tight')


    # bar plot of etiology distribution per cluster
    etiology_mapping = np.load(os.path.join(DATA_PATH, 'cohort1/etiology_mapping.npy'), 
                               allow_pickle=True).item()
    print(etiology_mapping)
    etiology_test = np.load(os.path.join(DATA_PATH, 'cohort1/etiology_test.npy'))
    fig,ax = plt.subplots(n_clusters, figsize=(PW*2/3,1*n_clusters), sharex=True)
    for i in range(n_clusters):
        etiology_counts = np.zeros(len(etiology_mapping.keys()))
        cluster_etiologies = etiology_test[xlbls_test==plot_order[i]]
        for j in range(len(etiology_mapping)):
            etiology_counts[j] = np.sum(cluster_etiologies == j)
        ax[i].bar(etiology_mapping.keys(), etiology_counts, color=GRAY)
        ax[i].set_ylabel(f'C{i+1}')

    ax[-1].set_xticks(range(len(etiology_mapping.keys())))
    ax[-1].set_xticklabels(etiology_mapping.keys(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cfl_etiology_distribution.png'), dpi=300,
                transparent=True, bbox_inches='tight')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--plot_order', type=int, nargs='+', default=None)

    main(**vars(parser.parse_args()))
    #plot_order 1 3 4 0 2