import os
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *
from src.vis.brain_vis import lesion_heatmap
from matplotlib.gridspec import GridSpec

fig_path = os.path.join(FIG_PATH, 'data_summary')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main():
    # set font size for all xlabels
    plt.rcParams.update({'font.size': FS})

    cut_coordss = [(-20,0,20), (20,0,0), (20,0,0)]
    fig = plt.figure(figsize=(CW1,4.5))
    gs_top = GridSpec(3, 1, figure=fig, top=1, bottom=0.5, hspace=0.2)
    gs_bot = GridSpec(2, 3, figure=fig, top=0.48, bottom=0.2, wspace=0.4, hspace=0.8)

    heatmap_axs = [fig.add_subplot(gs_top[i,:]) for i in range(3)]

    # FIG 2A, 2B, 2C
    for i, dataset in enumerate(['simulated', 'cohort1', 'cohort2']):
        X_tr = np.load(os.path.join(DATA_PATH, f'{dataset}/X.npy'))
        X_te = np.load(os.path.join(DATA_PATH, f'{dataset}/X_test.npy'))
        X = np.concatenate((X_tr, X_te), axis=0)
        lesion_heatmap(X, mode='sum', cut_coords=cut_coordss[i], 
                       threshold=1e-6, ax=heatmap_axs[i], 
                       cbar_ticks_increment=25, 
                       cbar_fancy_scaling=True,
                       cbarticks_right=False, 
                       ylabel=False)

    sim_dist_ax = fig.add_subplot(gs_bot[0,0])
    Y = np.concatenate([
        np.load(os.path.join(DATA_PATH, f'simulated_schaefer200/Y.npy')),
        np.load(os.path.join(DATA_PATH, f'simulated_schaefer200/Y_test.npy'))])
    # plot example deficit parcel 36
    # FIG 2D
    sim_dist_ax.hist(Y[36], bins=20, color=GREEN)
    sim_dist_ax.set_xlabel('Parcel 36 lesion deficit', labelpad=1)
    sim_dist_ax.set_ylabel('Count', labelpad=1)

    # FIG 2E, 2F
    lang_dist_ax = fig.add_subplot(gs_bot[0,1])
    vis_dist_ax = fig.add_subplot(gs_bot[0,2])
    lv_axs = [lang_dist_ax, vis_dist_ax]
    Y = np.load(os.path.join(DATA_PATH, f'cohort1/Yraw.npy'))
    deficit_names = ['Language score', 'Visuospatial score']
    deficit_histograms(Y, deficit_names, axs=lv_axs)
    for i, ax in enumerate(lv_axs):
        ax.set_ylabel('')
        ax.set_xlabel(deficit_names[i], labelpad=1)
    
    # FIG 2G
    bdi_dists_ax = fig.add_subplot(gs_bot[1,:])
    Y = np.load(os.path.join(DATA_PATH, f'cohort2/Yraw.npy'))
    deficit_names = np.load(os.path.join(DATA_PATH, f'cohort2/deficit_names.npy'))
    ylbls = np.zeros(Y.shape[0])
    bdi_dists(Y, ylbls, titles=deficit_names, axs=bdi_dists_ax)

                       
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'lesion_heatmaps.png'), dpi=300, transparent=True)

    # plot unraveled lesion heatmaps
    fig,ax = plt.subplots(3,1,figsize=(PW,3))
    for i, dataset in enumerate(['simulated', 'cohort1', 'cohort2']):
        X_tr = np.load(os.path.join(DATA_PATH, f'{dataset}/X.npy'))
        X_te = np.load(os.path.join(DATA_PATH, f'{dataset}/X_test.npy'))
        X = np.concatenate((X_tr, X_te), axis=0)
        lesion_heatmap(X, mode='sum', display_mode='z',
                       threshold=1e-6, ax=ax[i], cbar_width_factor=1.5,
                       cbar_label='Lesion frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'unraveled_lesion_heatmaps.png'), dpi=300, 
                transparent=True)

if __name__ == '__main__':
    main()