import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.util.constants import *
from src.cca_comparison.cfl_config import *
from src.vis.brain_vis import lesion_heatmap
from src.util.data_util import load_data
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from src.cca_comparison.standard_cut_coords import *

fig_path = os.path.join(FIG_PATH, 'cca_comparison')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(roi_idxs=EX_ROIS, parcel_series='schaefer200', 
    atlas_fn=ATLAS_FN):

    # load data
    X, X_test, Y, Y_test, deficit_names,_ = load_data('simulated_schaefer200')
    Y = Y[:,roi_idxs]
    Y_test = Y_test[:,roi_idxs]
    results_path = os.path.join(RESULTS_PATH, 'cca_comparison/cca_results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # pca reduction
    pca = PCA(n_components=30)
    Xsmall = pca.fit_transform(X-X.mean(axis=0))
    Xsmall_test = pca.transform(X_test-X_test.mean(axis=0))
    print('exp var 30 PC: ', np.sum(pca.explained_variance_ratio_))

    # fit cca
    n_components = Y.shape[1]
    cca = CCA(n_components=n_components)
    cca.fit(Xsmall, Y)
    weights = [pca.inverse_transform(cca.x_rotations_[:,i]) for i in range(n_components)]


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


    # plot voxel weights
    # FIG 4A
    fig,ax = plt.subplots(n_components,1,figsize=(CW1*0.75,0.8*n_components))
    cut_coords = [A_coords, B_coords, C_coords]
    for wi,w in enumerate(weights):        
        orthoslicer = lesion_heatmap(w[None,:], mode='mean', ax=ax[wi], 
                                     cut_coords=cut_coords[wi], vmin=None, vmax=None, 
                                     cmap='coolwarm', ylabel=False,
                                     cbar_tick_format='%.2f', 
                                     cbarticks_right=False,
                                     cbar_fancy_scaling=False,
                                     cbar_label='Weight',
                                     threshold=1e-6) # had to do this to get 
                                                     # rid of bg?? nl.p bug
        # set fewer ticks on cbar
        orthoslicer._colorbar_ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        ax[wi].set_ylabel('Component {}'.format(wi))
    plt.savefig(os.path.join(fig_path, 'cca_weights.png'), dpi=300)

    # plot deficit weights
    # FIG 4B
    plt.rcParams.update({'font.size': FS,
                        'axes.labelsize': FS,
                        'axes.titlesize': 8,
                        'xtick.labelsize': FS,
                        'ytick.labelsize': FS,
                        'legend.fontsize': FS})
    vb = np.max(np.abs(cca.y_rotations_))
    fig,ax = plt.subplots(figsize=(1.5,1.5))
    im = ax.imshow(cca.y_rotations_, cmap='coolwarm', vmin=-vb, vmax=vb)
    ax.set_xticks(range(n_components))
    ax.set_xticklabels(['A','B','C'])
    ax.set_ylabel('Mode')
    ax.set_yticks(range(n_components))
    ax.set_yticklabels(range(1,n_components+1))
    plt.colorbar(im, shrink=0.6, label='Weight')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'cca_deficit_weights.png'), dpi=300, 
        bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_idxs', type=int, nargs='+', default=EX_ROIS)
    main(**vars(parser.parse_args()))