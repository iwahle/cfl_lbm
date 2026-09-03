"""
This script generates multiple example deficit scores, each as a
function of lesion overlap with a given ROI. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.util.data_util import load_brain
from src.vis.brain_vis import lesion_heatmap
from src.util.constants import *
from src.loc_comparison_schaefer200.format_parcel_deficits import get_lesion_percentage, generate_deficits

def main(rois=[48, 101]):
    np.random.seed(RS)

    # load mask
    mask = load_brain(os.path.join(DATA_PATH, 'vol_mask_2mm.nii.gz'), 
                      to_flatten=True)
    # load atlas
    atlas = load_brain(os.path.join(DATA_PATH, 'simulated_schaefer200', ATLAS_FN), 
                       to_flatten=True)
    print(f'atlas: {atlas.shape}')
    atlas = atlas[np.where(mask)[0]]
    print(f'atlas: {atlas.shape}')

    #### TRAIN DATA ############################################################

    # load lesions
    train_masks = np.load(os.path.join(DATA_PATH, 'example_data/X.npy'))
    print(f'train lesions: {train_masks.shape}')
    train_deficits = np.zeros((train_masks.shape[0], len(rois)))

    # compute overlap
    for roi_idx,roi in enumerate(rois):
        lp_train = get_lesion_percentage(train_masks, atlas == roi)
        deficit_train = generate_deficits(lp_train, to_plot=False)
        train_deficits[:,roi_idx] = deficit_train

    #### TEST DATA ############################################################

    # load lesions
    test_masks = np.load(os.path.join(DATA_PATH, 'example_data/X_test.npy'))
    print(f'test lesions: {test_masks.shape}')
    test_deficits = np.zeros((test_masks.shape[0], len(rois)))

    # compute overlap
    for roi_idx,roi in enumerate(rois):
        lp_test = get_lesion_percentage(test_masks, atlas == roi)
        deficit_test = generate_deficits(lp_test, to_plot=False)
        test_deficits[:,roi_idx] = deficit_test

    ###########################################################################

    # save deficits
    np.save(os.path.join(DATA_PATH, 'example_data/Y.npy'), train_deficits)
    np.save(os.path.join(DATA_PATH, 'example_data/Y_test.npy'), test_deficits)
    np.save(os.path.join(DATA_PATH, 'example_data/deficit_names.npy'), 
                         [f'roi_{roi}' for roi in rois])
    print(f'Saved deficits to {os.path.join(DATA_PATH, "example_data")}')

    all_deficits = np.concatenate((train_deficits, test_deficits), axis=0)

    # plot distributions
    # set font sizes
    plt.rcParams.update({'font.size': FS})
    plt.rcParams.update({'axes.labelsize': FS})
    plt.rcParams.update({'axes.titlesize': FS})
    plt.rcParams.update({'xtick.labelsize': FS})
    plt.rcParams.update({'ytick.labelsize': FS})
    fig,ax = plt.subplots(1, len(rois), figsize=(PW//2,2), sharex=True, sharey=True)
    for roi_idx,roi in enumerate(rois):
        ax[roi_idx].hist(all_deficits[:,roi_idx], bins=20)
        ax[roi_idx].set_title(f'Example deficit caused\nby {roi} lesion')
        ax[roi_idx].set_xlabel('Deficit')
        ax[roi_idx].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'example_data/deficit_distributions.png'),
                bbox_inches='tight', dpi=300)

    # plot the ROIs
    fig,ax = plt.subplots(len(rois),1,figsize=(PW//2,1.2*len(rois)))
    for roi_idx,roi in enumerate(rois):
        roi_mask = np.zeros_like(atlas)
        roi_mask[np.where(atlas == roi)[0]] = 1
        orthoslicer = lesion_heatmap(roi_mask[None,:], mode='mean', ax=ax[roi_idx], 
                                     cut_coords=None, vmax=1, ylabel=False,
                                     threshold=1e-6, colorbar=False)
        ax[roi_idx].set_title(f'ROI {roi}')
        ax[roi_idx].set_xlabel('Deficit')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'example_data/roi_masks.png'),
                bbox_inches='tight', dpi=300)
if __name__ == '__main__':
    main()