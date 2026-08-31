import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.util.constants import *
from src.util.data_util import load_brain, load_brains_from_ids
from nilearn import plotting
import nibabel as nib
from matplotlib.colors import ListedColormap
from src.vis.brain_vis import lesion_heatmap

def get_lesion_percentage(masks, gt):

    # for each mask, measure % of ground truth that is lesioned
    lesion_percentages = []
    for mask in masks:
        lesion_percentages.append(np.sum(mask.astype(bool)&gt.astype(bool)) / np.sum(gt==1))
    return lesion_percentages

def generate_deficits(lesion_percentages, noise=0.1, to_plot=False):
    
    # generate deficits by adding noise then scaling 0-1
    deficit = np.array(lesion_percentages)
    deficit_noise = (1-noise)*deficit + np.random.normal(0, noise, len(deficit))
    deficit_noise = (deficit_noise - np.min(deficit_noise)) / (np.max(deficit_noise) - np.min(deficit_noise))

    if to_plot:
        plt.scatter(deficit, deficit_noise)
        plt.xlabel('Deficit')
        plt.ylabel('Deficit + Noise')
        plt.show()

    return deficit_noise


def main(parcel_series, atlas_fn):
    
    fig_path = os.path.join(FIG_PATH, f'loc_comparison_{parcel_series}')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    parcel_path = os.path.join(DATA_PATH, f'simulated_{parcel_series}/{atlas_fn}')
    parcels = load_brain(parcel_path)

    # plot atlas
    ni_parcels = nib.load(parcel_path)
    fig, ax = plt.subplots(1, 1, figsize=(PW,2))
    plotting.plot_roi(
        ni_parcels,
        cut_coords=(8, -4, 9),
        colorbar=True,
        cmap="Paired",
        axes=ax)
    fig.savefig(os.path.join(fig_path, f'{parcel_series}_atlas.png'), 
                bbox_inches='tight', dpi=300)

    # plot atlas
    parcels_data = ni_parcels.get_fdata().copy()[::-1] # RAS to LAS
    parcels_data[parcels_data.shape[0]//2:] = 0
    parcels_data = parcels_data[::-1] # LAS to RAS
    ni_parcels = nib.Nifti1Image(parcels_data, ni_parcels.affine, ni_parcels.header)
    fig, ax = plt.subplots(1, 1, figsize=(PW,2))
    plotting.plot_roi(
        ni_parcels,
        cut_coords=(-8, -4, 9),
        colorbar=True,
        cmap="Paired",
        axes=ax)
    fig.savefig(os.path.join(fig_path, f'{parcel_series}_atlas_left.png'), 
                bbox_inches='tight', dpi=300)

    # plot atlas and data heatmap on same figure
    X_tr = np.load(os.path.join(DATA_PATH, 'simulated/X.npy'))
    X_te = np.load(os.path.join(DATA_PATH, 'simulated/X_test.npy'))
    X = np.concatenate((X_tr, X_te), axis=0)
    fig,ax = plt.subplots(2,1, figsize=(PW-2, 3))
    orthoslicer = plotting.plot_roi(
        ni_parcels,
        cut_coords=(-8, -8, 8),
        colorbar=True,
        cmap="Paired",
        axes=ax[0],
        annotate=False)
    orthoslicer.annotate(size=FS)
    lesion_heatmap(X, mode='sum', cut_coords=(-20, 0, 20), threshold=1e-6, ax=ax[1],
                   ylabel=False, cbarticks_right=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "parcels_and_heatmap.png"), 
                             bbox_inches='tight', dpi=300)

    # zero out right hemisphere of ground truth
    parcels[parcels.shape[0]//2:] = 0

    # get list of parcels
    parcel_ids = np.unique(parcels)
    parcel_ids = parcel_ids[parcel_ids != 0] # remove background
    print(len(parcel_ids))
    
    # plot left hemisphere
    plot_parcels = nib.Nifti1Image(parcels, AFFINE, ni_parcels.header)
    fig, ax = plt.subplots(1, 1, figsize=(PW-2.5,1 ))
    cmap = np.concatenate([plt.get_cmap("Paired")(np.linspace(0, 1, 12))] * 9, axis=0)
    cmap[:,:3] += np.random.normal(0, 0.1, cmap[:,:3].shape) # make many colors
    cmap = np.clip(cmap, 0, 1)
    cmap = ListedColormap(cmap, name='custom_tab20c')
    plt.register_cmap(cmap=cmap)
    plotting.plot_roi(
        plot_parcels,
        cut_coords=(-8, -4, 9),
        colorbar=True,
        cmap=cmap,
        axes=ax)
    fig.savefig(os.path.join(fig_path, f'{parcel_series}_atlas_left.png'), 
                bbox_inches='tight', dpi=300)

    # flatten
    parcels_data = parcels_data[::-1] # RAS to LAS
    parcels = parcels.reshape(np.product(parcels.shape))

    parcel_ids = np.unique(parcels)
    parcel_ids = parcel_ids[parcel_ids != 0] # remove background

    # mask
    mask_2mm = load_brain(os.path.join('data/vol_mask_2mm.nii.gz'), 
                            to_flatten=True)
    parcels = parcels[np.where(mask_2mm)[0]]

    # make list of parcel sizes
    parcel_sizes = []
    for parcel_id in parcel_ids:
        parcel_sizes.append(np.sum(parcels == parcel_id))

    # plot parcel size histogram
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    ax.hist(parcel_sizes, bins=10)
    ax.set_xlabel('# voxels (2mm)')
    ax.set_ylabel('Count')
    ax.set_title('Parcel Size')
    # ax.set_xticks([0,4000,8000])
    plt.tight_layout()
    fig.savefig(os.path.join(fig_path, 'parcel_size_hist.png'), 
                bbox_inches='tight', dpi=300)
    

    #### TRAIN DATA ############################################################

    # load list of brain ids
    csv_path = os.path.join(DATA_PATH, 'simulated/n200_sev_linear_10noise.csv')
    ids = pd.read_csv(csv_path)['ID'].values

    # collect lesion mask for each id
    train_masks, _ = load_brains_from_ids(ids, flip=True)
    train_deficits = np.zeros((train_masks.shape[0], len(parcel_ids)))

    for pidi,pid in tqdm(enumerate(parcel_ids)):
        lp_train = get_lesion_percentage(train_masks, parcels == pid)
        deficit_train = generate_deficits(lp_train, to_plot=False)
        train_deficits[:,pidi] = deficit_train

    # plot example deficit parcel 36
    fig, ax = plt.subplots(1, 1, figsize=(1.2,1))
    ax.hist(train_deficits[36], bins=20)
    ax.set_title(f'Example deficit caused\nby parcel 36 lesion')
    ax.set_xlabel('Deficit')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'example_deficit_hist.png'), 
                bbox_inches='tight', dpi=300)
    

    #### TEST ##################################################################
    # load list of brain ids
    csv_path_test = os.path.join(DATA_PATH, 
                                    'simulated/r46_n200_sev_linear_10noise.csv')
    test_ids = pd.read_csv(csv_path_test)['ID'].values

    # collect lesion mask for each id
    test_masks, _ = load_brains_from_ids(test_ids, flip=True)
    test_deficits = np.zeros((test_masks.shape[0], len(parcel_ids)))

    for pidi,pid in tqdm(enumerate(parcel_ids)):
        # print(f'Parcel {pid}: {np.sum(parcels == pid)} voxels')
        lp_test = get_lesion_percentage(test_masks, parcels == pid)
        deficit_test = generate_deficits(lp_test, to_plot=False)
        test_deficits[:,pidi] = deficit_test

    np.save(os.path.join(DATA_PATH, f'simulated_{parcel_series}/X.npy'), train_masks)
    np.save(os.path.join(DATA_PATH, f'simulated_{parcel_series}/X_test.npy'), test_masks)
    np.save(os.path.join(DATA_PATH, f'simulated_{parcel_series}/Y.npy'), train_deficits)
    np.save(os.path.join(DATA_PATH, f'simulated_{parcel_series}/Y_test.npy'), test_deficits)
    np.save(os.path.join(DATA_PATH, f'simulated_{parcel_series}/deficit_names.npy'), parcel_ids)

if __name__ == '__main__':
    main(parcel_series='schaefer200', 
         atlas_fn=ATLAS_FN)