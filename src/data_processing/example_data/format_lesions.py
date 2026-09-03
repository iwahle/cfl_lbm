'''
This script generates an example dataset of lesions for the purpose of
demonstrating the CFL pipeline. It will generate random cubes of fixed
size as lesions (masked by the MNI brain mask) and save them as nifti files.
'''

import os
import numpy as np
from src.util.constants import *
from src.util.data_util import load_brain
from src.vis.brain_vis import lesion_heatmap

def main(n_samples=2000):

    rng = np.random.default_rng(RS)

    # get mask
    mask_2mm = load_brain(os.path.join(DATA_PATH, 'vol_mask_2mm.nii.gz'), 
                        to_flatten=False)
    dims = mask_2mm.shape
    lesion_dim = 30

    # randomly sample lesion centers within full volume
    margin = lesion_dim//2
    lesion_centers = [
        rng.integers(0+margin, dims[0]-margin, n_samples),
        rng.integers(0+margin, dims[1]-margin, n_samples),
        rng.integers(0+margin, dims[2]-margin, n_samples)
    ]
    lesion_centers = np.array(lesion_centers).T

    # generate cubes around centers
    lesions = np.zeros((n_samples, dims[0], dims[1], dims[2]))
    for i in range(n_samples):
        lesions[i,lesion_centers[i,0]-margin:lesion_centers[i,0]+margin,
                lesion_centers[i,1]-margin:lesion_centers[i,1]+margin,
                lesion_centers[i,2]-margin:lesion_centers[i,2]+margin] = 1

    # flatten lesions
    lesions = lesions.reshape(n_samples, np.product(dims))

    # mask by brain mask
    mask_2mm_flat = mask_2mm.reshape(np.product(dims))
    lesions = lesions[:,np.where(mask_2mm_flat)[0]]

    print(f'lesions: {lesions.shape}')

    # split into train and test
    train_idx = rng.choice(n_samples, size=int(n_samples*0.8), replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    train_lesions = lesions[train_idx]
    test_lesions = lesions[test_idx]

    # save lesions
    if not os.path.exists(os.path.join(DATA_PATH, f'example_data')):
        os.makedirs(os.path.join(DATA_PATH, f'example_data'))
    np.save(os.path.join(DATA_PATH, f'example_data/X.npy'), train_lesions)
    np.save(os.path.join(DATA_PATH, f'example_data/X_test.npy'), test_lesions)
    print(f'Saved lesions to {os.path.join(DATA_PATH, "example_data")}')

    # visualize heatmap
    if not os.path.exists(os.path.join(FIG_PATH, 'example_data')):
        os.makedirs(os.path.join(FIG_PATH, 'example_data'))
    X = np.concatenate((train_lesions, test_lesions), axis=0)
    fig,ax = plt.subplots(figsize=(PW//2,1))
    lesion_heatmap(X, mode='sum', cut_coords=(20,0,0),
                   save_path=os.path.join(FIG_PATH, 'example_data/lesion_heatmap.png'),
                   threshold=1e-6, ax=ax, cbar_ticks_increment=100, ylabel=False, dpi=300,
                   cbar_label='# Samples')


if __name__ == '__main__':
    main()
