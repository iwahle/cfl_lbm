
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.util.constants import *
from src.util.data_util import load_brain, load_brains_from_ids, orient_brain
import nibabel as nib
from src.vis.brain_vis import lesion_heatmap

def vis_heatmap(set_type):
    # Load the brain data
    suffix = '' if set_type=='train' else '_test'
    found_subj = np.load(os.path.join(DATA_PATH, f'simulated/found_subj{suffix}.npy'))
    masks,_ = load_brains_from_ids(found_subj, flip=False, data_series='lesion_masks_lh')

    fig,ax = plt.subplots(1,1,figsize=(PW//2,1.5))
    lesion_heatmap(masks, mode='sum', save_path=None, figure=fig, ax=ax,
                   cut_coords=None, colorbar=True, cmap=BLUE_CONT, vmin=0, vmax=None, 
                   threshold=1e-6, display_mode='ortho', ylabel=True)
    plt.show()

def main():

    # load IDs of lesions that are in right hemi to flip to left
    flipped_ids = pd.read_csv(os.path.join('data/simulated/flipped_lesions.csv'))['flip_ids'].values
    
    #### TRAIN DATA ############################################################
    # load list of brain ids
    found_subj = np.load(os.path.join(DATA_PATH, 'simulated/found_subj.npy'))
    for id in found_subj:
        fp = os.path.join(DATA_PATH, 'lesion_masks', f'{str(id).zfill(4)}.nii.gz')
        img = nib.load(fp)
        img = orient_brain(img, 'RAS')
        data = img.get_fdata()
        if id in flipped_ids:
            data = np.flip(data, axis=0)
        img = nib.Nifti1Image(data, img.affine, img.header)
        img.set_data_dtype(np.uint8)
        img.header.set_data_dtype(np.uint8)
        fp_save = os.path.join(DATA_PATH, 'lesion_masks_lh', f'{str(id).zfill(4)}.nii.gz')
        img.to_filename(fp_save)
        

    #### TEST DATA ############################################################
    # load list of brain ids
    found_subj_test = np.load(os.path.join(DATA_PATH, 'simulated/found_subj_test.npy'))
    for id in found_subj_test:
        fp = os.path.join(DATA_PATH, 'lesion_masks', f'{str(id).zfill(4)}.nii.gz')
        img = nib.load(fp)
        img = orient_brain(img, 'RAS')
        data = img.get_fdata()
        if id in flipped_ids:
            data = np.flip(data, axis=0)
        img = nib.Nifti1Image(data, img.affine, img.header)
        img.set_data_dtype(np.uint8)
        img.header.set_data_dtype(np.uint8)
        fp_save = os.path.join(DATA_PATH, 'lesion_masks_lh', f'{str(id).zfill(4)}.nii.gz')
        img.to_filename(fp_save)


if __name__ == '__main__':
    main()
    vis_heatmap(set_type='train')
    vis_heatmap(set_type='test')