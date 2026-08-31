import os
import numpy as np
from nilearn.image import resample_img, load_img
from src.util.constants import BRAIN_DIMS, DATA_PATH

def resample(id):
    # load lesion mask
    mask = load_img(os.path.join('data', f'lesion_masks/grafman_1mm/{id}_lesion.nii.gz'))
    target_affine = mask.affine.copy()
    target_affine[:3,:3] *= 2
    # resample to 2mm
    mask_resampled = resample_img(mask, target_affine=target_affine,
                                  target_shape=BRAIN_DIMS, 
                                  interpolation='nearest')
    assert list(mask_resampled.get_fdata().shape) == BRAIN_DIMS
    # save resampled mask
    fn = os.path.join('data', f'lesion_masks/{id}_lesion.nii.gz')
    mask_resampled.to_filename(fn)


def main():
    
     # load list of brain ids
    ids = np.load(os.path.join(DATA_PATH, 'cohort2/subject_ids.npy'),
                  allow_pickle=True).astype(str)
    print('cohort size: ', len(ids))
    for id in ids:
        if 'grafman' in id:
            resample(id)
    
if __name__ == '__main__':
    main()
