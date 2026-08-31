import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion


def load_data(dataset, dems=False):
    X = np.load(f'data/{dataset}/X.npy')
    X_test = np.load(f'data/{dataset}/X_test.npy')
    Y = np.load(f'data/{dataset}/Y.npy')
    Y_test = np.load(f'data/{dataset}/Y_test.npy')
    deficit_names = np.load(f'data/{dataset}/deficit_names.npy')
    dem_names = None

    if dems:
        dems = np.load(f'data/{dataset}/dems.npy')
        dems_test = np.load(f'data/{dataset}/dems_test.npy')
        dem_names = np.load(f'data/{dataset}/dem_names.npy')
        X = np.concatenate((X, dems), axis=1)
        X_test = np.concatenate((X_test, dems_test), axis=1)
    return X, X_test, Y, Y_test, deficit_names, dem_names


def orient_brain(img, ori='RAS'):
    ''' reorients a 3D brain array to a specified orientation'''


    # convert the affine transformation matrix (a matrix) to axis codes (a string eg 'RAS')
    cur_ori = nib.orientations.aff2axcodes(img.affine)
    target_ori = tuple(ori)

    # check that we have a valid orientation
    assert cur_ori[0] in ['R', 'L'], \
        f'is {cur_ori[0]}, should be R or L' # right, left
    assert cur_ori[1] in ['A', 'P'], \
        f'is {cur_ori[1]}, should be A or P' # anterior, posterior
    assert cur_ori[2] in ['S', 'I'], \
        f'is {cur_ori[2]}, should be S or I' # superior, inferior

    # if the orientation of the image is different than the specified orentation,
    # flip one axis of the affine transformation
    if cur_ori[0] != target_ori[0]:
        img = img.slicer[::-1, :, :]
    if cur_ori[1] != target_ori[1]:
        img = img.slicer[:, ::-1, :]
    if cur_ori[2] != target_ori[2]:
        img = img.slicer[:, :, ::-1]

    # and check that the orientations are equal now
    cur_ori = nib.orientations.aff2axcodes(img.affine)
    assert (cur_ori == target_ori), f"Problem with orientation: {cur_ori}, {target_ori}"

    return img

def load_brain(fp, to_flatten=False, mask=None, ori='RAS', flip_mask=False,
               dtype=np.float32):
    ''' loads one nii.gz file and reorients as needed
    arguments:
        fp: file path (from HOME_PATH) to nii.gz file (string)
        to_flatten: whether to return as 3D array or 1D flattened array (boolean)
        mask: 3D array specifying voxels to retain. If none, retains all voxels
        ori: what order voxels should be oriented in the 3D array (i.e. 'LAS', 'RAS', etc.)
            'LAS' = right-Left within posterior-Anterior within inferior-Superior
            'RAS' = left-Right within posterior-Anterior within inferior-Superior
            more info here: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    returns:
        img: if flatten is true, a 1D array. Otherwise, a 3D array
    '''

    #get the image information from the specified path
    img = nib.load(fp)

    img = orient_brain(img, ori)
    
    # flip mask if requested along R-L axis
    if flip_mask:
        img = img.slicer[::-1, :, :]

    # get the actual image array
    img = img.get_fdata()

    # flatten the image, and (if applicable) the mask, to 1D
    if to_flatten:
        img = np.reshape(img, int(np.product(img.shape)))
        if np.all(mask is not None):
            mask = np.reshape(mask, int(np.product(mask.shape)))

    #if a mask was given, apply the mask to the image
    if np.all(mask is not None):
        img = img[np.where(mask)[0]]
        assert len(img) == np.sum(mask, dtype=np.float32)

    return img.astype(dtype)


def load_brains_from_ids(ids, flip=False, data_series='lesion_masks'):
    ''' loads a list of brain masks from a list of IDs
    arguments:
        ids: list of IDs to load (list of strings)
        flip: whether to flip right hemi lesions to left (boolean)
        data_series: folder to load from (string)
    returns:
        masks: 2D array where each row is a flattened 3D array lesion mask
        found_subj_ids: list of IDs that were found
    '''

    # load brain volume mask
    mask_2mm = load_brain(os.path.join('data/vol_mask_2mm.nii.gz'))

    # load IDs of lesions that are in right hemi to flip to left
    flipped_ids = pd.read_csv(os.path.join('data/simulated/flipped_lesions.csv'))['flip_ids'].values
    
    # collect lesion mask for each id
    masks = []
    found_subj_ids = []
    for i,id in tqdm(enumerate(ids)):
        try:
            flip_mask = False
            if flip:
                if id in flipped_ids:
                    flip_mask = True
            
            fn = f'{data_series}/{str(id).zfill(4)}.nii.gz'
            if 'grafman' in str(id):
                fn = f'{data_series}/{id}_lesion.nii.gz'
            masks.append(load_brain(os.path.join('data', fn),
                mask=mask_2mm, to_flatten=True, flip_mask=flip_mask))
            found_subj_ids.append(id)
        except Exception as e:
            print(e)
            print(f'Error loading {id}')
    print('Number of masks not found: ', len(ids)-len(masks), '/', len(ids))
    masks = np.array(masks)
    found_subj_ids = np.array(found_subj_ids)
    return masks, found_subj_ids



def dice_eval(map_pred, map_gt, thresh=0.75):
    '''
    This is a spatial overlap index between two areas, rangingbetween 0 and 1. 
    For example, a perfect overlap between predicted and manual lesions would be 
    equal to 1, while nooverlap at all would be equal to 0. The formula for its 
    calculation divides the overlapping area by the sum area occupied by both 
    masks, multiplied by two: “[(A & B)*2/(A | B)]”.
    https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.23110?src=getftr
    Pustina 2016
    '''

    if np.any(map_pred > 0):
        map_gt = map_gt.astype(bool)
        return (np.sum(map_gt & map_pred) * 2.0) / (np.sum(map_gt) + np.sum(map_pred))
    else:
        print('No non-zero elements in map_pred')
        return np.nan

def peak_disp_eval(map_pred, map_gt, thresh=0.75):
    '''
    map_gt and map_pred are 3D arrays
    
    https://www.sciencedirect.com/science/article/pii/S0028393217303214#s0010
    We assigned PV-D a value of zero if the peak voxel fell within the 
    simulated brain parcel, otherwise we computed the distance of the peak 
    voxel to the closest point of the parcel that generated it.'''  

    if np.any(map_pred > 0):
        # Get index of maximum voxel value
        max_inds = np.where(map_pred == np.max(map_pred))
        max_inds = np.array(max_inds).T

        # Get target indices
        t_inds = np.array(np.nonzero(map_gt)).T

        # if several max_inds (which is more likely with CFl since we are using
        # mask averages), check all of them
        peak_displacement = np.inf
        for i in range(max_inds.shape[0]): # loop through peaks
            max_ind = max_inds[i]
            # Check if peak is in target
            if np.any(np.all(t_inds==max_ind)):
                peak_displacement = 0
                return peak_displacement
            else:
                # Get peak displacement
                distances = cdist(t_inds, max_ind[None,:], metric='euclidean')
                min_dist = distances.min()
                if min_dist < peak_displacement:
                    peak_displacement = min_dist
    else:
        print('No non-zero elements in map_pred')
        peak_displacement = np.nan
    return peak_displacement 


def centroid_disp_eval(map_pred, map_gt, thresh=0.75):

    # Get indices of non-zero voxel values in map_pred
    pred_inds = np.nonzero(map_pred)
    pred_coords = np.array(pred_inds).T
    
    if pred_inds[0].size > 0:
        # Get target indices
        gt_inds = np.nonzero(map_gt)
        gt_coords = np.array(gt_inds).T
        assert gt_inds[0].size > 0, 'No non-zero elements in map_gt'

        # Get centroid
        centroid = pred_coords.mean(axis=0)
  
        # Get minimum distances between target coords and centroid
        centroid_displacement = cdist(gt_coords, centroid.reshape(1, -1), 
                                      metric='euclidean').min()
        
    else:
        print('No non-zero elements in map_pred')
        centroid_displacement = np.nan
    
    return centroid_displacement

def contour_disp_eval(map_pred, map_gt):
    '''This is a metric of the average distance from the contour of the 
    predicted map to the closest point of the brain parcel that generated the 
    map. Different from dice, this measure is not sensitive to parcel size.
    
    '''

    # Check if map_pred contains any non-zero elements
    if np.any(map_pred > 0):

        # Get contour of map_pred (similar to bwperim in MATLAB)
        structure = np.ones((3, 3, 3), dtype=bool)  # 3x3x3 connectivity
        map_pred_contour = binary_erosion(map_pred, structure=structure) ^ map_pred
        
        # Get indices of contour voxel values
        m_inds = np.nonzero(map_pred_contour)
        
        # Get target indices
        t_inds = np.nonzero(map_gt)
        
        # Coordinates of contour
        pred_coords = np.array(m_inds).T
        
        # Coordinates of target voxels
        gt_coords = np.array(t_inds).T
        
        # Get distances between contour and target coords, find closest gt
        # to each pred
        distances = cdist(pred_coords, gt_coords, metric='euclidean').min(axis=1)

        # Find mean distance
        contour_displacement = np.mean(distances)
    else:
        print('No non-zero elements in map_pred')
        contour_displacement = np.nan
    
    return contour_displacement

def multi_region_disp_eval(map_pred, map_gts, thresh=0.75, thresh_nonzero=False):
    '''
    maps_gts is now a list of all the distinct gt regions that were used to
    generate the deficit

    gets smallest distance from pred to each gt region, return average
    '''

    # Check if map_pred contains any non-zero elements
    if np.any(map_pred > 0):

        min_dists = []
        for map_gt in map_gts:

            # Get indices of non-zero voxel values in map_pred
            pred_inds = np.nonzero(map_pred)
            pred_coords = np.array(pred_inds).T

            # Get target indices
            gt_inds = np.nonzero(map_gt)
            gt_coords = np.array(gt_inds).T

            # Compute distances from all points in map_pred to map_gt
            distances = cdist(pred_coords, gt_coords, metric='euclidean')
            min_dists.append(distances.min())

        return np.mean(min_dists)
    else:
        print('No non-zero elements in map_pred')
        return np.nan
    