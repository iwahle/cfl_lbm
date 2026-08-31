
import numpy as np
import os
import argparse
from src.util.constants import *
from src.util.data_util import *

def main(series=''):
    locs = range(1,101)
    eval_names = ['dice score', 'peak disp.', 'centroid disp.', 'contour disp.']
    evals_all = np.zeros((len(locs), len(eval_names)))
    fp = os.path.join(RESULTS_PATH, f'simulated_schaefer200/lesymap_results{series}')

    # load ground truth
    map_gt_all = load_brain(os.path.join(DATA_PATH, 
        f'simulated_schaefer200/{ATLAS_FN}'),
        mask=None, to_flatten=False)

    map_gt_all[map_gt_all.shape[0]//2:] = 0 # zero out right hemisphere
    map_gt_all = map_gt_all.reshape(np.product(map_gt_all.shape)) # flatten     
    flat_t = np.reshape(TEMPLATE, -1)

    no_sig_cnt = 0
    for Yidx in range(1,101):
        print(f'Yidx: {Yidx}')
        try:
            mask_pred = load_brain(os.path.join(fp, f'run_{Yidx}', 'stat_img.nii.gz'), 
                                   to_flatten=True, mask=TEMPLATE)
            map_pred = load_brain(os.path.join(fp, f'run_{Yidx}', 'rawWeights_img.nii.gz'), 
                                to_flatten=True, mask=TEMPLATE)
            map_pred[mask_pred==0] = 0
        except FileNotFoundError:
            print(f'run_{Yidx} not found')
            evals_all[Yidx-1] = np.nan
            continue
        
        if np.sum(map_pred!=0) == 0:
            print(f'run_{Yidx} has no non-zero elements')
            evals_all[Yidx-1] = [np.nan, np.nan, np.nan, np.nan]
            no_sig_cnt += 1
            continue
        
        map_gt = map_gt_all==Yidx
        map_gt = map_gt[flat_t.astype(bool)]

        # reshape to 3D
        map_pred_3d = np.zeros_like(flat_t)
        map_pred_3d[flat_t.astype(bool)] = map_pred
        map_pred_3d = map_pred_3d.reshape(TEMPLATE.shape)

        map_gt_3d = np.zeros_like(flat_t)
        map_gt_3d[flat_t.astype(bool)] = map_gt
        map_gt_3d = map_gt_3d.reshape(TEMPLATE.shape)

        print('Dice')
        map_pred_bool = map_pred > 0
        dice = dice_eval(map_pred_bool, map_gt)
        print(dice)

        print('Peak Disp')
        peak_disp = peak_disp_eval(map_pred_3d, map_gt_3d)
        print(peak_disp)

        print('Centroid Disp')
        cent_disp = centroid_disp_eval(map_pred_3d, map_gt_3d)
        print(cent_disp)

        print('Contour Disp')
        map_pred_3d_bool = map_pred_3d > 0
        contour_disp = contour_disp_eval(map_pred_3d_bool, map_gt_3d)
        print(contour_disp)

        evals_all[Yidx-1] = [dice, peak_disp, cent_disp, contour_disp]

    print(np.sum(np.isnan(evals_all), axis=0))
    np.save(os.path.join(fp, f'lesymap_evals{series}.npy'), evals_all)
    print('saved to:', os.path.join(fp, f'lesymap_evals{series}.npy'))
    print('no sig cnt:', no_sig_cnt)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', type=str, default='')
    args = parser.parse_args()
    main(args.series)