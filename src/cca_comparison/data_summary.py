import os
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *
from src.vis.brain_vis import lesion_heatmap, lesion_size_dist
from src.util.data_util import load_data


fig_path = os.path.join(FIG_PATH, 'sim_summary')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(parcel_series, atlas_fn, roi_idxs):


    # lesion heatmap
    X, X_test, Y, Y_test, _,_ = load_data(f'simulated_{parcel_series}')
    X = np.concatenate((X, X_test), axis=0)
    fig,ax = plt.subplots(figsize=(PW-2.5,1))
    lesion_heatmap(X, mode='sum', cut_coords=(-20,0,20),
                   save_path=os.path.join(fig_path, 'lesion_heatmap.png'),
                   threshold=1e-6, ax=ax)

    # unraveled lesion heatmap
    fig,ax = plt.subplots(figsize=(PW,1))
    lesion_heatmap(X, mode='sum', display_mode='z',
                   save_path=os.path.join(fig_path, 'unraveled_lesion_heatmap.png'),
                   threshold=1e-6, ax=ax)
    
    # distribution of lesion sizes
    lesion_size_dist(X, save_path=os.path.join(fig_path, 'lesion_sizes.png'))

    # histograms of deficits
    Y = np.concatenate((Y, Y_test), axis=0)
    Y = Y[:,roi_idxs]
    print(Y.shape)
    deficit_names = ['Defiict A', 'Deficit B', 'Deficit C']
    deficit_histograms_v(Y, deficit_names, 
                         os.path.join(fig_path, 'deficit_histograms.png'))
    deficit_names = ['A', 'B', 'C']
    deficit_corr(Y, deficit_names, vmin=0, 
                 save_path=os.path.join(fig_path, 'deficit_corr.png'))

    
    # ground truth regions
    parcel_path = os.path.join(DATA_PATH, f'simulated_{parcel_series}/{atlas_fn}')
    parcels = load_brain(parcel_path, to_flatten=True, mask=TEMPLATE)
    print(parcels.shape)

    rois = []
    for roi_idx in roi_idxs:
        rois.append(parcels == roi_idx)
    rois = np.array(rois)
    print(rois.shape)
    print(np.sum(rois, axis=1))

    fig,ax = plt.subplots(len(deficit_names),1,figsize=(PW//2,len(deficit_names)))
    for di,deficit in enumerate(deficit_names):
        lesion_heatmap(rois[di,:][None,:], mode='sum', ax=ax[di], cut_coords=None,
                        colorbar=False, cmap='Blues', threshold=1e-6)
        ax[di].set_ylabel(deficit)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'deficit_regions.png'), dpi=300)

if __name__ == '__main__':
    main(parcel_series='schaefer200', 
         atlas_fn=ATLAS_FN,
         roi_idxs=EX_ROIS)