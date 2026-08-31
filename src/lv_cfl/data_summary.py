import os
import numpy as np
from src.util.constants import *
from src.vis.deficit_vis import *
from src.vis.brain_vis import lesion_heatmap, lesion_size_dist

fig_path = os.path.join(FIG_PATH, 'cohort1_summary')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main():

    dataset = 'cohort1'

    # lesion heatmap
    X_tr = np.load(os.path.join(DATA_PATH, f'{dataset}/X.npy'))
    X_te = np.load(os.path.join(DATA_PATH, f'{dataset}/X_test.npy'))
    X = np.concatenate((X_tr, X_te), axis=0)
    fig,ax = plt.subplots(figsize=(PW//2,1))
    lesion_heatmap(X, mode='sum', cut_coords=(20,0,0),
                   save_path=os.path.join(fig_path, 'lesion_heatmap.png'),
                   threshold=1e-6, ax=ax)
    
    # unraveled heatmap
    fig,ax = plt.subplots(figsize=(PW,1))
    lesion_heatmap(X, mode='sum', display_mode='z',
                   save_path=os.path.join(fig_path, 'unraveled_lesion_heatmap.png'),
                   threshold=1e-6, ax=ax)

    # histograms of deficits
    Y = np.load(os.path.join(DATA_PATH, f'{dataset}/Yraw.npy'))
    deficit_names = ['Language', 'Visuospatial']
    deficit_histograms(Y, deficit_names, os.path.join(fig_path, 
                                                      'deficit_histograms.png'))
    deficit_names = ['L', 'V']
    deficit_corr(Y, deficit_names, tick_font=6,
                 save_path=os.path.join(fig_path, 'deficit_corr.png'))

    # histograms of demographics
    dems = np.load(os.path.join(DATA_PATH, f'{dataset}/dems_raw.npy'))
    dem_names = np.load(os.path.join(DATA_PATH, f'{dataset}/dem_names.npy'))
    deficit_histograms(dems, dem_names, os.path.join(fig_path, 
                       'dem_histograms.png'), dems=True)

    # distribution of lesion sizes
    lesion_size_dist(X, save_path=os.path.join(fig_path, 'lesion_sizes.png'))

if __name__ == '__main__':
    main()