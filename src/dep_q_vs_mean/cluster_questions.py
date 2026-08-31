

import os
import sys
import pickle
import argparse
import numpy as np
from src.util.constants import *
from scipy.cluster import hierarchy
from src.util.data_util import load_data
from scipy.spatial.distance import pdist
from cfl.post_cfl.microvariable_importance import discriminate_clusters

fig_path = os.path.join(FIG_PATH, 'dep_q_vs_mean')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main(exp_id=0, partition='', n_q_clusters=3):
    assert partition in ['cause', 'effect']
    plbl = 'Cause' if partition=='cause' else 'Effect'
    dlbl = 'x' if partition=='cause' else 'y'

    # load data
    _,_,_,Ytest,labels,_ = load_data('cohort2', dems=False)

    # load effect macrostates from 21q cfl
    exp_id_str = str(exp_id).zfill(4)
    with open(f'results/dep_cfl/cfl_results/experiment{exp_id_str}/test/{plbl}Clusterer_results.pickle', 'rb') as f:
        lbls = pickle.load(f)[f'{dlbl}_lbls']

    # compute KL distances between clusters per question
    print('computing microvariable importances: ')
    mi = discriminate_clusters(Ytest, lbls)

    fig,ax = plt.subplots(6,4,figsize=(12,12))
    for i,axi in enumerate(ax.flatten()):
        if i >= 21:
            axi.axis('off')
        else:
            axi.imshow(mi[:,:,i], cmap='viridis', vmin=np.min(mi), vmax=np.max(mi))
            axi.set_title(labels[i])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'microvariable_importances'), dpi=300)

    print('microvariable importances shape: ', mi.shape)

    # flatten and transpose importances into importance profiles for each question
    profiles = np.array([mi[:,:,i][np.tril_indices(mi.shape[0],-1)] for i in range(mi.shape[-1])])
    print('profiles shape: ', profiles.shape)

    # plot dendrogram
    # FIG 7A
    dist = pdist(profiles)
    Z = hierarchy.linkage(dist, 'centroid')
    fig = plt.figure(figsize=(2,3))
    dn = hierarchy.dendrogram(Z, orientation='left', labels=labels, 
                              color_threshold=0, leaf_font_size=6)
    C = hierarchy.fcluster(Z, n_q_clusters, 'maxclust')
    print('n question clusters', len(np.unique(C)))
    np.save(os.path.join(fig_path, f'q_dendrogram_C_{n_q_clusters}.npy'), C)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'q_dendrogram'), dpi=300)
    # plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--partition', type=str)
    parser.add_argument('--n_q_clusters', type=int)
    main(**vars(parser.parse_args()))