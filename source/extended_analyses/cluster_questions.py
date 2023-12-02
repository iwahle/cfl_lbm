import os
import argparse
import numpy as np
from source.util import *
from source.paths import *
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from cfl.post_cfl.microvariable_importance import compute_microvariable_importance

def main(exp_id, n_clusters=5):
    '''
    Cluster questions based on their microvariable importance profiles.
    Arguments:
        - exp_id (str): experiment ID to draw CFL partition results from
        - n_clusters (int): number of clusters to cut dendrogram off at.
    Returns: None
    '''

    analysis = 'bdi'
    exp_path = os.path.join(RESULTS_PATH, analysis, f'experiment{exp_id.zfill(4)}')
    _,Y,_ = load_scale_data(analysis=analysis)
    mi = compute_microvariable_importance(  exp_path, 
                                            Y,
                                            dataset_name='dataset_train',
                                            visualize=False,
                                            cause_or_effect='effect')
    print('microvariable importances shape: ', mi.shape)

    # flatten + transpose importances into importance profiles for each question
    profiles = np.array([mi[:,:,i][np.tril_indices(mi.shape[0],-1)] for i in range(mi.shape[-1])])
    print('profiles shape: ', profiles.shape)

    # prepend asterisk to somatic questions for dendrogram labels
    labels = np.load(os.path.join(DATA_PATH, analysis, 'question_names.npy'))
    for i in range(21):
        if SOMCOG[i]:
            labels[i] = '*' + labels[i]

    # plot dendrogram
    fig,ax = plt.subplots()
    dist = pdist(profiles)
    Z = hierarchy.linkage(dist, 'centroid')
    dn = hierarchy.dendrogram(Z, orientation='left', labels=labels, 
                              color_threshold=0, ax=ax)
    C = hierarchy.fcluster(Z, n_clusters, 'maxclust')
    np.save(os.path.join(exp_path, 'q_dendrogram_C.npy'), C)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, 'q_dendrogram'), dpi=300)

if __name__=='__main__':
    '''
    Example usage:
        python cluster_questions.py --exp_id 0 --n_clusters 3
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int)
    parser.add_argument('--n_clusters', type=int)

    main(**vars(parser.parse_args()))