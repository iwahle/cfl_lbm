

import os
import pickle
import numpy as np
from src.util.constants import *
from scipy.cluster import hierarchy
from src.util.data_util import load_data
from scipy.spatial.distance import pdist
from cfl.post_cfl.microvariable_importance import discriminate_clusters

fig_path = os.path.join(FIG_PATH, 'dep_q_vs_mean')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def main():

    # load data
    Yraw = np.load(os.path.join(DATA_PATH, 'cohort2/Yraw.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, 'cohort2/test_idx.npy'))
    Ytest = Yraw[test_idx]

    # load effect macrostates
    with open(os.path.join(RESULTS_PATH, 
              'dep_cfl/cfl_results/experiment0000/test/EffectClusterer_results.pickle'), 'rb') as f:
        ylbls = pickle.load(f)['y_lbls']

    # load question names
    _,_,_,_,labels, _ = load_data('cohort2')

    # prepend asterisk to somatic questions
    for i in range(21):
        if SOMCOG[i]:
            labels[i] = '*' + labels[i]

    # compute KL distances between clusters per question
    print('computing microvariable importances: ')
    mi = discriminate_clusters(Ytest, ylbls)

    print('microvariable importances shape: ', mi.shape)

    # flatten and transpose importances into importance profiles for each question
    profiles = np.array([mi[:,:,i][np.tril_indices(mi.shape[0],-1)] for i in range(mi.shape[-1])])
    print('profiles shape: ', profiles.shape)

    # plot dendrogram
    dist = pdist(profiles)
    Z = hierarchy.linkage(dist, 'centroid')
    dn = hierarchy.dendrogram(Z, orientation='left', labels=labels, color_threshold=0)
    C = hierarchy.fcluster(Z, 3, 'maxclust')
    print('n question clusters', np.unique(C))
    np.save(os.path.join(fig_path, 'q_dendrogram_C.npy'), C)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'q_dendrogram'), dpi=300)

if __name__=='__main__':
    main()