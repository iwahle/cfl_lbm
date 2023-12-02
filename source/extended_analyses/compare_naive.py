import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from source.util import *
from source.paths import *

def main():
    '''
    Compare naive clustering of lesion masks to partition found by CFL.
    Arguments: None
    Returns: None
    '''

    save_path = os.path.join(RESULTS_PATH, 'bdi', 'naive_comparison')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    X,_,_ = load_scale_data(analysis='bdi')

    # cluster X
    Ks = np.arange(2,11)
    cluster_scores = np.zeros(len(Ks))
    for K in Ks:
        C = KMeans(n_clusters=K, random_state=0).fit_predict(X)
        cluster_scores[K-Ks[0]] = davies_bouldin_score(X, C)

    fig,ax = plt.subplots()
    ax.plot(Ks, cluster_scores)
    ax.set_xticks(Ks)
    ax.set_xlabel('K')
    ax.set_ylabel('davies_bouldin_score')
    plt.savefig(os.path.join(save_path, 'tmp'), bbox_inches='tight')

    # compute final clustering based on user input
    final_k = int(input('Final K value: '))
    C = KMeans(n_clusters=final_k, random_state=0).fit_predict(X)
    np.save(os.path.join(save_path, 'naive_C.npy'), C)

if __name__=='__main__':
    '''
    Example usage:
        python compare_naive.py
    '''
    main()