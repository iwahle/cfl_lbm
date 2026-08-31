import numpy as np
import matplotlib.pyplot as plt
from src.vis.draw_relation import _compute_transition_matrix

def _compute_sig(scores, mask, alternative, n_resample):
    ''' Given a list of scores on a single test and a mask of scores within a
        cluster, resample the same number of scores as in the cluster n_iter
        times and compute the p-value of the cluster score.
        alternative:'greater' or 'less'. if greater, what is the probability
        that a random sample is greater than the mask sample. if less, what is
        the probability that random sample is less than the mask sample.
    '''
    
    n_mask = np.sum(mask)
    sample_means = np.zeros(n_resample)
    for i in range(n_resample):
        sample_means[i] = np.mean(np.random.choice(scores, n_mask, replace=False))

    cluster_mean = np.mean(scores[mask])
    if alternative == 'greater':
        pval = np.mean(sample_means > cluster_mean)
    elif alternative == 'less':
        pval = np.mean(sample_means < cluster_mean)
    else:
        raise ValueError('alternative must be greater or less')
    return pval, sample_means

    
def sig_test(Y, C, n_resample):
    n_clusters = len(np.unique(C))
    pvals_less = np.zeros((n_clusters, Y.shape[1]))
    pvals_greater = np.zeros((n_clusters, Y.shape[1]))
    smls = np.zeros((n_clusters, Y.shape[1], n_resample))
    smgs = np.zeros((n_clusters, Y.shape[1], n_resample))
    for i in range(Y.shape[1]):
        scores = Y[:, i]
        for j in range(n_clusters):
            mask = C == j
            pvals_less[j,i],smls[j,i] = _compute_sig(scores, mask, 'less', n_resample)
            pvals_greater[j,i],smgs[j,i] = _compute_sig(scores, mask, 'greater', n_resample)

    return pvals_less, pvals_greater, np.mean(smls,axis=2), np.mean(smgs,axis=2)


def sig_test_relation(xlbls, ylbls, n_resample):
    true_rel = _compute_transition_matrix(xlbls, ylbls)
    nxlbls = len(np.unique(xlbls))
    nylbls = len(np.unique(ylbls))
    pvals_less = np.zeros((nxlbls, nylbls))
    pvals_greater = np.zeros((nxlbls, nylbls))
    perm_rels = np.zeros((n_resample, nxlbls, nylbls))
    for i in range(n_resample):
        xlbls_perm = np.random.permutation(xlbls)
        ylbls_perm = np.random.permutation(ylbls)
        perm_rels[i] = _compute_transition_matrix(xlbls_perm, ylbls_perm)
    pvals_greater = np.mean(perm_rels > true_rel, axis=0)
    pvals_less = np.mean(perm_rels < true_rel, axis=0)

    return pvals_less, pvals_greater


def sig_test_voxels(X, xlbls, xi, n_resample):
    ''' 
    For each voxel, test whether it is lesioned more often in the observed
    cluster than in a randomly sampled cluster of same size.
    Arguments:
        X: n_subjects x n_voxels flattened lesion masks
        xlbls: observed cluster assignments
        xi: cluster to test
    Returns:
        gt_cnt: voxel-wise percentage of resamples where observed cluster
        is lesioned more than resampled cluster
    '''

    Xobs = np.mean(X[xlbls==xi],axis=0)
    k = np.sum(xlbls==xi)

    resamples = np.zeros((n_resample, Xobs.shape[0]))
    for i in range(n_resample):
        sample_idx = np.random.choice(X.shape[0], k, replace=False)
        resamples[i] = np.mean(X[sample_idx],axis=0)
    gt_cnt = np.mean(Xobs > resamples, axis=0)
    return gt_cnt
    


