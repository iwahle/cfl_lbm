import os
import pickle
import argparse
import numpy as np
from source.util import *
from source.paths import *
import matplotlib.pyplot as plt
from matplotlib import cm as cmx
from matplotlib.colors import Normalize, LinearSegmentedColormap
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR


def compute_sc_aggs(Y):
    '''
    Compute somatic-affective/cognitive aggregates.
    Arguments:
        - Y (np.ndarray): 21-dimensional question-wise BDI responses.
    Returns:
        - Y_sc (np.ndarray): somatic-affective/cognitive aggregates.
    '''

    Y_sc = np.array([np.mean(Y[:,SOMCOG==i],axis=1) for i in range(max(SOMCOG)+1)]).T
    assert Y_sc.shape[1]==3
    return Y_sc

def compute_cfl_aggs(Y, exp_id, analysis):
    '''
    Compute CFL aggregates based on question clustering from exp_id.
    Arguments:
        - Y (np.ndarray): 21-dimensional question-wise BDI responses.
        - exp_id (str): experiment ID to draw CFL partition results from
    Returns:
        - Y_cfl (np.ndarray): CFL aggregates.
    '''
    
    qc = np.load(os.path.join(RESULTS_PATH, analysis, 
        f'experiment{exp_id.zfill(4)}/q_dendrogram_C.npy'))  
    assert qc.shape==(21,)
    qc_lbls = np.unique(qc)
    n_q_clusters = len(qc_lbls)
    cluster_sizes = np.array([np.sum(qc==i) for i in qc_lbls])
    sorted_size_idx = np.flip(np.argsort(cluster_sizes))
    
    Y_cfl = np.zeros((Y.shape[0],n_q_clusters))
    for i in range(n_q_clusters):
        print('cluster {} size: {}'.format(sorted_size_idx[i], 
                                           np.sum(qc==qc_lbls[sorted_size_idx[i]])))
        Y_cfl[:,i] = np.mean(Y[:,qc==qc_lbls[sorted_size_idx[i]]], axis=1)
    assert Y_cfl.shape[1]==n_q_clusters
    return Y_cfl

def compute_mbdi_aggs(Y):
    '''
    Compute mean BDI aggregate.
    Arguments:
        - Y (np.ndarray): 21-dimensional question-wise BDI responses.
    Returns:
        - Y_mbdi (np.ndarray): mean BDI aggregate.
    '''
    Y_mbdi = np.mean(Y, axis=1)[:,np.newaxis]
    assert Y_mbdi.shape[1]==1
    return Y_mbdi

def plot_aggs(Y_mbdi, Y_sc, Y_cfl, labels, exp_id, analysis='bdi'):
    '''
    Plot aggregate distributions.
    Arguments:
        - Y_mbdi (np.ndarray): mean BDI aggregate.
        - Y_sc (np.ndarray): somatic/cognitive aggregate.
        - Y_cfl (np.ndarray): CFL aggregate.
        - labels (np.ndarray): effect categories.
        - exp_id (str): experiment ID to draw CFL partition results from
    Returns: None
    '''
        
    fig,axs = plt.subplots(2,3,figsize=(20,10))
    for i,c in enumerate([labels+1, Y_mbdi]):
        colorblind_cmap = LinearSegmentedColormap.from_list('colorblind_cmap', 
            ['#E0A46C', '#B5CCFF', '#C49FE0'], N=3)
        cmap = colorblind_cmap if i==0 else 'viridis'
        alpha = 1 if i==0 else 1

        # 1D hist of total BDI by MS
        if i==0:
            elabel = [1,2,0]
            for ms in np.unique(labels):
                axs[i,0].hist(Y_mbdi[labels==ms], bins=np.linspace(0,3,65),
                    color=['#E0A46C', '#B5CCFF', '#C49FE0'][ms],
                    label=f'E{elabel[ms]+1}', alpha=alpha)
            axs[i,0].legend(loc='upper right')
            # change order of legend entries
            handles, labels = axs[i,0].get_legend_handles_labels()
            order = [2,0,1]
            axs[i,0].legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
        else:
            im = axs[i,0].hist(Y_mbdi, bins=np.linspace(0,3,65), 
                alpha=alpha, label='mean BDI')
            scalar_map = cmx.ScalarMappable(norm=Normalize(vmin=0, vmax=3), 
                                            cmap='viridis')
            for bar in axs[i,0].containers[0]:
                x = bar.get_x()
                bar.set_color(scalar_map.to_rgba(x))
            cbaxes = fig.add_axes([0.04, 0.15, 0.01, 0.2]) 
            cbar = fig.colorbar(scalar_map, cax=cbaxes, ax=axs[i,0])
            cbar.set_label('mean BDI', labelpad=-60)
        axs[i,0].set_ylabel('# participants')

        axs[i,0].set_xlim((-0.1,3))
        if i==1: axs[i,0].set_xlabel('Mean BDI')

        # 2D scatter of som cog
        axs[i,1].scatter(Y_sc[:,0], Y_sc[:,1], c=c, alpha=alpha, cmap=cmap)
        axs[i,1].set_xlim((-0.1,3))
        axs[i,1].set_ylim((-0.1,3))
        axs[i,1].set_ylabel('Somatic-affective Agg')
        if i==1: axs[i,1].set_xlabel('Cognitive Agg')

        # 3D scatter of cfl aggs
        axs[i,2].remove()
        ax = fig.add_subplot(2, 3, 3*i+3, projection='3d')
        ax.scatter(Y_cfl[:,0], Y_cfl[:,1], Y_cfl[:,2], c=c, alpha=alpha, 
                   cmap=cmap)
        ax.set_xlabel('\nCFL Agg 1')
        ax.set_ylabel('\nCFL Agg 2')
        ax.set_zlabel('\nCFL Agg 3')
        ax.set_xlim((0,3))
        ax.set_ylim((0,3))
        ax.set_zlim((0,3))
        ax.view_init(15, -65)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, analysis, 
                             f'experiment{exp_id.zfill(4)}',
                             'agg_dists'), dpi=300)
    return fig


def evalute_predictive_ability(Y_mbdi, Y_sc, Y_cfl, labels, exp_id, analysis):
    '''
    Evaluate predictions of CFL effect categories from each aggregate 
    configuration.
    Arguments:
        - Y_mbdi (np.ndarray): mean BDI aggregate.
        - Y_sc (np.ndarray): somatic/cognitive aggregate.
        - Y_cfl (np.ndarray): CFL aggregate.
        - labels (np.ndarray): effect categories.
        - exp_id (str): experiment ID to draw CFL partition results from
    Returns: None
    '''

    aggs = [Y_mbdi, Y_sc, Y_cfl]

    scores = np.zeros((len(aggs), 10))
    for i,agg in enumerate(aggs):
        model = LR(multi_class='multinomial')
        scores[i] = cross_val_score(model, agg, labels, cv=10, 
                                    scoring='accuracy')
    print('multiclass logisitic regression')
    print('mean scores:', np.mean(scores,axis=1))

    # plot scores
    fig,ax = plt.subplots()
    ax.bar(range(len(aggs)), np.mean(scores,axis=1), yerr=np.std(scores,axis=1)) 
    ax.set_xticks(range(len(aggs))) 
    ax.set_xticklabels(['mean BDI', 'som-cog', 'cfl aggregates'], rotation=45, 
                       ha='right')    
    ax.set_ylabel('classification accuracy')  
    ax.set_ylim((0,1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, analysis, 
                             f'experiment{exp_id.zfill(4)}',
                             'agg_pred_performance'), dpi=300)

def main(exp_id):
    '''
    Compare BDI question aggregate configurations for a given experiment.
    Arguments:
        - exp_id (str): experiment ID to draw CFL partition results from
    Returns: None
    '''
    # load data
    analysis = 'bdi'
    _,Y,_ = load_scale_data(analysis=analysis)
    Y *= 3 # unscale

    # load effect macrostates
    with open(os.path.join(RESULTS_PATH, analysis, f'experiment{exp_id.zfill(4)}', 
        'dataset_train/EffectClusterer_results.pickle'), 'rb') as f:
        labels = pickle.load(f)['y_lbls']
        
    Y_mbdi = compute_mbdi_aggs(Y)
    Y_sc = compute_sc_aggs(Y)
    Y_cfl = compute_cfl_aggs(Y, exp_id=exp_id, analysis=analysis)

    plot_aggs(Y_mbdi, Y_sc, Y_cfl, labels, exp_id=exp_id, analysis=analysis)
    evalute_predictive_ability(Y_mbdi, Y_sc, Y_cfl, labels, exp_id=exp_id, 
                               analysis=analysis)
    
if __name__ == '__main__':
    '''
    Example usage:
        python compare_aggregates.py --exp_id 0
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str)
    main(**vars(parser.parse_args()))