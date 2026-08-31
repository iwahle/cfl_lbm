import os
import pickle
import argparse
import numpy as np
from src.util.constants import *
import matplotlib.pyplot as plt
from matplotlib import cm as cmx
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR

fig_path = os.path.join(FIG_PATH, 'dep_q_vs_mean')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def compute_sc_aggs(Y):
    Y_sc = np.array([np.mean(Y[:,SOMCOG==i],axis=1) for i in range(max(SOMCOG)+1)]).T
    assert Y_sc.shape[1]==3
    return Y_sc

def compute_cfl_aggs(Y):
    qc = np.load('figures/dep_q_vs_mean/q_dendrogram_C.npy')
    
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
    Y_mbdi = np.mean(Y, axis=1)[:,np.newaxis]
    assert Y_mbdi.shape[1]==1
    return Y_mbdi

def plot_aggs(Y_mbdi, Y_sc, Y_cfl, labels):
        
    fig,axs = plt.subplots(2,3,figsize=(20,10))
    for i,c in enumerate([labels+1, Y_mbdi]):
        # make categorical colorblind friendly colormap
        cb_friendly_map = LinearSegmentedColormap.from_list('cb_friendly_map', 
                                        ['#E0A46C', '#B5CCFF', '#C49FE0', '#B12FE0'], N=4)
        cmap = cb_friendly_map if i==0 else 'viridis'
        alpha = 1 if i==0 else 1

        # 1D hist of total BDI by MS
        if i==0:
            elabel = [1,2,0,3]
            for ms in np.unique(labels):
                axs[i,0].hist(Y_mbdi[labels==ms], bins=np.linspace(0,3,65),
                    color=['#E0A46C', '#B5CCFF', '#C49FE0', '#B12FE0'][ms],
                    label=f'E{elabel[ms]+1}', alpha=alpha)
            axs[i,0].legend(loc='upper right')
            # change order of legend entries
            handles, labels = axs[i,0].get_legend_handles_labels()
            order = [2,0,1]
            axs[i,0].legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
        else:
            im = axs[i,0].hist(Y_mbdi, bins=np.linspace(0,3,65), 
                alpha=alpha, label='total BDI')
            scalar_map = cmx.ScalarMappable(norm=Normalize(vmin=0, vmax=3), cmap='viridis')
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
        jitter = np.random.uniform(-0.02,0.02,Y_sc.shape)
        axs[i,1].scatter(Y_sc[:,0]+jitter[:,0], Y_sc[:,1]+jitter[:,1], c=c, alpha=alpha, cmap=cmap)
        axs[i,1].set_xlim((-0.1,3))
        axs[i,1].set_ylim((-0.1,3))
        axs[i,1].set_ylabel('Somatic-Affective Agg')
        if i==1: axs[i,1].set_xlabel('Cognitive Agg')

        # 2D scatter of cfl aggs
        jitter = np.random.uniform(-0.02,0.02,Y_cfl.shape)
        axs[i,2].scatter(Y_cfl[:,0]+jitter[:,0], Y_cfl[:,1]+jitter[:,1], c=c, alpha=alpha, cmap=cmap)
        axs[i,2].set_xlim((-0.1,3))
        axs[i,2].set_ylim((-0.1,3))
        axs[i,2].set_ylabel('CFL Agg 2')
        if i==1: axs[i,2].set_xlabel('CFL Agg 1')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'agg_plots'), dpi=300)
    return fig


def evalute_predictive_ability(Y_mbdi, Y_sc, Y_cfl, labels):
    ''' predict CFL effect macrostate from each agg config '''

    aggs = [Y_mbdi, Y_sc, Y_cfl]

    scores = np.zeros((len(aggs), 10))
    for i,agg in enumerate(aggs):
        model = LR(multi_class='multinomial')
        scores[i] = cross_val_score(model, agg, labels, cv=10, scoring='accuracy')
    print('multiclass logisitic regression')
    print(scores)
    print(np.mean(scores,axis=1))

    # plot scores
    fig,ax = plt.subplots()
    ax.bar(range(len(aggs)), np.mean(scores,axis=1), yerr=np.std(scores,axis=1)) 
    ax.set_xticks(range(len(aggs))) 
    ax.set_xticklabels(['mean BDI', 'som-cog', 'cfl aggregates'], rotation=45, ha='right')    
    ax.set_ylabel('classification accuracy')  
    ax.set_ylim((0,1))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'pred_eval.png'), dpi=300)

def main():

    # load data
    X_test = np.load('data/cohort2/X_test.npy')
    Yraw = np.load(os.path.join(DATA_PATH, 'cohort2/Yraw.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, 'cohort2/test_idx.npy'))
    Ytest = Yraw[test_idx]

    # load effect macrostates
    with open('results/dep_cfl/cfl_results/experiment0000/test/EffectClusterer_results.pickle', 'rb') as f:
        labels = pickle.load(f)['y_lbls']

    Y_mbdi = compute_mbdi_aggs(Ytest)
    Y_sc = compute_sc_aggs(Ytest)
    Y_cfl = compute_cfl_aggs(Ytest)

    plot_aggs(Y_mbdi, Y_sc, Y_cfl, labels)
    evalute_predictive_ability(Y_mbdi, Y_sc, Y_cfl, labels)
    
if __name__ == '__main__':
    main()