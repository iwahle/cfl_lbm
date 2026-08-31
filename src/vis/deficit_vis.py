import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src.util.constants import *


def deficit_histograms(deficits, titles, save_path=None, dems=False, axs=None):
    """
    Create histograms of the deficits (or demographics).
    Arguments:
        deficits: np.array of shape (n_subjects, n_deficits)
    """
    if axs is None:
        fig, ax = plt.subplots(1,deficits.shape[1],figsize=(1*len(titles),1.5), sharey=True)
    else:
        ax = axs
    for i in range(deficits.shape[1]):
        bins = np.linspace(np.min(deficits[:,i]), np.max(deficits[:,i])+1, 20)
        if titles[i]=='Sex':
            bins = [-0.2,0.2,0.8,1.2]
        ax[i].hist(deficits[:,i], bins=bins, color=GREEN)
        if dems:
            if titles[i]=='Sex':
                ax[i].set_xticks([0,1])
                ax[i].set_xticklabels(['M','F'])
            ax[i].set_xlabel(titles[i])
        else:
            ax[i].set_xlabel('Score')
        
    ax[0].set_ylabel('Count')
    if save_path is not None:
        fig.align_ylabels(ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, transparent=True)
    
def deficit_histograms_v(deficits, titles, save_path, dems=False):
    """
    Create histograms of the deficits (or demographics).
    Arguments:
        deficits: np.array of shape (n_subjects, n_deficits)
    """
    fig, ax = plt.subplots(deficits.shape[1],1,figsize=(1.6,1.2*len(titles)))
    for i in range(deficits.shape[1]):
        bins = np.linspace(np.min(deficits[:,i]), np.max(deficits[:,i])+.1, 20)
        if titles[i]=='Sex':
            bins = [-0.2,0.2,0.8,1.2]
        ax[i].hist(deficits[:,i], bins=bins, color=GREEN)
        ax[i].set_title(titles[i])
        ax[i].set_ylabel('Count')
        if dems:
            if titles[i]=='Sex':
                ax[i].set_xticks([0,1])
                ax[i].set_xticklabels(['M','F'])
        else:
            ax[-1].set_xlabel('Score')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def deficit_corr(deficits, titles, tick_font=FS, figsize=(2,2), 
                 cbar=True, vmin=-1, save_path=None):
    """
    Create a heatmap of the correlation matrix of the deficits.
    Arguments:
        deficits: np.array of shape (n_subjects, n_deficits)
    """

    corr = np.corrcoef(deficits, rowvar=False)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = _truncate_colormap(plt.cm.coolwarm, minval=vmin/2+0.5, maxval=1)
    cax = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=1)
    if cbar:
        cbar = fig.colorbar(cax, shrink=0.6)
        # set cbar font
        cbar.ax.tick_params(labelsize=tick_font)
        cbar.set_label('Correlation', fontsize=FS)
    ax.set_xticks(range(len(titles)))
    ax.set_yticks(range(len(titles)))
    if len(titles[0])>5:
        titles = range(1,len(titles)+1)

    ax.set_xticklabels(titles, fontsize=tick_font)
    ax.set_yticklabels(titles, fontsize=tick_font)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')

def bdi_dists(deficits, ylbls, titles, save_path=None, plot_order=None, 
              labeled=True, figsize=None,axs=None):
    """
    Plot the response distributions for each cluster.
    """
    n_questions = deficits.shape[1]
    n_clusters = len(np.unique(ylbls))
    if figsize is None:
        figsize = (PW//2.5,n_clusters)
    if axs is None:
        fig,axs = plt.subplots(n_clusters, 1, figsize=figsize, 
                            sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.2)
    if n_clusters==1:
        axs = [axs]
    bins = np.arange(5)

    if plot_order is None:
        plot_order = range(n_clusters)
    else:
        assert len(plot_order)==n_clusters

    # precompute histograms
    hists = np.zeros((n_questions,n_clusters,4))
    for i in range(n_questions):
        for j in range(n_clusters):
            hists[i,j,:] = np.histogram(deficits[ylbls==j,i],bins=bins)[0]
            # this hist now contains for each question and each cluster, the
            # count of each type of response (0,1,2,3)
    # normalize hist by number of subjects in each cluster
    for j in range(n_clusters):
        hists[:,j,:] = hists[:,j,:]/np.sum(ylbls==j)
    
    for j,ax in zip(range(n_clusters), axs):
        data = np.hstack([np.expand_dims(np.arange(n_questions),-1), 
                          hists[:,plot_order[j],:]])
        # hists[:,j,:] will be an n_questions x 4 array
        # we need to append row labels (number each question)
        assert data.shape==(n_questions,5)
        # ax = ax[0]

        df = pd.DataFrame(data, columns=['', 'BDI Resp. 0', 'BDI Resp. 1', 
                                         'BDI Resp. 2', 'BDI Resp. 3'])
        df.plot(x='', y=['BDI Resp. 0', 'BDI Resp. 1', 'BDI Resp. 2', 
                         'BDI Resp. 3'], kind='bar', stacked=True, 
                ax=ax, colormap=GREEN_DISC, legend=True)

        if j==n_clusters-1:
            ax.set_xticks(np.arange(n_questions))
            ax.set_xticklabels(titles, ha='right', rotation=50, fontsize=FS-1)
        else:
            ax.set_xticks([]) 
        ax.set_yticks([])
        if (n_clusters!=1) & labeled:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(f'N={np.sum(ylbls==plot_order[j])}', rotation=270, labelpad=10, ha='center')
        ax.get_legend().remove()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
