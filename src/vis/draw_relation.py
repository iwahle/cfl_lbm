import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.util.constants import *
from src.vis.sankey import plot_sankey

def _compute_transition_matrix(xlbls, ylbls):

    xu = np.unique(xlbls)
    yu = np.unique(ylbls)
    P_CE = np.zeros((len(xu), len(yu)))
    for i, x in enumerate(xu):
        for j, y in enumerate(yu):
            P_CE[i,j] = np.sum(np.logical_and(xlbls==x, ylbls==y))
    P_CE = P_CE / np.sum(P_CE)
    P_C = np.sum(P_CE, axis=1)
    P_E_given_C = P_CE / P_C[:,np.newaxis]
    return P_E_given_C

def draw_relation(cms, ems, plot_order_c, plot_order_e, save_path, 
                  sigs_lt=None, sigs_gt=None, sig_thresh=0.05):
    n_cms,n_ems = len(np.unique(cms)),len(np.unique(ems))
    assert n_cms < 10
    assert n_ems < 10

    tm = _compute_transition_matrix(cms, ems)
    print('P(Effect | Cause):')
    print(tm)
    print(tm.shape)
    
    # reverse order because networkx plots upside down (:
    tm = tm[plot_order_c[::-1]][:,plot_order_e[::-1]]
    if sigs_lt is not None:
        sigs_lt = sigs_lt[plot_order_c[::-1]][:,plot_order_e[::-1]]
    if sigs_gt is not None:
        sigs_gt = sigs_gt[plot_order_c[::-1]][:,plot_order_e[::-1]]

    G = nx.DiGraph()
    elist = []
    edge_colors = []
    edge_alphas = []
    edge_labels = {}
    for i in range(n_cms):
        for j in range(n_ems):
            elist.append((i, j+10))
            edge_colors.append(cm.get_cmap('Oranges')(tm[i,j]))
            edge_alphas.append(1)
            # add significance labels
            if (sigs_lt is not None) and (sigs_gt is not None):
                if sigs_lt[i,j] < sig_thresh:
                    edge_labels[(i, j+10)] = f'{tm[i,j]:.2f}' + r'$^*$'
                elif sigs_gt[i,j] < sig_thresh:
                    edge_labels[(i, j+10)] = f'{tm[i,j]:.2f}' + r'$^\dagger$'

            else:
                edge_labels[(i, j+10)] = ''

    print('edge_labels:', edge_labels)            
    G.add_edges_from(elist)

    fig, ax = plt.subplots(1, 1, figsize=(CW1*.6, n_cms*0.8))
    ax.axis('off')
    pos = nx.bipartite_layout(G, nodes=range(n_cms))
    edges = nx.draw_networkx(G, pos=pos, with_labels=False,
                            edge_color=edge_colors, node_size=0,
                            width=2, arrows=True, arrowsize=8, ax=ax,
                            alpha=edge_alphas)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels, font_size=FS-2,
        font_color='black', label_pos=0.7, rotate=True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('Oranges'), 
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.03, shrink=0.6)
    cbar.set_label('Edge weight', fontsize=FS-2)
    cbar.ax.tick_params(labelsize=FS-2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

def draw_relation_sankey(xlbls, ylbls, save_path):
    plot_sankey([xlbls, ylbls], save_path=save_path)
