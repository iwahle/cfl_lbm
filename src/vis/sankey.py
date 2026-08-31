import os
from src.util.constants import *
import plotly.graph_objects as go
from cfl.visualization import clustering_to_sankey as sk


def plot_sankey(lbls, save_path):

    link, label = sk.convert_lbls_to_sankey_nodes(lbls)
    # plot
    fig = go.Figure(data=
            [go.Sankey(node=dict(pad=15, thickness=20, label=label, 
                                 color="gray"), link=link)])
    fig.write_image(save_path, scale=1)

    link, label = sk.convert_lbls_to_sankey_nodes(lbls)
    # plot
    fig = go.Figure(data=
            [go.Sankey(node=dict(pad=15, thickness=20, label=None, 
                                 color="gray"), link=link)])
    save_path = save_path.replace('.png', '_no_labels.png')
    fig.write_image(save_path, scale=1)
