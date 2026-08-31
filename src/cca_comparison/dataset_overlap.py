import numpy as np
from corner import corner
import matplotlib.pyplot as plt
from src.util.constants import *
from src.util.data_util import load_data

fig_path = os.path.join(FIG_PATH, 'cca_comparison')

def main():

    _, _, Y, Y_test, deficit_names, _ = load_data('simulated')

    # corner plot of deficit histograms, no contours just data points
    corner(Y_test, labels=deficit_names, plot_contours=False, plot_datapoints=True,
           fill_contours=False, bins=20, smooth=1, color='b', plot_density=False,
           data_kwargs={'alpha':1})
           
    plt.savefig(os.path.join(fig_path, 'corner_plot.png'), dpi=300)

    c = np.corrcoef(Y_test.T)
    fig,ax = plt.subplots(figsize=(2,2))
    im = ax.imshow(c, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(deficit_names)))
    ax.set_yticks(range(len(deficit_names)))
    ax.set_xticklabels(deficit_names, rotation=90)
    ax.set_yticklabels(deficit_names)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'correlation_matrix.png'), dpi=300)

    print(c)
if __name__ == '__main__':
    main()