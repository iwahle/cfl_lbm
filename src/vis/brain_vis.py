import os
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
import matplotlib.pyplot as plt
from src.util.constants import *

def lesion_heatmap(masks, mode='mean', save_path=None, figure=None, ax=None, 
                   cut_coords=None, colorbar=True, cmap=BLUE_CONT, vmin=0, vmax=None, 
                   threshold=None, display_mode='ortho', ylabel=True, contour_map=None,
                   fontsize1=FS, fontsize2=FSNL, cbarticks_right=True, cbar_tick_format=None,
                   cbar_ticks_increment=20, cbar_fancy_scaling=True, cbar_label=None,
                   cbar_width_factor=1):
    """
    Create a heatmap of the lesion masks.
    Arguments:
        masks: np.array of shape (n_subjects, n_voxels)
        mode: str, 'mean' or 'sum', how to aggregate across subjects
        fn: str, filename
    """
    template_flat = TEMPLATE.flatten()
    mask = np.mean(masks, axis=0) if mode == 'mean' else np.sum(masks, axis=0)
    mean_in_brain_space = np.zeros(template_flat.shape)
    mean_in_brain_space[template_flat==1] = mask
    mean_in_brain_space = np.reshape(mean_in_brain_space, TEMPLATE.shape)
    img = nib.Nifti1Image(mean_in_brain_space, AFFINE)
    if contour_map is not None:
        contour_in_brain_space = np.zeros(template_flat.shape)
        contour_in_brain_space[template_flat==1] = contour_map
        contour_in_brain_space = np.reshape(contour_in_brain_space, TEMPLATE.shape)
        contour_img = nib.Nifti1Image(contour_in_brain_space, AFFINE)

    if ax is None:
        fig, ax = plt.subplots(figsize=(PW-2,2))
    if cut_coords is None:
        if (contour_map is not None) and (contour_map.sum()>0):
            cut_coords = find_xyz_cut_coords(contour_img)
        else:
            cut_coords = find_xyz_cut_coords(img)
        if display_mode =='ortho':
            pass
        elif display_mode=='yz':
            cut_coords = cut_coords[1:]
        elif display_mode=='z':
            cut_coords = 12
        elif display_mode=='z_ex':
            cut_coords = [cut_coords[2]]
            display_mode = 'z'
        else:
            raise ValueError('Invalid display_mode')
    if cbar_tick_format is None:
        cbar_tick_format = '%i' if mode == 'sum' else '%.1f'
    orthoslicer = plot_stat_map(img, cut_coords=cut_coords, cmap=cmap, vmax=vmax,
                  colorbar=colorbar, vmin=vmin, draw_cross=False, figure=figure, axes=ax,
                  threshold=threshold, cbar_tick_format=cbar_tick_format,
                  annotate=False, display_mode=display_mode)
    
    orthoslicer.draw_cross(cut_coords, linewidth=1, alpha=0.6)
    orthoslicer.annotate(size=fontsize2)
    if contour_map is not None:
        orthoslicer.add_contours(contour_img, colors=RED, linewidths=0.25)
    if colorbar:
        # Move colorbar tick labels closer by adjusting pad (default is 4, reduce it)
        orthoslicer._cbar.ax.tick_params(labelsize=fontsize2, pad=0.5)
        cbar = orthoslicer._cbar
        if cbar_fancy_scaling:
            bounds = cbar.vmin, cbar.vmax
            ticks = np.arange(np.ceil(bounds[0] / cbar_ticks_increment) * cbar_ticks_increment, 
                              bounds[1], #+cbar_ticks_increment, 
                              cbar_ticks_increment)
            # ensure at least two ticks, fall back if vmax<20
            if len(ticks) < 2:
                ticks = np.linspace(bounds[0], bounds[1], 2)
            cbar.set_ticks(ticks)
        if cbarticks_right:
            orthoslicer._cbar.ax.yaxis.set_ticks_position('right')
            orthoslicer._cbar.ax.yaxis.set_label_position('right')
        if cbar_label is not None:
            orthoslicer._cbar.set_label(cbar_label, fontsize=fontsize2)
        # make cbar wider
        pos = cbar.ax.get_position()
        cbar.ax.set_position([pos.x0, pos.y0, pos.width*cbar_width_factor, pos.height])
    if ylabel:
        xloc = -0.05 if display_mode in ['ortho', 'yz'] else -0.03
        orthoslicer.title(f'N={masks.shape[0]}', size=fontsize1, rotation='vertical', 
                        x=xloc, y=0.7, color='black', bgcolor='white', alpha=0)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, transparent=True)
        
    return orthoslicer


def lesion_size_dist(X, save_path):
    fig,ax = plt.subplots(figsize=(2,2))
    lesion_sizes = np.sum(np.reshape(X,(X.shape[0],-1)),axis=1)
    print("min #: of lesioned voxels: ", np.min(lesion_sizes))
    print("max #: of lesioned voxels: ", np.max(lesion_sizes))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.hist(lesion_sizes, bins=20, color=GRAY)
    ax.set_yscale('log')
    ax.set_xlabel('# lesioned voxels')
    ax.set_ylabel('Subject count')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)


    