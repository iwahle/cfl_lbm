import os
import numpy as np
import nibabel as nib
from src.util.data_util import load_brain
import matplotlib.colors as clr
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
FS = 7
plt.rcParams["font.size"] = FS
FSNL = 6 # nilearn fonts funny


DATA_PATH = 'data'
RESULTS_PATH = 'results'
FIG_PATH = 'figures'

ATLAS_FN = 'Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'

BRAIN_DIMS = [91, 109, 91]

img_for_affine = nib.load(os.path.join(DATA_PATH, 'lesion_masks/0018.nii.gz'))
ori = nib.orientations.aff2axcodes(img_for_affine.affine)
if ori[0]!='R':
    img_for_affine = img_for_affine.slicer[::-1, :, :]
ori = nib.orientations.aff2axcodes(img_for_affine.affine)
AFFINE = img_for_affine.affine
AFFINE[0,-1] *= -1 # shift onto template

EX_ROIS = [25,30,35]

PW = 6.5
CW1 = 3.42
CW1p5 = 4.5
CW2 = 7

TEMPLATE = load_brain(os.path.join(DATA_PATH, 'vol_mask_2mm.nii.gz'))


# colors
GRAY = "#8C979A"
BLUE = '#0076be'
DARK_BLUE = '#005284'
BLUE_CONT = clr.LinearSegmentedColormap.from_list('blue_continous', 
                                                  ['#DFE9F5', BLUE], N=256)
GREEN = "#65a765"
DARK_GREEN = "#0A6522"
YELLOW = "#FBBC05"
GRAYBLUE = '#708090'
RED = '#D0312D'

GREEN_DISC = clr.LinearSegmentedColormap.from_list('green_discrete', 
    ['#CCE7C9', '#8BCA84', '#5BB450', '#3B8132'], N=4)
GREEN_CONT = clr.LinearSegmentedColormap.from_list('green_continous', 
    ['#CCE7C9', '#276221'], N=256)

# cog = 0, som = 1, else = 2
SOMCOG = np.array([0,2,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,2])
SOMCOG_ORDER = np.array([0,2,3,4,5,6,7,8,9,10,11,12,13,16,14,15,17,18,19,1,20])

CFLC = np.zeros(21)
CFLC[[15,18]] = 1
CFLC[[14,19]] = 2
CFLC[[8,5,9]] = 3

RS = 42