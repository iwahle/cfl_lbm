import numpy as np
from scipy.stats import mode
from source.paths import *
import os

'''
This script generates simulated data of lesion masks and associated language and
visuospatial test scores. For randomly selected lesion centers, it constructs
spherical lesions of uniformly varying radii and then generates corresponding
test scores. To build in established relationships between lesion hemispheres
and test scores, language scores for left-hemispheric lesions are drawn from a
normal distribution with a lower mean than right-hemispheric lesions.
Visuospatial scores for right-hemispheric lesions are drawn from a normal
distribution with a lower mean than left-hemispheric lesions.

The data constructed here is purely for demonstrating the usage of the code,
and does not capture many of the complexities of real lesion-behavior data.'''

BRAIN_DIMS = (19,22,19)

def generate_lesion_mask(x, brain_mask):
    
    # generate 1mm mask
    m = np.zeros(BRAIN_DIMS)
    y = np.random.choice(BRAIN_DIMS[1])
    z = np.random.choice(BRAIN_DIMS[2])
    rad = np.random.choice(range(1,8))
    xmin,xmax = np.max([0, x-rad]), np.min([BRAIN_DIMS[0], x+rad])
    ymin,ymax = np.max([0, y-rad]), np.min([BRAIN_DIMS[1], y+rad])
    zmin,zmax = np.max([0, z-rad]), np.min([BRAIN_DIMS[2], z+rad])
    m[xmin:xmax, ymin:ymax, zmin:zmax] = 1

    # mask anything outside of brain region
    m[np.where(brain_mask==0)] = 0

    # flatten m
    m = np.reshape(m, -1)
    return m

def generate_X(n_samples):
    X = []
    x_coord = np.random.choice(BRAIN_DIMS[0], n_samples)
    brain_mask = np.load(os.path.join(DATA_PATH, 'simulated/mask_1cm.npy'))
    for i in range(n_samples):
        X.append(generate_lesion_mask(x_coord[i], brain_mask))
    X = np.array(X)
    return X, x_coord


def generate_Y(hemi):

    l = np.zeros(len(hemi))
    l[hemi==0] = np.random.normal(0.4, 0.1, np.sum(hemi==0))
    l[hemi==1] = np.random.normal(0.6, 0.1, np.sum(hemi==1))

    v = np.zeros(len(hemi))
    v[hemi==0] = np.random.normal(0.6, 0.1, np.sum(hemi==0))
    v[hemi==1] = np.random.normal(0.4, 0.1, np.sum(hemi==1))

    lv = np.stack((l,v), axis=1)
    return lv


def main():
    
    n_samples = 1000
    X,x_coord = generate_X(n_samples)
    print(X.shape, x_coord.shape)
    np.save(os.path.join(DATA_PATH, 'simulated/X.npy'), X)
    np.save(os.path.join(DATA_PATH, 'simulated/x_coord.npy'), x_coord)

    hemi = x_coord > (BRAIN_DIMS[0]//2)
    Y = generate_Y(hemi)
    print(Y.shape)
    np.save(os.path.join(DATA_PATH, 'simulated/Y.npy'), Y)

if __name__ == '__main__':
    main()