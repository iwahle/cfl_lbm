import os
import numpy as np
from source.paths import *


def load_scale_data(analysis='cowa_jlo', include_dem=False):
    # load data
    X = np.load(os.path.join(DATA_PATH, analysis, 'X.npy'))
    Y = np.load(os.path.join(DATA_PATH, analysis, 'Y.npy'))
    print(X.shape)
    print(Y.shape)

    # scale data
    assert (np.min(X), np.max(X)) == (0,1)
    Y = Y - np.mean(Y, axis=0)
    Y = Y / np.std(Y, axis=0)

    # add demographics to input if requested    
    if include_dem:
        D = np.load(os.path.join(DATA_PATH, analysis, 'dems.npy'))
        D = D - np.min(D, axis=0)
        D = D / np.max(D, axis=0)
        X = np.concatenate([X,D],axis=1)
        
    # data info
    data_info = {'X_dims': X.shape, 'Y_dims': Y.shape, 'Y_type': 'continuous'}

    return X,Y,data_info
