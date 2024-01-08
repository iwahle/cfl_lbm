import os
import numpy as np
from source.paths import *

# cog = 0, som = 1, else = 2
SOMCOG = np.array([0,2,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,2])
SOMCOG_ORDER = np.array([0,2,3,4,5,6,7,8,9,10,11,12,13,16,14,15,17,18,19,1,20])

def load_scale_data(analysis, include_dem=False):
    '''
    Load and scale data for a given analysis.
    Arguments:
        - analysis (str): name of directory within `data` containing data for
          the current analysis.
        - include_dem (bool): whether to include demographic variables in the
            analysis. 1 = yes, 0 = no.
    Returns:
        - X (np.ndarray): input data.
        - Y (np.ndarray): output data.
        - data_info (dict): dictionary containing information about the data
          needed by CFL.
    '''
    
    # load data
    X = np.load(os.path.join(DATA_PATH, analysis, 'X.npy'))
    Y = np.load(os.path.join(DATA_PATH, analysis, 'Y.npy'))
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)

    # scale data
    assert (np.min(X), np.max(X)) == (0,1), f'{np.min(X)}, {np.max(X)}' # lesion masks
    if (analysis=='cowa_jlo') or (analysis=='simulated'):
        Y = Y - np.mean(Y, axis=0)
        Y = Y / np.std(Y, axis=0)
    elif analysis=='bdi':
        Y = Y / 3
        assert (np.min(Y), np.max(Y)) == (0,1)
    else:
        raise ValueError('Unknown analysis: {analysis}, modify source/util.py' +
                         ' to add preprocessing support for this analysis.')

    # add demographics to input if requested    
    if include_dem:
        D = np.load(os.path.join(DATA_PATH, analysis, 'dems.npy'))
        D = D - np.min(D, axis=0)
        D = D / np.max(D, axis=0)
        X = np.concatenate([X,D],axis=1)
        
    # data info
    data_info = {'X_dims': X.shape, 'Y_dims': Y.shape, 'Y_type': 'continuous'}

    return X,Y,data_info
