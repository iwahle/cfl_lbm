import numpy as np
from src.util.constants import RS

block_names = ['CondDensityEstimator', 'CauseClusterer']
cde_params = {
    'model' : 'CondExpRidgeCV',
    'model_params' : {  'cv_split' : 10,
                        'random_state' : RS,
                        'alphas' : 4.64158883e+04,
                        'score_fxn' : None}}
cc_params = {
    'model' : 'KMeans',
    'model_params' : {  'n_clusters' : 5,
                        'random_state' : RS,
                        'n_init' : 50,
                        'init' : 'random'}}
block_params = [cde_params, cc_params]