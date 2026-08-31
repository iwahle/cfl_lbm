import numpy as np
from src.util.constants import RS

block_names = ['CondDensityEstimator', 'CauseClusterer']
cde_params_tune = {
    'model' : 'CondExpRidgeCV',
    'model_params' : {  'cv_split' : 10,
                        'random_state' : RS,
                        'alphas' : np.logspace(-2,8,22),
                        'score_fxn' : None}}
cc_params_tune = {
    'model' : 'KMeans',
    'model_params' : {  'n_clusters' : range(1,15),
                        'random_state' : [RS],
                        'n_init' : [50],
                        'init' : ['random']},
    'tune' : True}
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
                        
# block_params = [cde_params_tune, cc_params_tune]
block_params = [cde_params, cc_params]