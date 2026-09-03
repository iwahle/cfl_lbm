from src.util.constants import RS

block_names = ['CondDensityEstimator', 'CauseClusterer', 'EffectClusterer']

# parameters here are fixed for simplicity - see other analyses for examples
# of parameter tuning

cde_params = {
    'model' : 'CondExpRidgeCV',
    'model_params' : {  'cv_split' : 10,
                        'random_state' : RS,
                        'alphas' : 10,
                        'score_fxn' : None}}

cc_params = {
    'model' : 'KMeans',
    'model_params' : {  'n_clusters' : 3,
                        'random_state' : RS,
                        'n_init' : 50,
                        'init' : 'random'}}

ec_params = {
    'model' : 'KMeans',
    'model_params' : {  'n_clusters' : 3,
                        'random_state' : RS,
                        'n_init' : 50,
                        'init' : 'random'}}
                        
block_params = [cde_params, cc_params, ec_params]