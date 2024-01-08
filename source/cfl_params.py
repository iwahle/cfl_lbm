import numpy as np

# choice of random_state parameters in the following may slightly vary 
# macrostate assignments

# cowa_jlo analysis params
cj_CDE_params = {  'model' : 'CondExpRidgeCV',
                'model_params' : {  'cv_split' : None,
                                    'random_state' : 0,
                                    'alphas' : 10,
                                    'score_fxn' : None}}
cj_cause_cluster_params =  {'model' : 'KMeans',
                            'model_params' : {  'n_clusters' : 5,
                                                'random_state' : 0},
                            'verbose' : 0,
                            'tune' : False}
cowa_jlo_params = [cj_CDE_params, cj_cause_cluster_params]

# fixed bdi params for cca comparison
bdi_CDE_params = {'model' : 'CondExpRidgeCV',
                        'model_params' : {  'cv_split' : None,
                                            'random_state' : 0,
                                            'alphas' : 10,
                                            'score_fxn' : None}}
bdi_cause_cluster_params=  { 'model' : 'KMeans',
                                    'model_params' : {'n_clusters' : 3,
                                                      'random_state' : 0},
                                    'verbose' : 0,
                                    'tune' : False}
bdi_effect_cluster_params =  {'model' : 'KMeans',
                              'model_params' : {'n_clusters' : 3,
                                                'random_state' : 0},
                              'verbose' : 0,
                              'tune' : False,
                              'precompute_distances' : True}                                    
bdi_params = [bdi_CDE_params, bdi_cause_cluster_params, 
              bdi_effect_cluster_params]

# simulated data example params
sim_CDE_params = {  'model' : 'CondExpRidgeCV',
                'model_params' : {  'cv_split' : None,
                                    'random_state' : 0,
                                    'alphas' : 1,
                                    'score_fxn' : None}}
sim_cause_cluster_params =  {'model' : 'KMeans',
                            'model_params' : {  'n_clusters' : 2,
                                                'random_state' : 0},
                            'verbose' : 0,
                            'tune' : False}
sim_params = [sim_CDE_params, sim_cause_cluster_params]
################################################################################
# params to tune hyperparameters from scratch

# cowa_jlo
cj_CDE_params_tune = {  'model' : 'CondExpRidgeCV',
                'model_params' : {  'cv_split' : 10,
                                    'random_state' : 0,
                                    'alphas' : np.logspace(-5,5,11),
                                    'score_fxn' : None}}
cj_cause_cluster_params_tune =  {'model' : 'KMeans',
                            'model_params' : {  'n_clusters' : range(1,11),
                                                'random_state' : [0]},
                            'verbose' : 0,
                            'tune' : True}
cowa_jlo_params_tune = [cj_CDE_params_tune, cj_cause_cluster_params_tune]

# bdi analysis params
bdi_CDE_params_tune = {  'model' : 'CondExpRidgeCV',
                    'model_params' : {  'cv_split' : 10,
                                        'random_state' : 0,
                                        'alphas' : np.logspace(-5,5,11),
                                        'score_fxn' : None}}
bdi_cause_cluster_params_tune =  {'model' : 'KMeans',
                             'model_params' : { 'n_clusters' : range(1,11),
                                                'random_state' : [0]},
                             'verbose' : 0,
                             'tune' : True}
bdi_effect_cluster_params_tune =  {'model' : 'KMeans',
                              'model_params' : {'n_clusters' : range(1,11),
                                                'random_state' : [0]},
                              'verbose' : 0,
                              'tune' : True,
                              'precompute_distances' : True}
bdi_params_tune = [bdi_CDE_params_tune, bdi_cause_cluster_params_tune, 
                   bdi_effect_cluster_params_tune]