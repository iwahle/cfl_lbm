import numpy as np

# cowa_jlo analysis params
cj_CDE_params = {  'model' : 'CondExpRidgeCV',
                'model_params' : {  'cv_split' : 10,
                                    'random_state' : 0,
                                    'alphas' : np.logspace(-5,5,11),
                                    'score_fxn' : None}}
cj_cause_cluster_params =  {'model' : 'KMeans',
                            'model_params' : {  'n_clusters' : range(1,11)},
                                                'verbose' : 0,
                                                'tune' : True}
cowa_jlo_params = [cj_CDE_params, cj_cause_cluster_params]

# bdi analysis params
bdi_CDE_params = {  'model' : 'CondExpRidgeCV',
                'model_params' : {  'cv_split' : 10,
                                    'random_state' : 0,
                                    'alphas' : np.logspace(-5,5,11),
                                    'score_fxn' : None}}

# clusterer parameters
bdi_cause_cluster_params =  {'model' : 'KMeans',
                             'model_params' : { 'n_clusters' : range(1,11)},
                             'verbose' : 0,
                             'tune' : True}
bdi_effect_cluster_params =  {'model' : 'KMeans',
                              'model_params' : {'n_clusters' : range(1,11)},
                              'verbose' : 0,
                              'tune' : True,
                              'precompute_distances' : True}
bdi_params = [bdi_CDE_params, bdi_cause_cluster_params, bdi_effect_cluster_params]