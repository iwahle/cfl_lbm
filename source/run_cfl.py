import os
from cfl_params import *
from source.paths import *
from source.util import load_scale_data
from cfl import Experiment
import argparse

def main(analysis, include_dem):
    X,Y,data_info = load_scale_data(analysis=analysis, include_dem=include_dem)
    
    block_names = ['CondDensityEstimator', 'CauseClusterer']
    if analysis=='bdi':
        block_names.append('EffectClusterer')
    block_params = cowa_jlo_params if analysis=='cowa_jlo' else bdi_params
    save_path = os.path.join(RESULTS_PATH, analysis)

    my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                        block_names=block_names, block_params=block_params, 
                        blocks=None, verbose=1, results_path=save_path)
    train_results = my_exp.train()
    print(train_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=str)
    parser.add_argument('--include_dem', type=int)
    
    main(**vars(parser.parse_args()))