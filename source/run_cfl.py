import os
import argparse
from cfl import Experiment
from source.paths import *
from source.cfl_params import *
from source.util import load_scale_data

def main(analysis, include_dem):
    '''
    Run CFL on data located at `data/[analysis]`.
    Arguments:
        - analysis (str): name of directory within `data` containing data for
          the current analysis.
        - include_dem (bool): whether to include demographic variables in the
            analysis. 1 = yes, 0 = no.
    Returns:
        - my_exp (Experiment): CFL experiment object.
    '''

    # load data
    X,Y,data_info = load_scale_data(analysis=analysis, include_dem=include_dem)
    
    # design CFL experiment
    block_names = ['CondDensityEstimator', 'CauseClusterer']
    if analysis=='bdi':
        block_names.append('EffectClusterer')
    if analysis=='cowa_jlo':
        block_params = cowa_jlo_params
    elif analysis=='bdi':
        block_params = bdi_params
    elif analysis=='simulated':
        block_params = sim_params
    save_path = os.path.join(RESULTS_PATH, analysis)

    my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                        block_names=block_names, block_params=block_params, 
                        blocks=None, verbose=1, results_path=save_path)
    
    # fit model
    train_results = my_exp.train()
    print(train_results)
    return my_exp

if __name__ == '__main__':
    ''' 
    Example usage:
        python run_cfl.py --analysis bdi --include_dem 0
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=str)
    parser.add_argument('--include_dem', type=int)
    main(**vars(parser.parse_args()))