import numpy as np
from source.util import *
from source.paths import *
from source.cfl_params import *
from cfl.experiment import Experiment
        
def main():
    '''
    Compare CFL partition found from 21 question-wise BDI representation to
    mean BDI aggregate quantity.
    Arguments: None
    Returns: None
    '''

    save_path = os.path.join(RESULTS_PATH, 'bdi','compare_mbdi')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    X,Y,data_info = load_scale_data(analysis='bdi')
    Y_mbdi = np.mean(Y, axis=1)[:,np.newaxis]
    data_info['Y_dims'] = Y_mbdi.shape
    assert Y_mbdi.shape==(520,1)

    # run cfl with mbdi instead of 21 questions
    block_names = ['CondDensityEstimator', 'CauseClusterer']
    block_params = bdi_params[:2]
    my_exp = Experiment(X_train=X, Y_train=Y_mbdi, data_info=data_info, 
                        block_names=block_names, block_params=block_params, 
                        blocks=None, verbose=1, results_path=save_path)
    train_results = my_exp.train()
    C = train_results['CauseClusterer']['x_lbls']
    np.save(os.path.join(save_path, 'C.npy'), C)

if __name__ == '__main__':
    '''
    Example usage:
        python compare_mbdi.py
    '''
    main()