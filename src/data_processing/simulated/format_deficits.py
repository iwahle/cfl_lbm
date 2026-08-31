
import os
import argparse
import numpy as np
import pandas as pd
from src.util.constants import *

def main(deficits):
    #### TRAIN #################################################################

    # load list of ids
    csv_path = os.path.join(DATA_PATH, f'simulated/n200_sev_linear_10noise.csv')
    deficits_df = pd.read_csv(csv_path)
    ids = deficits_df['ID'].values

    # load available subjects
    found_subj_ids = np.load(os.path.join(DATA_PATH, 'simulated/found_subj.npy'))
    found_subj = np.array([1 if id in found_subj_ids else 0 for id in ids])

    Y = np.zeros((len(found_subj_ids),len(deficits)))
    for di,deficit in enumerate(deficits):
        # save deficits
        deficit_vals = deficits_df[deficit].values[found_subj == 1]
        np.save(os.path.join(DATA_PATH, f'simulated/Y_{deficit}.npy'), 
                deficit_vals)
        Y[:,di] = deficit_vals        
    np.save(os.path.join(DATA_PATH, 'simulated/Y.npy'), Y)

    #### TEST ##################################################################

    # load list of ids
    csv_path_test = os.path.join(DATA_PATH, 
                                 f'simulated/r46_n200_sev_linear_10noise.csv')
    deficits_test_df = pd.read_csv(csv_path_test)
    ids = deficits_test_df['ID'].values

    # load available subjects
    found_subj_id_test = np.load(os.path.join(DATA_PATH, 
                                 'simulated/found_subj_test.npy'))
    found_subj_test = np.array([1 if id in found_subj_id_test else 0 for id in ids])

    Y_test = np.zeros((len(found_subj_id_test),len(deficits)))
    for di,deficit in enumerate(deficits):
        csv_path_test = os.path.join(DATA_PATH, 
            f'simulated/{deficit}_n200_sev_linear_10noise.csv')
        deficits_test_vals = pd.read_csv(csv_path_test)[deficit].values[found_subj_test==1]
        np.save(os.path.join(DATA_PATH, f'simulated/Y_{deficit}_test.npy'), 
                deficits_test_vals)
        Y_test[:,di] = deficits_test_vals  
    print('Y_test shape', Y_test.shape) 
    np.save(os.path.join(DATA_PATH, 'simulated/Y_test.npy'), Y_test)     

    np.save(os.path.join(DATA_PATH, 'simulated/deficit_names.npy'), deficits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deficits', nargs='+', default=['r46', 'r74', 'r77', 'r87'])
    main(**vars(parser.parse_args()))