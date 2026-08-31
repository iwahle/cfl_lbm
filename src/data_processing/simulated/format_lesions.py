import os
import numpy as np
import pandas as pd
from src.util.constants import *
from src.util.data_util import load_brains_from_ids

def main():

    #### TRAIN DATA ############################################################
    # load list of brain ids
    csv_path = os.path.join(DATA_PATH, 'simulated/n200_sev_linear_10noise.csv')
    ids = pd.read_csv(csv_path)['ID'].values

    # collect lesion mask for each id
    masks, found_subj_ids = load_brains_from_ids(ids, flip=True)
    print('masks shape', masks.shape)
    np.save(os.path.join(DATA_PATH, 'simulated/X.npy'), masks)
    np.save(os.path.join(DATA_PATH, 'simulated/found_subj.npy'), found_subj_ids)


    #### TEST ##################################################################
    # load list of brain ids
    csv_path_test = os.path.join(DATA_PATH, 
                                 'simulated/r46_n200_sev_linear_10noise.csv')
    test_ids = pd.read_csv(csv_path_test)['ID'].values

    # collect lesion mask for each id
    test_masks, test_found_subj_ids = load_brains_from_ids(test_ids, flip=True)
    print('test_masks shape', test_masks.shape)
    np.save(os.path.join(DATA_PATH, 'simulated/X_test.npy'), test_masks)
    np.save(os.path.join(DATA_PATH, 'simulated/found_subj_test.npy'), 
            test_found_subj_ids)

if __name__ == '__main__':
    main()