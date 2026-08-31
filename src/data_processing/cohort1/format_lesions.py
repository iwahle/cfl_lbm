import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.util.constants import *
from src.util.data_util import load_brain, load_brains_from_ids

def main():

    dataset = 'cohort1'

    # load list of brain ids
    ids = np.load(os.path.join(DATA_PATH, f'{dataset}/subject_ids.npy'),
                  allow_pickle=True).astype(int).astype(str)
    print('cohort size: ', len(ids))

    # collect lesion mask for each id
    masks, found_subj_ids = load_brains_from_ids(ids)
    print('masks shape', masks.shape)

    # split into train and test
    train_size = int(0.5 * len(masks))
    rnp = np.random.RandomState(RS)
    train_idx = rnp.choice(len(masks), train_size, replace=False)
    test_idx = np.array(list(set(range(len(masks))) - set(train_idx)))
    np.save(os.path.join(DATA_PATH, f'{dataset}/train_idx.npy'), train_idx)
    np.save(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'), test_idx)

    np.save(os.path.join(DATA_PATH, f'{dataset}/X.npy'), masks[train_idx])
    np.save(os.path.join(DATA_PATH, f'{dataset}/X_test.npy'), masks[test_idx])
    np.save(os.path.join(DATA_PATH, f'{dataset}/found_subj.npy'), 
            found_subj_ids[train_idx])
    np.save(os.path.join(DATA_PATH, f'{dataset}/found_subj_test.npy'), 
            found_subj_ids[test_idx])

if __name__ == '__main__':
    main()