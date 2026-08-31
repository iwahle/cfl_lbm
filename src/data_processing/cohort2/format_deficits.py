import os
import numpy as np
from src.util.constants import *

def main():

    dataset = 'cohort2'
    Yraw = np.load(os.path.join(DATA_PATH, f'{dataset}/Yraw.npy'))

    # train test split
    train_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/train_idx.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'))
    Y = Yraw[train_idx]
    Y_test = Yraw[test_idx]

    assert np.all(Y>=0) & np.all(Y<=3)
    assert np.all(Y_test>=0) & np.all(Y_test<=3)

    Ymean = np.mean(Y)
    Ystd = np.std(Y)
    Y = (Y - Ymean) / Ystd
    Y_test = (Y_test - Ymean) / Ystd

    
    np.save(os.path.join(DATA_PATH, f'{dataset}/Y.npy'), Y)
    np.save(os.path.join(DATA_PATH, f'{dataset}/Y_test.npy'), Y_test)

if __name__ == '__main__':
    main()