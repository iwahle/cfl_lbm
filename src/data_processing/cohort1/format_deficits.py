import os
import numpy as np
from src.util.constants import *

def main():

    dataset = 'cohort1'
    Yraw = np.load(os.path.join(DATA_PATH, f'{dataset}/Yraw.npy'))

    # train test split
    train_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/train_idx.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'))
    Y = Yraw[train_idx]
    Y_test = Yraw[test_idx]

    # normalize Y
    Ymean = np.mean(Y,axis=0)
    Y -= Ymean
    Ystd = np.std(Y,axis=0)
    Y /= Ystd
    Y_test -= Ymean
    Y_test /= Ystd

    np.save(os.path.join(DATA_PATH, f'{dataset}/Y.npy'), Y)
    np.save(os.path.join(DATA_PATH, f'{dataset}/Y_test.npy'), Y_test)
    np.save(os.path.join(DATA_PATH, f'{dataset}/deficit_names.npy'), 
                         ['Language', 'Visuospatial'])

if __name__ == '__main__':
    main()
                       
