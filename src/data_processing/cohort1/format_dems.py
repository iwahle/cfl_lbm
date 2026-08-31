import os
import numpy as np
import pandas as pd
from src.util.constants import *

def main():
    
    dataset = 'cohort1'

    demdf = pd.read_excel(os.path.join(DATA_PATH, f'{dataset}/demographics.xlsx'))
    ids = np.load(os.path.join(DATA_PATH, f'{dataset}/subject_ids.npy'), 
                  allow_pickle=True)
    
    dems = ['AGE_AT_SCAN', 'SEX', 'EDUC']
    dems_arr = np.zeros((len(ids), len(dems)))
    types = []
    for idi,id in enumerate(ids):
        for di, dem in enumerate(dems):
            val = demdf[dem].values[demdf['RedID'].values==id][0]
            if val=='F':
                val = 1
            elif val=='M':
                val = 0
            elif (isinstance(val,str)) and (val[-1]=='+'):
                val = int(val[:-1])
            dems_arr[idi,di] = val

    assert not np.any(np.isnan(dems_arr))
    np.save(os.path.join(DATA_PATH, f'{dataset}/dems_raw.npy'), dems_arr)

    # train test split
    train_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/train_idx.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'))
    dems_train = dems_arr[train_idx]
    dems_test = dems_arr[test_idx]

    # scale
    dems_mean = np.mean(dems_train,axis=0)
    dems_train -= dems_mean
    dems_std = np.std(dems_train,axis=0)
    dems_train /= dems_std
    dems_test -= dems_mean
    dems_test /= dems_std

    np.save(os.path.join(DATA_PATH, f'{dataset}/dems.npy'), dems_train)
    np.save(os.path.join(DATA_PATH, f'{dataset}/dems_test.npy'), dems_test)

    dem_names = ['Age at scan (years)', 'Sex', 'Educ. (years)']
    np.save(os.path.join(DATA_PATH, f'{dataset}/dem_names.npy'), dem_names)

if __name__ == '__main__':
    main()