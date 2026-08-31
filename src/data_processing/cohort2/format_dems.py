import os
import numpy as np
import pandas as pd
from src.util.constants import *

def main():
    
    dataset = 'cohort2'

    demdf1 = pd.read_excel(os.path.join(DATA_PATH, f'{dataset}/iowa_demographics.xlsx'))
    demdf2 = pd.read_excel(os.path.join(DATA_PATH, f'{dataset}/grafman_demographics.xlsx'))
    ids = np.load(os.path.join(DATA_PATH, f'{dataset}/subject_ids.npy'), 
                  allow_pickle=True)
    
    dems = ['AGE_AT_SCAN', 'SEX', 'EDUC']
    dems_graf = ['age', '', 'Education']
    dems_arr = np.zeros((len(ids), len(dems)))
    types = []
    for idi,id in enumerate(ids):
        for di, dem in enumerate(dems):
            if 'grafman' in str(id):
                if dem=='SEX': # all male cohort
                    val = 0
                else:
                    val = demdf2[dems_graf[di]].values[demdf2['vhis_id'].values==int(id.split('_')[-1])][0]
            else:
                val = demdf1[dem].values[demdf1['RedID'].values==id][0]
                if val=='F':
                    val = 1
                elif val=='M':
                    val = 0
                elif (isinstance(val,str)) and (val[-1]=='+'):
                    val = int(val[:-1])
                elif isinstance(val,str):
                    print(id, dem, val)
                    val = np.nan
                elif np.isnan(val) & (dem=='AGE_AT_SCAN'):
                    val = demdf1['Age_at_Onset_Years'].values[demdf1['RedID'].values==id][0]
            
            dems_arr[idi,di] = val

    # impute missing education values with mean (should be 4 of them, including
    # 3573 EDUC '-', and 3 empty values)
    print(np.sum(np.isnan(dems_arr),axis=0))
    dems_arr[np.isnan(dems_arr[:,2]),2] = np.nanmean(dems_arr[:,2])
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