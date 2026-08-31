import os
import numpy as np
import pandas as pd
from src.util.constants import *

def main():
    ''' Saves:

    - etiology_mapping.npy: dictionary mapping etiology names to numeric values
    - etiology.npy: numeric etiology values for train set
    - etiology_test.npy: numeric etiology values for test set
    - etiology_names.npy: list of etiology names
    '''
    dataset = 'cohort1'

    demdf = pd.read_excel(os.path.join(DATA_PATH, f'{dataset}/demographics.xlsx'))
    ids = np.load(os.path.join(DATA_PATH, f'{dataset}/subject_ids.npy'), 
                  allow_pickle=True)
    
    # Extract etiology descriptions for each subject
    etiology_descriptions = []
    for id in ids:
        etiology_val = demdf['Etiology_description'].values[demdf['RedID'].values==id][0]
        etiology_descriptions.append(etiology_val)
    
    # Create mapping from etiology names to numeric values
    unique_etiologies = list(set(etiology_descriptions))
    unique_etiologies.sort()  # Sort for consistent ordering
    etiology_to_num = {etiology: i for i, etiology in enumerate(unique_etiologies)}
    
    # Convert etiology descriptions to numeric values
    etiology_nums = np.array([etiology_to_num[etiology] for etiology in etiology_descriptions])
    
    # Save raw etiology data
    np.save(os.path.join(DATA_PATH, f'{dataset}/etiology_raw.npy'), etiology_nums)
    
    # Save etiology mapping dictionary
    np.save(os.path.join(DATA_PATH, f'{dataset}/etiology_mapping.npy'), 
            etiology_to_num, allow_pickle=True)
    
    # train test split
    train_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/train_idx.npy'))
    test_idx = np.load(os.path.join(DATA_PATH, f'{dataset}/test_idx.npy'))
    etiology_train = etiology_nums[train_idx]
    etiology_test = etiology_nums[test_idx]

    # Save train/test splits
    np.save(os.path.join(DATA_PATH, f'{dataset}/etiology.npy'), etiology_train)
    np.save(os.path.join(DATA_PATH, f'{dataset}/etiology_test.npy'), etiology_test)
    
    # Save etiology names for reference
    etiology_names = list(etiology_to_num.keys())
    np.save(os.path.join(DATA_PATH, f'{dataset}/etiology_names.npy'), etiology_names)
    
    print(f"Found {len(unique_etiologies)} unique etiologies:")
    for etiology, num in etiology_to_num.items():
        print(f"  {num}: {etiology}")

if __name__ == '__main__':
    main()
