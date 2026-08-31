import os
import numpy as np
from src.util.constants import *


found_subj = np.load(os.path.join(DATA_PATH, 'simulated/found_subj.npy'))
fns = []
for id in found_subj:
    fn = f'{str(id).zfill(4)}.nii.gz'
    fn = os.path.join(DATA_PATH, 'lesion_masks_lh', fn)
    fns.append(fn)

# load list of brain ids
found_subj_test = np.load(os.path.join(DATA_PATH, 'simulated/found_subj_test.npy'))
test_fns = []
for id in found_subj_test:
    fn = f'{str(id).zfill(4)}.nii.gz'
    fn = os.path.join(DATA_PATH, 'lesion_masks_lh', fn)
    test_fns.append(fn)

print('fns', fns)
print('test_fns', test_fns)
print(len(fns), len(test_fns))
# save as txt
np.savetxt(os.path.join(DATA_PATH, 'simulated/lesion_fps.txt'), fns, fmt='%s')
np.savetxt(os.path.join(DATA_PATH, 'simulated/lesion_fps_test.txt'), test_fns, fmt='%s')