"""
This script takes a hdf5 or csv expression dataset and will normalize it values
It will run quantile normalization and log 2 normalization per gene
"""

import os, sys
import pandas as pd
import numpy as np

sys.path.append('../')
from tools import to_hdf5_file, read_hdf5_file

# Parameters
quantile_value = 0.75
scaling_factor = 1000
set_name = 'TCGA'
subset_name = '_coded'
from_csv = False
to_csv = False

raw_path = '../data/{}'.format(set_name)
raw_set = 'exp_{}{}_raw.hdf5'.format(set_name, subset_name)

output_path = '../data/normalized/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_fname = 'exp_{}{}_normalized'.format(set_name, subset_name)
output_file = os.path.join(output_path, output_fname)


print('Reading file')
if from_csv:
    raw_data = pd.read_csv(os.path.join(raw_path, raw_set), header=0, index_col=0)
else:
    raw_data = read_hdf5_file(os.path.join(raw_path, raw_set))

# Nomalization functions
def normalize(x):
    sample_quantile = x.quantile(quantile_value)
    if sample_quantile != 0:
        x = (scaling_factor / sample_quantile) * x
    x = np.log2(x + 1)
    return x

print('Normalizing data')
normalized_data = raw_data.apply(normalize)

if to_csv:
    print('Writing to CSV')
    normalized_data.to_csv(output_file + '.csv')

print('Writing to HDF5')
to_hdf5_file(output_file + '.hdf5', normalized_data)