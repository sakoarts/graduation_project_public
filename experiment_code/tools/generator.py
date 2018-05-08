import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import h5py


class DataGeneratorNumpy(object):
    'Generates data for Keras'

    def __init__(self, path, train=True, batch_size=32, split=0.33, random_state=33):
        'Initialization'
        self.batch_size = batch_size
        self.path = path
        self.split= split
        self.random_state = random_state

        data = np.load(self.path)
        X_train, X_test = train_test_split(data, test_size=split, random_state=random_state)

        self.set = X_train if train else X_test

    def generate(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            if self.shuffle:
                np.random.shuffle(self.set)
            X = self.set[0:32]

            yield X, X


class DataGeneratorHDF5(object):
    'Generates data for Keras'

    def __init__(self, path, keys, batch_size=32, preload=True):
        'Initialization'
        self.batch_size = batch_size
        self.path = path
        self.preload = preload
        self.keys = keys

    def generate(self):
        'Generates batches of samples'
        f = h5py.File(self.path, 'r')

        if self.preload:
            cached_hdf5_file = {}
            for key in self.keys:
                cached_hdf5_file[key] = f[key][:]
        else:
            cached_hdf5_file = f

        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batch = np.random.choice(self.keys, self.batch_size)

            X = np.empty(self.batch_size, dtype=object)
            for idx, b in enumerate(batch):
                X[idx] = cached_hdf5_file[b][:]
            X = np.stack(X)

            yield X, X

class DataGeneratorClassifier(object):
    'Generates data for Keras'

    def __init__(self, data_path, sample_to_label, keys, number_of_classes, rows, batch_size=32, class_weights=False, genes_to_select=None):
        'Initialization'
        self.batch_size = batch_size
        self.data_path = data_path
        self.rows = rows
        self.keys = keys
        self.class_weights = class_weights
        self.sample_to_label = sample_to_label.loc[keys]
        self.number_of_classes = number_of_classes
        self.genes_to_select = genes_to_select

    def generate(self):
        'Generates batches of samples'
        f = h5py.File(self.data_path, 'r')

        # Reading data
        cached_hdf5_file = {}
        if self.genes_to_select is None:
            for key in self.keys:
                cached_hdf5_file[key] = f[key][:]
        else:
            for key in self.keys:
                cached_hdf5_file[key] = np.take(f[key][:], self.genes_to_select)

        # Reading labels
        labels_one_hot = self.sample_to_label['onehot']

        # Reading sample weights
        if self.class_weights:
            sample_weights = compute_sample_weight('balanced', self.sample_to_label['label'])
            sample_weights = pd.DataFrame(sample_weights, index=self.sample_to_label.index)
            sw_shape = np.empty([self.batch_size])
        else:
            sw_shape = None

        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batch = np.random.choice(self.keys, self.batch_size)

            X = np.empty([self.batch_size, self.rows])
            y = np.empty([self.batch_size, self.number_of_classes])
            sw = sw_shape
            for idx, b in enumerate(batch):
                X[idx][:] = cached_hdf5_file[b][:]
                y[idx][:] = labels_one_hot.loc[b]
                if sw_shape is not None:
                    sw[idx] = sample_weights.loc[b]

            yield X, y, sw