import numpy as np
import pandas as pd
import h5py
import os, sys
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVR, LinearSVC


def data(data_path, label_path, subset_label, predict_label, split_label, subset_query, tt_split, genes_to_select):
    # Select genes and set row size variable
    data_h5 = h5py.File(data_path)
    rows = data_h5['gene_id'].shape[0]
    gene_names = data_h5['gene_id'][:].astype('str')

    # Read labels CSV
    label_set = pd.read_csv(label_path, index_col=0, header=0)

    # Account for All query
    if subset_query == 'All':
        subset_query = np.unique(label_set[subset_label])

    # Select the subset
    subset_keys = []
    for subset in subset_query:
        subset_selection = label_set[subset_label] == subset
        notnull_selection = label_set[predict_label].notnull()
        subset_keys += list(label_set[notnull_selection & subset_selection].index.values)
    subset_set = label_set.loc[subset_keys]

    # Split the data to test and train on split_label level
    train_keys = []
    test_keys = []
    split_labels = np.unique(subset_set[split_label])
    for split in split_labels:
        split_keys = list(subset_set[subset_set[split_label] == split].index.values)
        train_k, test_k = train_test_split(split_keys, test_size=tt_split)
        train_keys += train_k
        test_keys += test_k

    # Get all labels and set related variables
    sample_to_label = subset_set[predict_label]
    unique_labels = np.unique(sample_to_label).astype(str)

    y_train = subset_set.loc[train_keys].as_matrix(columns=[predict_label]).flatten().astype(str)
    y_test = subset_set.loc[test_keys].as_matrix(columns=[predict_label]).flatten().astype(str)

    X_train = np.empty([len(train_keys), rows])
    for idx, k in enumerate(train_keys):
        X_train[idx][:] = np.asarray(data_h5[k][:])

    X_test = np.empty([len(test_keys), rows])
    for idx, k in enumerate(test_keys):
        X_test[idx][:] = np.asarray(data_h5[k][:])

    if genes_to_select is not None:
        gene_index = pd.Series(range(len(gene_names)), index=gene_names)
        indexes = [gene_index[x] for x in genes_to_select]
        X_train = np.take(X_train, indexes, axis=1)
        X_test = np.take(X_test, indexes, axis=1)
        gene_names = np.array(genes_to_select)

    return X_train, X_test, y_train, y_test, unique_labels, gene_names, sample_to_label


def gene_predictive_accuracy(output, selected_genes_ids, folds=10, latex=False, decimal_val=4):
    genes_to_select = selected_genes_ids
    out_conf = output[0]['config']
    classifier_name = out_conf['classifier_name']
    regression = classifier_name in ['RandomForestRegressor', 'LinearSVR']

    data_file = '../data/TCGA/exp_TCGA_coded_normalized.hdf5'
    label_file = '../data/TCGA/exp_TCGA_coded_labels_add.csv'

    tt_split = out_conf['tt_split']
    subset_label = out_conf['subset_label']
    subset_query = out_conf['subset_query']
    predict_label = out_conf['predict_label']
    split_label = out_conf['split_label']

    if regression:
        classifiers = [
            RandomForestRegressor(n_estimators=100, n_jobs=-1),
            LinearSVR(C=0.25),
        ]
    else:
        classifiers = [
            LinearSVC(C=0.25),
            RandomForestClassifier(n_estimators=100, n_jobs=-1),
        ]

    print('Reading data')
    X_train, X_test, y_train, y_test, unique_labels, gene_names, sample_to_label = data(data_file, label_file,
                                                                                        subset_label, predict_label,
                                                                                        split_label, subset_query,
                                                                                        tt_split, genes_to_select)

    results = {}

    for c in classifiers:
        c_name = c.__class__.__name__
        results[c_name] = []
        for _ in range(folds):
            print('Predicting and evaluating {}'.format(c_name))
            c.fit(X_train, y_train)
            y_pred = c.predict(X_test)
            y_test = y_test.astype(type(y_pred[0]))
            if regression:
                score = r2_score(y_test, y_pred)
            else:
                score = accuracy_score(y_test, y_pred)
            results[c_name].append(score)

    if latex:
        print('We have {} selected genes as a result, doing the same prediction again but only with these genes we get the following accuracies averaged over {} folds:\\\\'.format(len(selected_genes_ids), folds))

    keys = results.keys()
    if folds > 1:
        res = {}
        for k in keys:
            avr = np.average(results[k])
            var = np.var(results[k])
            res[k + '_avr'] = np.average(results[k])
            if latex:
                print('{}: {} ({})\\\\'.format(k, '%.{}f'.format(decimal_val)%(avr), '%.{}f'.format(decimal_val)%(var)))
        results = {**res, **results}
    else:
        for k in keys:
            if latex:
                print('{}: {}\\\\'.format(k, '%.{}f'.format(decimal_val)%(results[k][0])))
    return results











