import numpy as np
import pandas as pd

from tools.plots import evaluate_model

import h5py
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #, AdaBoostClassifier
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier

def escape_latex(string, printing=False):
    string = str(string)
    string = string.replace('_', '\_')
    string = string.replace('|', '$\\text{\\textbar}$')
    
    
    if printing:
        print(string)
    return string

def run_calssifier(m, X_train, X_test, y_train, y_test, unique_labels, gene_names, n_genes=100, roc_auc=True):
    m.fit(X_train, y_train)

    scores, cm = evaluate_model(m, X_test, y_test, unique_labels, plot=False, roc_auc=roc_auc)

    gene_importance = feature_importances(m, gene_names, n_genes=n_genes)
    
    results = {'scores': scores, 'confusion_matrix': cm, 'gene_importance': gene_importance}

    return results

def get_data(data_file, label_file, subset_label, predict_label, split_label, subset_query, genes_to_select=None):
    # Open file pointers
    data_h5 = h5py.File(data_file)
    label_set = pd.read_csv(label_file, index_col=0, header=0)
    
    # Select genes and set row size variable
    rows = data_h5['gene_id'].shape[0]
    gene_names = data_h5['gene_id'][:].astype('str')

    # Account for All query
    if subset_query == 'All':
        subset_query = np.unique(label_set[subset_label])

    # Select the subset
    subset_keys = []
    for subset in subset_query:
        subset_selection = label_set[subset_label] == subset
        notnull_selection = label_set[predict_label].notnull()
        subset_keys += list(label_set[notnull_selection & subset_selection].index.values)
    subset_labels = label_set.loc[subset_keys]

    

    # Get all labels and set related variables
    sample_to_label = subset_labels[predict_label]
    unique_labels = np.unique(sample_to_label).astype(str)

    
    dataset = np.empty([len(subset_keys), rows])
    for idx, k in enumerate(subset_keys):
        dataset[idx][:] = np.asarray(data_h5[k][:])

    if genes_to_select is not None:
        gene_index = pd.Series(range(len(gene_names)), index=gene_names)
        indexes = [gene_index[x] for x in genes_to_select]
        dataset = np.take(dataset, indexes, axis=1)
        gene_names = np.array(genes_to_select)

    return dataset, gene_names, sample_to_label, subset_labels, unique_labels

def split_data(data_file, subset_labels, split_label, predict_label, tt_split):
    data_h5 = h5py.File(data_file)
    rows = data_h5['gene_id'].shape[0]
    
    # Split the data to test and train on split_label level
    train_keys = []
    test_keys = []
    split_labels = np.unique(subset_labels[split_label])
    for split in split_labels:
        split_keys = list(subset_labels[subset_labels[split_label] == split].index.values)
        train_k, test_k = train_test_split(split_keys, test_size=tt_split)
        train_keys += train_k
        test_keys += test_k
    
    y_train = subset_labels.loc[train_keys].as_matrix(columns=[predict_label]).flatten().astype(str)
    y_test = subset_labels.loc[test_keys].as_matrix(columns=[predict_label]).flatten().astype(str)

    X_train = np.empty([len(train_keys), rows])
    for idx, k in enumerate(train_keys):
        X_train[idx][:] = np.asarray(data_h5[k][:])

    X_test = np.empty([len(test_keys), rows])
    for idx, k in enumerate(test_keys):
        X_test[idx][:] = np.asarray(data_h5[k][:])
        
    return X_train, X_test, y_train, y_test
    
def feature_importances(model, gene_names, n_genes=100):
    imp = None
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.average(model.coef_, axis=0)

    imp, names, idx = zip(*sorted(zip(imp, gene_names, range(len(gene_names)))))

    if imp[0] < 0:
        ng = int(n_genes/2)

        down_imp = imp[:ng]
        down_names = names[:ng]
        down_idx = idx[:ng]

        up_imp = imp[-ng:]
        up_names = names[-ng:]
        up_idx = idx[-ng:]

        output = [list(down_imp + up_imp), list(down_names + up_names), list(down_idx + up_idx)]
    else:
        ng = n_genes

        up_imp = imp[-ng:]
        up_names = names[-ng:]
        up_idx = idx[-ng:]

        output = [list(up_imp), list(up_names), list(up_idx)]

    #drop importance levels 0
    for idx, i in enumerate(output[0]):
        if i == 0:
            for t_idx, l in enumerate(output):
                del output[t_idx][idx]

    # [importances, names, indexes]
    return output

def rank_result_on_importance(class_genes):
    class_genes = (pd.concat(class_genes))
    class_genes = class_genes.groupby([class_genes.index, class_genes.gene_id]).mean()
    class_genes['sort'] = class_genes.importance.abs()
    class_genes = class_genes.sort_values(by=['sort']).drop('sort', axis=1)
    class_genes['rank'] = np.arange(class_genes.shape[0], 0, -1).astype('int')
    
    return class_genes

def get_intersection(total_genes, latex=False, printing=True):
    gene_list = []
    if printing:
        if latex:
            print('\\textbf{Gene analysis:}\\\\\nThe models selected have the following number of genes:\\\\')
    for key in total_genes.keys():
        genes = total_genes[key]
        gene_list.append(genes.index)
        if printing:
            if latex:
                key = escape_latex(key)
                print('{}: {}\\\\'.format(key, len(genes)))
            else:
                print('The {} Classifier found {} relevant genes over all folds'.format(key, len(genes)))
        
    intersection = reduce(np.intersect1d, gene_list)
    if printing:
            if latex:
                print('Intersection: {}\\\\'.format(len(intersection)))
            else:
                print('\nThe total intersection has {} elements'.format(len(intersection)))
    
    return intersection
    
def get_rank_per_calssifier(total_genes, intersection):
    intersection_rank = {}
    for key in total_genes.keys():
        ranks = []
        for i in intersection:
            ranks.append([i[1], i[0], int(total_genes[key].loc[i]['rank']), total_genes[key].loc[i]['importance']])
        intersection_rank[key] = sorted(ranks, key=lambda x:x[2])
        for idx, j in enumerate(intersection_rank[key]):
            intersection_rank[key][idx].append(len(intersection) - idx)
        
    return intersection_rank
        
def combine_ranks(intersection_rank):
    for key in intersection_rank.keys():
        intersection_rank[key] = sorted(intersection_rank[key])
        
    gene_rank = []
    for i in range(len(list(intersection_rank.values())[0])):
        rank = 0
        ranks = []
        for k in intersection_rank.keys():
            rank += intersection_rank[k][i][-1]
            ranks.append(intersection_rank[k][i][-1])
            gene = intersection_rank[k][i][0]
            index = intersection_rank[k][i][1]
        gene_rank.append([rank, gene, index, ranks])

    return sorted(gene_rank, reverse=True)
    