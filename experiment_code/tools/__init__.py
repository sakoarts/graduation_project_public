import numpy as np
import pandas as pd

import tempfile
import pickle
from functools import reduce
import h5py

def pickle_object(obj):
    fp = tempfile.mktemp()
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)
    return fp

# HDF5 parsing function
def to_hdf5_file(file, A):
    gene_ids = A.index.values.astype('S')
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('gene_id', data=gene_ids)
        for idx, a in A.iteritems():
            case = a.name
            data = a.as_matrix().astype('float32')
            hf.create_dataset(case, data=data)

def read_hdf5_file(path):
    data = h5py.File(path)
    gene_ids = data['gene_id'][:].astype(str)
    samples = list(data.keys())
    samples.remove('gene_id')

    print('Loading {} samples from HDF5'.format(len(samples)))
    data_array = np.zeros([len(samples), len(gene_ids)])
    for idx, key in enumerate(samples):
        data_array[idx][:] = data[key][:]

    data_array = np.transpose(data_array)
    return pd.DataFrame(data_array, index=gene_ids, columns=samples)

def feature_importances(model, gene_names, n_genes=100):
    imp = None
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        if model.__class__.__name__ is 'LinearSVR':
            imp = model.coef_
        else:
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


def get_intersection(total_genes):
    gene_list = []
    for key in total_genes.keys():
        genes = total_genes[key]
        gene_list.append(genes.index)
        print('The {} Classifier found {} relevant genes over all folds'.format(key, len(genes)))

    intersection = reduce(np.intersect1d, gene_list)
    print('\nThe total intersection has {} elements'.format(len(intersection)))

    return intersection


def get_rank_per_calssifier(total_genes, intersection):
    intersection_rank = {}
    for key in total_genes.keys():
        ranks = [
            [
                i[1],
                i[0],
                int(total_genes[key].loc[i]['rank']),
                total_genes[key].loc[i]['importance'],
            ]
            for i in intersection
        ]

        intersection_rank[key] = sorted(ranks, key=lambda x: x[2])
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
        importances = []
        for k in intersection_rank.keys():
            rank += intersection_rank[k][i][-1]
            ranks.append(intersection_rank[k][i][-1])
            importances.append(intersection_rank[k][i][-2])
            gene = intersection_rank[k][i][0]
            index = intersection_rank[k][i][1]
        gene_rank.append([rank, gene, index, ranks, importances])

    return sorted(gene_rank, reverse=True)