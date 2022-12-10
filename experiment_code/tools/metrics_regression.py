import numpy as np
import pandas as pd
import os

from mygene import MyGeneInfo

from tabulate import tabulate

from sklearn.metrics import explained_variance_score, mean_absolute_error, \
    mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from scipy.stats import hypergeom


def evaluate_model(m, X_test, y_test, labels=None):
    y_pred = m.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    try:
        msle = mean_squared_log_error(y_test, y_pred)
    except Exception as e:
        msle = 0
        print('mean_squared_log_error  failed because of {}'.format(e))
    mdae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'explained_variance_score': evs,
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
        'labels': labels,
        'r2_score': r2,
        'median_absolute_error': mdae,
        'mean_squared_log_error': msle,
    }

def pathway_enrichment(gene_names, pipe_section=1, dbs=None, total_genes=20531, p_cutoff=0.05, cache_path='../data/cache/'):
    mg = MyGeneInfo()
    mg.set_caching(cache_db=os.path.join(cache_path, 'mygene_cache'), verbose=False)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    gene_ids = [g.split('|')[pipe_section] for g in gene_names]
    gene_info = mg.getgenes(geneids=gene_ids, fields='pathway', as_dataframe=True, df_index=False)
    try:
        pathways = gene_info['pathway']
    except Exception as e:
        print(e)
        print('No pathways found with the selected genes:')
        print(gene_names)
        return None
    p_df = []
    for idx, p in pathways.iteritems():
        if p is not np.nan and p == p:
            # print(p)
            path = dict(p)
            for key, p_dict in path.items():
                if dbs is not None and key not in dbs:
                    continue
                if type(p_dict) is list:
                    for k in p_dict:
                        p_df.append([k['id'], k['name'], key, str(gene_info['query'][idx])])
                else:
                    p_df.append([p_dict['id'], p_dict['name'], key, str(gene_info['query'][idx])])

    p_df = pd.DataFrame(p_df, columns=['id', 'name', 'db', 'genes'])
    p_df = p_df.groupby(['id', 'name', 'db'], as_index=False)['genes'].apply(list)
    p_df = p_df.reset_index()
    p_df.columns = ['id', 'name', 'db', 'genes']
    pathway_size = []
    for idx, p_row in p_df.iterrows():
        if idx % 50 == 0:
            print('querying {}/{}'.format(idx, p_df.shape[0]))
        p_size = mg.query('pathway.{}.id:{}'.format(p_row.db, p_row.id), size=0, verbose=False)['total']
        pathway_size.append(p_size)

    p_df['sup'] = [len(x) for x in p_df.genes.as_matrix()]
    p_df['size'] = pathway_size

    nb_slected_genes = len(gene_names)
    p_p = [
        hypergeom.sf(
            p_row['sup'] - 1, total_genes, p_row['size'], nb_slected_genes
        )
        for idx, p_row in p_df.iterrows()
    ]

    p_df['p_value'] = p_p

    p_df = p_df[p_df['p_value'] <= p_cutoff]

    p_df['ratio'] = [x['sup'] / x['size'] for i, x in p_df.iterrows()]
    p_df = p_df.sort_values(by=['p_value']).reset_index(drop=True)

    return p_df

def pathway_df_to_table(p_df, drop_columns=['genes'], name_slice=30, id_slice=15):
    table_df = p_df.drop(drop_columns, axis=1)
    table_df['name'] = table_df['name'].apply(lambda x: (x[:name_slice - 3] + '...') if len(x) > name_slice else x)
    table_df['id'] = table_df['id'].apply(lambda x: (x[:id_slice - 3] + '...') if len(x) > id_slice else x)

    return tabulate(
        table_df.as_matrix(),
        headers=table_df.columns.values,
        tablefmt='orgtbl',
    )


