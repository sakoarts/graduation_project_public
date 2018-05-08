import numpy as np
import pandas as pd
import os

from mygene import MyGeneInfo

from matplotlib import pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support, auc, \
    classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import hypergeom


def evaluate_model(m, X_test, y_test, labels=None, plot=False, roc_auc=True):
    y_pred = m.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    scores = precision_recall_fscore_support(y_test, y_pred, labels=labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    results = {'accuracy': accuracy, 'cohen_kappa': cohen_kappa, 'labels': labels, 'precision': scores[0],
               'recall': scores[1], 'fscore': scores[2],
               'support': scores[3], 'confusion_matrix': cm}

    if plot:
        plot_metrics(X_test, y_test, labels, y_pred)

    if roc_auc:
        results['ROC_AUC'] = calc_roc_auc(m, X_test, y_test, labels, plot=plot)

    return results


def plot_metrics(m, X_test, y_test, labels, y_pred=None):
    if y_pred is None:
        y_pred = m.predict(X_test)

    print(classification_report(y_test, y_pred, labels=labels))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels)

def plot_confusion_matrix(matrix, labels, large_font=True):
    cm = pd.DataFrame(matrix, columns=labels, index=labels).rename_axis('predicted label', axis='columns').rename_axis(
        'true label', axis='index')

    plt.ylabel('true label')
    plt.xlabel('predicted label')
    if large_font:
        sns.set(font_scale=2)
    ax = sns.heatmap(cm, annot=True, square=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    plt.show()


def calc_roc_auc(m, X_test, y_test, labels, plot=False):
    if hasattr(m, 'decision_function'):
        y_score = m.decision_function(X_test)
    else:
        y_score = m.predict_proba(X_test)[:, 1]

    lb = LabelBinarizer().fit(m.classes_)
    y_test_lb = lb.transform(y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if len(y_score.shape) < 2:
        y_score = np.expand_dims(y_score, 1)
    for idx, i in enumerate(labels):
        fpr[i], tpr[i], _ = roc_curve(y_test_lb[idx], y_score[idx])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_lb.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    if plot:
        lw = 2
        for key in roc_auc.keys():
            plt.plot(fpr[key], tpr[key], lw=lw, label=key + ' (auc=%0.2f)' % roc_auc[key])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc

def pathway_enrichment(gene_names, pipe_section=1, dbs=None, total_genes=20531, p_cutoff=0.05, cache_path='../data/cache/'):
    mg = MyGeneInfo()
    mg.set_caching(cache_db=os.path.join(cache_path, 'mygene_cache'), verbose=False)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    gene_ids = []
    for g in gene_names:
        gene_ids.append(g.split('|')[pipe_section])
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
        if not (p is np.nan or p != p):
            # print(p)
            path = dict(p)
            for key in path.keys():
                if dbs is not None and key not in dbs:
                    continue
                p_dict = path[key]
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
        if False:
            p_size = 0
            while p_size % 1000 == 0:
                pathway_info = mg.query('pathway.{}.id:{}'.format(p_row.db, p_row.id), size=1000, skip=p_size,
                                        as_dataframe=True, df_index=False, verbose=False)
                s = pathway_info.shape[0]
                if s == 0:
                    break
                p_size += s
        else:
            p_size = mg.query('pathway.{}.id:{}'.format(p_row.db, p_row.id), size=0, verbose=False)['total']
        pathway_size.append(p_size)

    p_df['sup'] = [len(x) for x in p_df.genes.as_matrix()]
    p_df['size'] = pathway_size

    p_p = []
    nb_slected_genes = len(gene_names)
    for idx, p_row in p_df.iterrows():
        p_p.append(hypergeom.sf(p_row['sup'] - 1, total_genes, p_row['size'], nb_slected_genes))
    p_df['p_value'] = p_p

    p_df = p_df[p_df['p_value'] <= p_cutoff]

    p_df['ratio'] = [x['sup'] / x['size'] for i, x in p_df.iterrows()]
    p_df = p_df.sort_values(by=['p_value']).reset_index(drop=True)

    return p_df

def pathway_df_to_table(p_df, drop_columns=['genes'], name_slice=30, id_slice=15):
    table_df = p_df.drop(drop_columns, axis=1)
    table_df['name'] = table_df['name'].apply(lambda x: (x[:name_slice - 3] + '...') if len(x) > name_slice else x)
    table_df['id'] = table_df['id'].apply(lambda x: (x[:id_slice - 3] + '...') if len(x) > id_slice else x)

    table = tabulate(table_df.as_matrix(), headers=table_df.columns.values, tablefmt='orgtbl')
    return table