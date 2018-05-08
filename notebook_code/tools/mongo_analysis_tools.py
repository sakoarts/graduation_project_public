import sys, os
import numpy as np
import pandas as pd
import base64
from datetime import datetime, timedelta
import h5py

import pprint
pp = pprint.PrettyPrinter(indent=4)

from matplotlib import pyplot as plt
import seaborn as sns

from plotly.graph_objs import *
import plotly.figure_factory as FF
from plotly import tools
import plotly
import plotly.plotly as py
py.sign_in('user', 'key')

sys.path.append('../')
from tools.mongo_tools import run
from tools.compare_genes_across_classifiers_utils import feature_importances, get_intersection, get_rank_per_calssifier, combine_ranks, rank_result_on_importance
from tools.plots import evaluate_model, pathway_enrichment, pathway_df_to_table, plot_confusion_matrix

def collect_run_output(start_id, end_id, experiment_name=None, skip_ids=[], verbose=10):
    output = []
    names = []
    for i in range(start_id, end_id+1):
        if any(i == x for x in skip_ids):
            print('Run {} skipped'.format(i))
            continue
        try:
            run_dict = run(i)
        except IndexError as e:
            if verbose >= 5:
                print('No experiment exists with id {}'.format(i))
                print('Skipping next ids...')
                break
        try:
            result = run_dict['result']
        except KeyError as e:
            result = None
        if result is None:
            if verbose >= 10:
                print('Run {} did not finish'.format(i))
            continue
        
        name = run_dict['experiment']['name']
        if experiment_name is not None and experiment_name != name:
            if verbose >= 5:
                print('Run {} skipped since experiment name does not match set parameter: {}'.format(i, name))
            continue
        names.append(name)

        output.append(run_dict)
        
    uni_names = np.unique(names, return_counts=True)
    if len(uni_names[0]) == 1:
        if verbose >= 1:
            print('All these experiments belong to experiment: {}'.format(uni_names[0][0]))
    else:
        print('WARNING: there are multiple experiments present in given range')
        print('It is advised to set the experiment_name parameter to filter just one')
        print(uni_names)
    
    return output

def runs_per_classifier(output):
    class_id = [[x['config']['classifier_name'], x['_id']] for x in output]
    class_id = pd.DataFrame(class_id, columns=['name', 'id'])
    
    uni_class = class_id.groupby('name')['id'].apply(list)
    uni_class2 = pd.Series([len(x) for x in uni_class], index=uni_class.index, name='#runs')
    uni_class = uni_class.to_frame().join(uni_class2.to_frame())
    return uni_class

def compute_averages(output, plot=False, add_labels=False, skip_keys_add=None, decimal_tab=2, decimal_val=-1, annot=True, cm_normalize=False, res_list=False, table=False):
    if res_list:
        results = output
    else:
        results = [x['result'] for x in output]
        
    skip_keys = ['gene_importance', 'gene_names', 'labels', 'test_pc_shape', 'train_pc_shape', 'test_score_samples', 'train_score_samples']
    if skip_keys_add is not None:
        skip_keys += skip_keys_add
    average_runs = {}

    for i, key in enumerate(results[0].keys()):
        if key in skip_keys:
            if key == 'labels' and add_labels:
                average_runs[key] = results[0][key]['values']
            continue
        stat = []
        for idx, result in enumerate(results):
            try:
                res = result[key]
                if type(res) == dict:
                    res = res['values']
                stat.append(np.asarray(res))
            except Exception as e:
                pass
        avr = np.average(stat, axis=0)
        if decimal_val != -1:
            avr = float('%.{}f'.format(decimal_val)%(avr))
        if len(np.shape(stat)) == 1:
            var = np.var(stat, axis=0)
            if decimal_val != -1:
                var = float('%.{}f'.format(decimal_val)%(var))
            if table:
                average_runs['{} (var)'.format(key)] = '{} ({})'.format(avr, var)
            else:
                average_runs[key] = avr
                average_runs['{}_variance'.format(key)] = var
        else:
            average_runs[key] = avr
    
    if plot:
        labels = results[0]['labels']['values']
        plot_confusion_matrix(average_runs['confusion_matrix'], labels, large_font=False, decimal=decimal_tab, annot=annot, normalize=cm_normalize)
        average_runs.pop('confusion_matrix')

    return average_runs

def compute_averages_per_classifier(output, plot=False, skip_keys_add=None, decimal_val=-1, table=False, printing=False):
    out_per_classifier = {}
    for out in output:
        classifier = out['config']['classifier_name']
        if classifier not in out_per_classifier:
            out_per_classifier[classifier] = []
        out_per_classifier[classifier].append(out)
    
    avg_per_classifier = {}
    for classifier, out in out_per_classifier.items():
        if printing:
            print('Averages for {} algorithm'.format(classifier))
        avg = compute_averages(out, plot=plot, skip_keys_add=skip_keys_add, decimal_val=decimal_val, table=table)
        avg_per_classifier[classifier] = avg
        
    return avg_per_classifier

def compute_gene_importance(output, plot=False, similarity=None, latex=False):
    importance_per_classifier = {}
    for out in output:
        classifier = out['config']['classifier_name']
        if classifier not in importance_per_classifier:
            importance_per_classifier[classifier] = []
        importance = out['result']['gene_importance']
        for idx, value in enumerate(importance[1]):
            if type(value) is not str:
                value = decode_gene_name(value)
                importance[1][idx] = value
        importance_df = pd.DataFrame({'gene_id': importance[1], 'importance': importance[0]}, index=importance[2])
        importance_per_classifier[classifier].append(importance_df)
    
    total_genes = {}
    similarities = {}
    for key in importance_per_classifier:
        class_rank = rank_result_on_importance(importance_per_classifier[key])
        total_genes[key] = class_rank
        if similarity is not None:
            similarities[key] = []
            prev = rank_result_on_importance(importance_per_classifier[key][:-(similarity)])
            for i in reversed(range(similarity)):
                if i != 0:
                    cur = rank_result_on_importance(importance_per_classifier[key][:-(i)])
                else:
                    cur = class_rank
                similarities[key].append(compare_ids(cur, prev))
                prev = cur
            
    intersection = get_intersection(total_genes, latex=latex)
    intersection_rank = get_rank_per_calssifier(total_genes, intersection)
    combined_ranking = combine_ranks(intersection_rank)

    selected_genes = [x[1] for x in combined_ranking]
    if similarity is not None:
        return selected_genes, similarities
    return selected_genes

def decode_gene_name(endoded):
    decoded = base64.b64decode(endoded['py/reduce'][1]['py/tuple'][-1]['py/b64']).replace(b'\x00', b'').decode()
    return decoded

def compute_pathway_enrichment(selected_genes, top_pathways=50, table=False):
    pathway_df = pathway_enrichment(selected_genes)
    if pathway_df is not None:
        pathway_df = pathway_df[:top_pathways]
        print('collected the {} most enriched pathways'.format(top_pathways))
        if table:
            pathway_df = pathway_df_to_table(pathway_df)
        return pathway_df
    
def time_per_classifier(output):
    time_per_classifier = {}
    for out in output:
        if 'stop_time' not in out:
            print('Run {} does not have a stop time'.format(out['_id']))
            continue
        classifier_name = out['config']['classifier_name']
        if classifier_name not in time_per_classifier:
            time_per_classifier[classifier_name] = []
        time = out['stop_time'] - out['start_time']
        time_per_classifier[classifier_name].append(time)
    
    avg_time_per_classifier = {}
    for classifier, times in time_per_classifier.items():
        avg = sum(times, timedelta()) / len(times)
        print('Algorithm {} on average took {} to run'.format(classifier, avg))
        avg_time_per_classifier[classifier] = avg
    
    return avg_time_per_classifier

def get_best_scores(avg_per_class, avoid_strings=['variance', 'test']):
    scores = {}
    for algo, values in avg_per_class.items():
        if scores == {}:
            for key in values.keys():
                if  any(x in key for x in avoid_strings):
                    continue
                scores[key] = []
        
        for key in scores.keys():
            scores[key].append((values[key], algo))
            
    for score_name, values in scores.items():
        max_val = max(values)
        print('{} top value is {} from {} algorithm'.format(score_name, max_val[0], max_val[1]))
        
def get_dendro_data(predict_label, selected_genes_ids, subset=None, subset_label=None, top=50, data_path='../data/TCGA/exp_TCGA_coded_normalized.hdf5', label_path='../data/TCGA/exp_TCGA_coded_labels_add.csv'):
    if subset is None:
        label_set = pd.read_csv(label_path, index_col=0, header=0)
        sample_to_label = label_set[predict_label].dropna()
    else:
        df = pd.read_csv(label_path, index_col=0, header=0)
        if type(subset) is list:
            sub = df.loc[df[subset_label].isin(subset)]
        else:
            sub = df.loc[df[subset_label] == subset]
        sample_to_label = sub[predict_label].dropna()
    
    selected_genes_ids = selected_genes_ids[:top]
    
    
    data_h5 = h5py.File(data_path)
    
    relevant_keys = list(sample_to_label.index.values)
    gene_ids = data_h5['gene_id'][:].astype(str)
    gene_indexes = np.where(np.isin(gene_ids, selected_genes_ids))[0]

    selected_genes_data = np.zeros((len(relevant_keys), len(gene_indexes)))
    for idx, key in enumerate(relevant_keys):
        sample_data = data_h5[key][list(gene_indexes)]
        selected_genes_data[idx][:] = sample_data
    selected_genes_data = np.transpose(selected_genes_data)
    
    return selected_genes_data, selected_genes_ids, sample_to_label

def cluster_plot(selected_genes, sample_to_cluster, gene_names, nb_genes_used=None, color_scale='Jet', flip=False, title='Gene relevance', z_score=False):
    if nb_genes_used is not None:
        selected_genes = selected_genes[:nb_genes_used]
        gene_names = gene_names[:nb_genes_used]
        for key, value in importances.items():
            importances[key] = value[:nb_genes_used]
    
    # Z-score normalization
    if z_score:
        selected_genes = z_score_normalization(selected_genes)
        
    sample_names = sample_to_cluster.index.values
    sample_labels = sample_to_cluster['labels'].values
    sample_cluster = sample_to_cluster['cluster_labels'].values
    if flip:
        selected_genes = np.flipud(selected_genes)
        sample_names = np.flipud(sample_names)
        for i in importances.keys():
            importances[i] = np.flipud(importances[i])
        gene_names = np.flipud(gene_names)
        
    gene_names = [x.split('|')[0] for x in gene_names]
    hmap = Heatmap(
                z = selected_genes,
                x = sample_names,
                y = gene_names,
                colorscale = color_scale,
                showscale = False,
            )
    bars = []
        
    uni_labels = list(np.unique(sample_labels))
    val_label = pd.Series(range(len(uni_labels)), index=uni_labels)
    val_set = [val_label[x] for x in sample_cluster]
    n_genes = len(selected_genes)
    fragment = Heatmap(
        z=[val_set],
        showscale = False,
        colorscale = 'Jet',
    )
    #fragment[0]['x'] = hmap['layout']['xaxis']['tickvals']

    fig = tools.make_subplots(rows=2, cols=1, print_grid=False)
    fig.append_trace(hmap, 1, 1)
        
    fig.append_trace(fragment, 2, 1)

    fig['layout'].update({'yaxis': {'domain' : [0.05, 1],}})
    fig['layout'].update({'xaxis': {'mirror': False,
                                           'showgrid': False,
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""}})
    fig['layout'].update({'xaxis2': {'mirror': False,
                                           'showgrid': False,
                                           'domain' : [0, 1],
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""}})
    fig['layout'].update({'yaxis2': {'mirror': False,
                                           'showgrid': False,
                                           'domain' : [0, 0.05],
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""}})
    fig['layout'].update(title=title)

    return fig

def order_clusters(sample_to_label, selected_genes_data, run_id, output):
    used_run = None
    for out in output:
        if out['_id'] == run_id:
            used_run = out
            break
    else:
        print('Specified id {} not found in this output'.format(run_id))
    
    key_order = used_run['result']['train_keys']
    cluster_labels = used_run['result']['cluster_labels']['values']
    sample_to_label = pd.DataFrame(list(zip(sample_to_label.values, range(len(sample_to_label)))), index=sample_to_label.index, columns=['labels', 'index'])
    sample_to_label_reorderd = sample_to_label.reindex(key_order)
    sample_index = sample_to_label_reorderd['index']
    key_labels = sample_to_label_reorderd['labels']
    
    key_to_cluster = pd.DataFrame({'cluster_labels': cluster_labels, 'labels': key_labels, 'index': sample_index}, index=key_order)
    key_to_cluster = key_to_cluster.sort_values('cluster_labels')
    
    input_df = pd.DataFrame(selected_genes_data, columns=sample_to_label.index)
    selected_genes_data_reorderd = input_df[key_to_cluster.index.values].as_matrix()
    
    return selected_genes_data_reorderd, key_to_cluster

def get_heatmap_data(predict_label, selected_genes_ids, data_path='../data/TCGA/exp_TCGA_coded_normalized.hdf5', label_path='../data/TCGA/exp_TCGA_coded_labels_add.csv'):    
    label_set = pd.read_csv(label_path, index_col=0, header=0)
    sample_to_label = label_set[predict_label]
    
    data_h5 = h5py.File(data_path)
    relevant_keys = list(data_h5.keys())
    relevant_keys.remove('gene_id')
    gene_ids = data_h5['gene_id'][:].astype(str)
    gene_indexes = np.where(np.isin(gene_ids, selected_genes_ids))[0]
    
    selected_genes_data = np.zeros((len(relevant_keys), len(gene_indexes)))
    for idx, key in enumerate(relevant_keys):
        sample_data = data_h5[key][list(gene_indexes)]
        selected_genes_data[idx][:] = sample_data
    selected_genes_data = np.transpose(selected_genes_data)
    
    return selected_genes_data, selected_genes_ids, sample_to_label

def get_results_config(output, config, printing=False, drop_labels=[]):
    pca_comps = {}
    for idx, out in enumerate(output):
        if out['status'] != 'COMPLETED':
            print(out['status'])
            print('Experiment {} skipped since it failed'.format(out['_id']))
            continue
        elif 'result' not in out:
            print('Experiment {} skipped since no results are present'.format(out['_id']))
            continue
        result = out['result']
        if 'acc' in result:
            if np.isnan(result['acc']):
                #print('lossfound {}'.format(idx))
                continue
        for l in drop_labels:
            result.pop(l, None)
        comps = str(out['config'][config])

        if printing:
            pp_metric_dic(result)
            print(comps)
            print('\n')

        if comps in pca_comps:
            pca_comps[comps].append(result)
        else:
            pca_comps[comps] = [result]
            
    return pca_comps

def pp_metric_dic(result):
    for metric, value in result.items():
        print('{}: {}'.format(metric, value))
        
def print_avg_config(output, config):
    pca_comps = get_results_config(output, config)
    for key, value in pca_comps.items():
        print('{}: {}'.format(config, key))
        print('support: {}'.format(len(value)))
        pp_metric_dic(compute_averages(value, res_list=True))
        print('\n')
        
def make_latex_table(dictionary, table=True, skip_keys_add=None):
    head = 'index'
    res_table = []
    first = True
    for key, value in dictionary.items():
        folds = len(value)
        value = compute_averages(value, res_list=True, table=table, decimal_val=4, skip_keys_add=skip_keys_add)
        value['folds'] = folds
        line = str(key)
        for k, v in value.items():
            if first:
                #if 'variance'
                head = head + ' &    {}'.format(k)
            line = line + ' &    {}'.format(v)
        if first:
            head = head + ' \\\\'
            first = False
        line = line + ' \\\\'
        res_table.append(line)
        
    res_table.insert(0, head)
    return res_table

def print_latex_table(table, caption='My caption', label='my-label', sort_int=False):
    head = table[0]
    n_cols = len(head.split('&'))
    begin_arg = 'l'*n_cols
    print('\n\\begin{table}[]\n\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\begin{tabular}{' + begin_arg + '}')
    print_escape_latex(head)
    del table[0]
    if sort_int:
        dic = []
        for idx, t in enumerate(table):
            try:
                dic.append((int(t.split(' ')[0]), idx))
            except Exception as e:
                print(e)
                break
        dic = sorted(dic)
        for d in dic:
            print_escape_latex(table[d[1]])
    else:
        for t in table:
            print_escape_latex(t)
        
    print('\\end{tabular}\n\\end{table}')
    
def print_escape_latex(string):
    string = string.replace('_', '\_')
    
    print(string)

def escape_latex(string, printing=False):
    string = str(string)
    string = string.replace('_', '\_')
    string = string.replace('|', '$\\text{\\textbar}$')
    
    
    if printing:
        print(string)
    return string
    
def print_latex_config(output):
    out = output[0]
    print('\\textbf{Framework configuration:}\\\\\nData subset:' + '\\\\')
    predict_label = escape_latex(out['config']['predict_label'])
    print('Predict label: ' + predict_label + '\\\\')
    split_label = escape_latex(out['config']['split_label'])
    print('Split label: ' + split_label + '\\\\')
    tt_split = escape_latex(out['config']['tt_split'])
    print('Train test split: ' + tt_split + '\\\\')
    classifiers = 'LinearSVC and RandomForestClassifier'
    print('Classifiers: ' + classifiers + '\\\\')
    print('Stop running: similarity $> 0.8$, for 10 consecutive runs' + '\\\\')
    n_genes = out['config']['n_genes']
    print('Number of genes selected per algorithm: ' + str(n_genes) + '\\\\')
    
def print_latex_prediction_metrics(output):
    classifier_name = output[0]['config']['classifier_name']
    regression = classifier_name == 'RandomForestRegressor' or classifier_name == 'LinearSVR'
    skipkeys = ['confusion_matrix', 'recall', 'support', 'precision', 'fscore']
    avgs = compute_averages(output, plot=False, decimal_val=4, table=True, skip_keys_add=skipkeys)
    avgC = compute_averages_per_classifier(output, plot=False, decimal_val=4, table=True, skip_keys_add=skipkeys)
    print('\\textbf{Prediction metrics:}' + '\\\\')
    if regression:
        avg_acc = avgs['mean_absolute_error (var)']
        print('Average MAE: ' + avg_acc + '\\\\')
        avg_cohen = avgs['mean_squared_error (var)']
        print('Average MSE: ' + avg_cohen + '\\\\')
        print('Average MSLE: ' + avgs['mean_squared_log_error (var)'] + '\\\\')
        print('Average MdAE: ' + avgs['median_absolute_error (var)'] + '\\\\')
        print('Average $r^2$: ' + avgs['r2_score (var)'] + '\\\\')
        
        lsvm_acc = avgC['LinearSVR']['mean_squared_error (var)']
        print('Average MSE LSVM: ' + lsvm_acc + '\\\\')
        print('Average MdAE LSVM: ' + avgC['LinearSVR']['median_absolute_error (var)'] + '\\\\')
        print('Average $r^2$ LSVM: ' + avgC['LinearSVR']['r2_score (var)'] + '\\\\')
        rf_acc = avgC['RandomForestRegressor']['mean_squared_error (var)']
        print('Average MSE RandomForestRegressor: ' + rf_acc + '\\\\')
        print('Average MdAE RandomForestRegressor: ' + avgC['RandomForestRegressor']['median_absolute_error (var)'] + '\\\\')
        print('Average $r^2$ RandomForestRegressor: ' + avgC['RandomForestRegressor']['r2_score (var)'] + '\\\\')
    else:
        avg_acc = avgs['accuracy (var)']
        print('Average accuracy: ' + avg_acc + '\\\\')
        avg_cohen = avgs['cohen_kappa (var)']
        print('Average Cohen kappa: ' + avg_cohen + '\\\\')
        lsvm_acc = avgC['LinearSVC']['accuracy (var)']
        print('Average accuracy LSVM: ' + lsvm_acc + '\\\\')
        rf_acc = avgC['RandomForestClassifier']['accuracy (var)']
        print('Average accuracy RandomForestClassifier: ' + rf_acc + '\\\\')
    
def save_confusion_matrix(output, notebook_name, path='../plots/figures/', decimal_tab=1, large_font=False, annot=False, cm_normalize=True, latex=True, caption='Normalized Confusion Matrix over the test set', label=None):
    postfix = notebook_name.split('_')[-1]
    if label is None:
        label = 'fig:cm-{}'.format(postfix)
    cr_label = 'tab:cr-{}'.format(postfix)
    avgs = compute_averages(output, plot=False, decimal_tab=decimal_tab, annot=annot, cm_normalize=cm_normalize)
    labels = output[0]['result']['labels']['values']
    cm = avgs['confusion_matrix']
    plt = plot_confusion_matrix(cm, labels, large_font=large_font, decimal=decimal_tab, annot=annot, normalize=cm_normalize, plot=False)
    cm_name = '{}_cm.png'.format(notebook_name)
    plt.savefig(os.path.join(path, cm_name)) 
    print('Saved cm image {}\n'.format(cm_name))
    
    if latex:
        print('The Confusion matrix can be found in figure \\ref{' + label + '}. The Classification Report can be found in table \\ref{' + cr_label + '}.\n')
        print('\\begin{figure}[H]\n\\centering')
        print('\\includegraphics[height=5cm]{' + cm_name + '}')
        print('\\caption{' + caption + '}\n\\label{' + label + '}\n\\end{figure}')
        
def avg_to_cr(avgs, output):
    labels = output[0]['result']['labels']['values']
    head = ['fscore', 'precision', 'recall', 'support']
    data = np.asarray([avgs[x] for x in head]).T
    df = pd.DataFrame(data, index=labels, columns=head)
    return df

def print_latex_cr(output):
    avgs = compute_averages(output, plot=False)
    cr_df = avg_to_cr(avgs, output)
    print('\\textbf{Classification Report:}\\\\')
    print(cr_df.to_latex())
    
def compare_ids(cur, prev, window=9, printing=False):
    if prev is None:
        return 0
    cur_ids = cur.reset_index()['gene_id'].values
    prev_ids = prev.reset_index()['gene_id'].values
    same = 0
    for idx, gene in enumerate(prev_ids):
        half = np.floor(window/2)
        for step in range(window):
            rel_idx = step - half
            try:
                if gene == cur_ids[int(idx + rel_idx)]:
                    same += 1/(np.abs(rel_idx) +1)
                    break
            except IndexError:
                print('no id {}'.format(idx + rel_idx))
                pass
    if printing:
        print('Same: {}'.format(same))
    return same/len(cur_ids)
    
def save_plotly(plot, name, path='../plots/figures/'):
    try:
        py.image.save_as(plot, filename=os.path.join(path, name))
        print('Succeeded in saving {}'.format(name))
    except Exception as e:
        print('Failed to save {} because of {}'.format(name, e))
        
def data_latex(label, subset=None, subset_label=None, label_path='../data/TCGA/exp_TCGA_coded_labels_add.csv', regression=False):
    if subset is None:
        labels = pd.read_csv(label_path)[label].dropna().values
    else:
        df = pd.read_csv(label_path)
        sub = df.loc[df[subset_label] == subset]
        labels = sub[label].dropna().values
    
    amount = labels.shape[0]
    if regression:
        print('For the label {} we have {} samples'.format(label, amount))
        sns.distplot(labels)
    else:
        labels, count = np.unique(labels, return_counts=True)
        n_labels = len(labels)
        print('For the label {} we have {} samples and {} different labels. The classes and their support are as follows:\\\\'.format(label, amount, n_labels))
        string = ''
        for i, l in enumerate(labels):
            c = count[i]
            l = escape_latex(l)
            string += '{}: {}, '.format(l, c)
        string = string[:-2]
        print(string)
    
def latex_genes(genes):
    length = len(genes)
    print('These are the top {} genes in order of importance:\\\\'.format(length))
    string = ''
    for g in genes:
        g = escape_latex(g)
        string = string + '{}, '.format(g)
    string = string[:-2] + '\\\\'
    print(string)
    
def pw_latex(genes):
    pw_df = compute_pathway_enrichment(genes)
    print('The top 50 most enriched pathways based on these genes are:\\\\')
    pw_df = pw_df[['name', 'p_value']]
    print(pw_df.to_latex())
    
def latex_heatmap(heatmap, histogram, notebook_name, path='../plots/figures/', caption='Dendro-heatmap for the top genes', label=None, save_heat=False):
    postfix = notebook_name.split('_')[-1]
    if label is None:
        label = 'fig:heat-{}'.format(postfix)
    pw_label = 'tab:pw-{}'.format(postfix)
    
    hist_name = '{}_hist.png'.format(notebook_name)
    heat_name = '{}_heat.png'.format(notebook_name)
    save_plotly(histogram, hist_name)
    if save_heat:
        save_plotly(heatmap, heat_name)
    print('The dendro-heatmap can be found in figure \\ref{' + label + '}. The top 50 most enriched pathways can be found in table \\ref{' + pw_label + '}.\n')
    print('\\begin{figure}[H]\n\\centering')
    print('\\includegraphics[height=5cm]{' + heat_name + '}')
    print('\\caption{' + caption + '}\n\\label{' + label + '}\n\\end{figure}')
          
          
          
          
          
          