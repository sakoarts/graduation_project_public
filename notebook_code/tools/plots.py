import os, sys
import numpy as np
import pandas as pd
import h5py

import colorlover as cl

from mygene import MyGeneInfo

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import seaborn as sns

from plotly.graph_objs import *
import plotly.figure_factory as FF
from plotly import tools

from tabulate import tabulate

from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support, auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import hypergeom
from scipy.stats import zscore

def evaluate_model(m, X_test, y_test, labels=None, plot=False, roc_auc=True):
    y_pred = m.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    scores = precision_recall_fscore_support(y_test, y_pred, labels=labels)

    scores = {'accuracy': accuracy, 'cohen_kappa': cohen_kappa, 'labels': labels, 'precision': scores[0], 'recall': scores[1], 'fscore': scores[2],
               'support': scores[3]}

    if plot:
        plot_metrics(X_test, y_test, labels, y_pred)
    
    if roc_auc:
        scores['ROC_AUC'] = calc_roc_auc(m, X_test, y_test, labels, plot=plot)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return scores, cm

def get_confusion_matrix(y_test, y_pred, labels):
    return confusion_matrix(y_test, y_pred, labels=labels)

def plot_metrics(m, X_test, y_test, labels, y_pred=None):
    if y_pred is None:
        y_pred = m.predict(X_test)

    print(classification_report(y_test, y_pred, labels=labels))

    cm = get_confusion_matrix(y_test, y_pred, labels)
    plot_confusion_matrix(cm, labels)
    
def plot_confusion_matrix(matrix, labels, large_font=True, decimal=2, annot=True, cmap='viridis', normalize=False, plot=True):
    cm = pd.DataFrame(matrix, columns=labels, index=labels).rename_axis('predicted label', axis='columns').rename_axis(
        'true label', axis='index')

    plt.ylabel('true label')
    plt.xlabel('predicted label')
    if large_font:
        sns.set(font_scale=2)
    plt.figure(figsize = (16,16))
    if normalize:
        cm=(cm-cm.mean())/cm.std()
    ax = sns.heatmap(cm, annot=annot, square=True, fmt='.{}f'.format(decimal), cmap='viridis')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    
    if plot:
        plt.show()
    else:
        return plt
    
def calc_roc_auc(m, X_test, y_test, labels, plot=False):
    if hasattr(m, 'decision_function'):
        y_score = m.decision_function(X_test)
    else:
        y_score = m.predict_proba(X_test)[:, 1]

    lb = LabelBinarizer().fit(m.classes_)
    y_test_lb = lb.transform(y_test)

    fpr = {}
    tpr = {}
    roc_auc = {}
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
        for key, value in roc_auc.items():
            plt.plot(fpr[key], tpr[key], lw=lw, label=key + ' (auc=%0.2f)' % value)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc


def get_dendro_heatmap(selected_genes, gene_labels, sample_to_label, nb_genes_used=None, color_option=0, z_score=False):
    if nb_genes_used is not None:
        selected_genes = selected_genes[:nb_genes_used]
        gene_labels = gene_labels[:nb_genes_used]

    selected_samples = np.transpose(selected_genes)
    sample_labels = sample_to_label.index.values
    subtype_labels = sample_to_label.values

    # Initialize Dendograms
    figure = FF.create_dendrogram(selected_samples, orientation='bottom', labels=sample_labels)
    for i in range(len(figure['data'])):
        figure['data'][i]['yaxis'] = 'y2'

    dendro_side = FF.create_dendrogram(selected_genes, orientation='right', labels=gene_labels)
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    figure['data'].extend(dendro_side['data'])
    figure['layout']['yaxis'] = dendro_side['layout']['yaxis']
    samples_dendro_leaves = list(figure['layout']['xaxis']['ticktext'])
    genes_dendro_leaves = list(dendro_side['layout']['yaxis']['ticktext'])

    # Set color scales
    c_map, c_map_legend, val_set, val_label = custom_color(sample_to_label, samples_dendro_leaves)
    color_options = ['Viridis', 'Jet', 'Spectral', c_map]
    color_scale = color_options[color_option]

    # Create Heatmap
    input_df = pd.DataFrame(selected_genes, index=gene_labels, columns=sample_to_label.index)
    column_order_df = input_df[samples_dendro_leaves]
    reorederd_df = column_order_df.reindex(genes_dendro_leaves)
    heat_data = reorederd_df.as_matrix()

    # Z-score normalization
    if z_score:
        heat_data = z_score_normalization(heat_data)

    heatmap = Data([
        Heatmap(
            z = heat_data,
            colorscale = color_scale,
            showscale = False,
        )
    ])
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']
    heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
    figure['data'].extend(Data(heatmap))

    # Create Fragmentation plot
    n_genes = len(gene_labels)
    fragmentation_bar = Data([Heatmap(
        z=[val_set],
        y=[-(n_genes/2), -(n_genes)],
        colorscale=c_map,
        showscale = False,
    )])

    fragmentation_bar[0]['x'] = figure['layout']['xaxis']['tickvals']
    figure['data'].extend(Data(fragmentation_bar))

    # Edit Layout
    figure['layout'].update({'width':900, 'height':900,
                         'showlegend':False, 'hovermode': 'closest', 
                         })

    # https://plot.ly/python/axes/
    # Edit xaxis
    figure['layout']['xaxis'].update({'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks':""})
    # Edit xaxis2
    figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}})

    # Edit yaxis
    figure['layout']['yaxis'].update({'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'side': 'right',
                                      'showticklabels': True,
                                      'ticks': ""})
    # Edit yaxis2
    figure['layout'].update({'yaxis2':{'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'side': 'right',
                                       'showticklabels': False,
                                       'ticks':""}})

    # Sample label fragmentation legend
    legends = [
        mpatches.Patch(color=c_map_legend[i][1], label=idx)
        for idx, i in val_label.iteritems()
    ]

    legend = plt.figure(figsize=(1, 0.25))
    ax = legend.add_subplot(111)  #create the axes 
    ax.set_axis_off()
    ax.legend(handles=legends, ncol=len(legends))
    plt.close(legend)

    return figure, legend, color_scale

def custom_color(sample_to_label, samples_dendro_leaves, color_scale='Spectral'):
    # Sample label fragmentation
    sample_to_label = sample_to_label.astype(str).fillna('unkown')
    label_leaves = [str(sample_to_label[x]) for x in samples_dendro_leaves]
    uni_labels = list(np.unique(sample_to_label))
    uni_len = len(uni_labels)
    uni_range = range(uni_len)
    uni_normal = [x/uni_range[-1] for x in uni_range]
    val_label = pd.Series(uni_range, index=uni_labels)
    val_set = [val_label[x] for x in label_leaves]

    
    colors = cl.scales[str(min(max(uni_len, 3), 11))]['div']['Spectral']
    if uni_len > len(colors):
        if uni_len > 20:
            colors = cl.to_rgb(cl.interp(colors, 20))
            for i in range(uni_len - 20):
                i = i % 20
                colors.append(colors[i])
        else:
            colors = cl.to_rgb(cl.interp(colors, uni_len))
    colors_legend = [tuple(map(lambda x:x/255, x)) for x in cl.to_numeric(colors)]
    c_map_legend = list(zip(uni_range, colors_legend))
    c_map = list(zip(uni_normal, colors))
    return c_map, c_map_legend, val_set, val_label

def get_label_bar(sample_to_label):
    labels = np.unique(sample_to_label)
    bar_traces = []
    for l in labels:
        type_label = sample_to_label[sample_to_label == l]
        bar_traces.append(Bar(y=['subtype'], x = [len(type_label)], name = l, orientation = 'h'))
    layout = Layout(
        barmode='stack'
    )
    return Figure(data=bar_traces, layout=layout)
                          
def get_expression_hist(selected_genes, color_scale='Viridis', nb_genes_used=-1, log=True, drop_zero=False, decimals=1, size=10, drop_axis=False, z_score=False):
    selected_genes = selected_genes[:nb_genes_used]
    
    # Z-score normalization
    if z_score:
        selected_genes = z_score_normalization(selected_genes)
        
    all_values = selected_genes.flatten()
    if drop_zero:
        all_values = np.delete(all_values, np.where(all_values == 0))
    rounded_values = np.around(all_values, decimals=decimals)
    count = np.unique(rounded_values, return_counts=True)

    trace1 = Scatter(
        x=count[0],
        y=count[1],
        mode='markers',
        marker=dict(
            size=str(size),
            color = count[0], #set color equal to a variable
            colorscale=color_scale,
            showscale=True
        )
    )
    data = [trace1]
    layout = Layout()
    
    hist_figure = Figure(data=data, layout=layout)
    
    if log:
        hist_figure['layout'].update({'yaxis': {'type': 'log',
                                         'autorange': True,}})
        
    if drop_axis:
        hist_figure['layout'].update({'xaxis': {'mirror': False,
                                         'showgrid': False,
                                         'showline': False,
                                         'zeroline': False,
                                         'showticklabels': False,
                                         'ticks':""}})
        hist_figure['layout'].update({'yaxis': {'mirror': False,
                                         'showgrid': False,}})
        
    return hist_figure
                          
def get_plot_data(dataset, combined_ranking, nb_genes_used=-1, z_score=False):
    combined_ranking = combined_ranking[:nb_genes_used]

    gene_labels, indexes = [], []
    for i in combined_ranking:
        gene_labels.append(i[1])
        indexes.append(i[2])

    selected_genes = np.transpose(dataset)[np.array(indexes)]
    
    # Z-score normalization
    if z_score:
        selected_genes = z_score_normalization(selected_genes)
    
    return selected_genes, gene_labels

def z_score_normalization(data):
    out = np.zeros(data.shape)
    for idx, dat in enumerate(data):
        out[idx][:] = zscore(dat)
    return out

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

def plot_importances(selected_genes, sample_to_label, gene_names, importances, nb_genes_used=-1, color_scale='Viridis', flip=False, title='Gene relevance', z_score=False):
    selected_genes = selected_genes[:nb_genes_used]
    sample_to_label = sample_to_label[:nb_genes_used]
    gene_names = gene_names[:nb_genes_used]
    for key, value in importances.items():
        importances[key] = value[:nb_genes_used]
    
    # Z-score normalization
    if z_score:
        selected_genes = z_score_normalization(selected_genes)
        
    sample_names = sample_to_label.index.values
    sample_labels = sample_to_label.as_matrix()
    if flip:
        selected_genes = np.flipud(selected_genes)
        sample_names = np.flipud(sample_names)
        for i in importances.keys():
            importances[i] = np.flipud(importances[i])
        gene_names = np.flipud(gene_names)
        
    
    hmap = Heatmap(
                z = selected_genes,
                x = sample_names,
                y = gene_names,
                colorscale = color_scale,
                showscale = False,
            )
    bars = []
    colors = ['rgba(246, 78, 139, X)', 'rgba(58, 71, 80, X)']
    for idx, key in enumerate(importances.keys()):
        imp = importances[key]
        bar = Bar(
                    x=imp,
                    y=gene_names,
                    orientation = 'h',
                    name = key, 
                    marker = dict(
                        color = colors[idx].replace('X', '0.6'),
                        line = dict(
                            color = colors[idx].replace('X', '1.0'),
                            width = 1)
                    ),
                )
        bars.append(bar)
        
    uni_labels = list(np.unique(sample_labels))
    val_label = pd.Series(range(len(uni_labels)), index=uni_labels)
    val_set = [val_label[x] for x in sample_labels]
    n_genes = len(selected_genes)
    fragment = Heatmap(
        z=[val_set],
        showscale = False,
    )
    #fragment[0]['x'] = hmap['layout']['xaxis']['tickvals']

    fig = tools.make_subplots(rows=2, cols=2, print_grid=False, specs=[[{}, {}], [{'colspan': 2}, None]])
    fig.append_trace(hmap, 1, 1)
    
    for bar in bars:
        fig.append_trace(bar, 1, 2)
        
    fig.append_trace(fragment, 2, 1)

    fig['layout'].update({'yaxis': {'domain' : [0, 0.95],}})
    fig['layout'].update({'yaxis2': {'mirror': False,
                                           'showgrid': False,
                                           'domain' : [0, 0.95],
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""}})
    fig['layout'].update({'xaxis': {'mirror': False,
                                           'showgrid': False,
                                           'domain' : [0, 0.65],

                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""}})
    fig['layout'].update({'xaxis2': {'domain' : [0.65, 1],}})
    fig['layout'].update({'xaxis3': {'mirror': False,
                                           'showgrid': False,
                                           'domain' : [0, 0.65],
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""}})
    fig['layout'].update({'yaxis3': {'mirror': False,
                                           'showgrid': False,
                                           'domain' : [0.95, 1],
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""}})
    fig['layout'].update(title=title, barmode='group')

    return fig

def imortance_plot_data(intersection_rank, combined_ranking):
    gene_ids = [x[1] for x in combined_ranking]
    importances = {}
    for key in intersection_rank:
        key_ranking = intersection_rank[key]
        imp = [x[3] for x in key_ranking]
        
        
        #normalize to 0 - 1 range
        imp_norm = (imp-np.min(imp))/np.ptp(imp)
        #keep negatives
        #imp_norm = [-1 * x if imp[idx] < 0 else x for idx, x in enumerate(imp_norm)]
        #imp_norm =  imp
        
        gene = [x[0] for x in key_ranking]
        gene_imp = pd.Series(imp_norm, index=gene)
        gene_imp = gene_imp.reindex(gene_ids)
        importances[key] = gene_imp.as_matrix()
        
    return gene_ids, importances

def pathway_df_to_table(p_df, drop_columns=['genes'], name_slice=30, id_slice=15):
    table_df = p_df.drop(drop_columns, axis=1)
    table_df['name'] = table_df['name'].apply(lambda x: (x[:name_slice - 3] + '...') if len(x) > name_slice else x)
    table_df['id'] = table_df['id'].apply(lambda x: (x[:id_slice - 3] + '...') if len(x) > id_slice else x)

    return tabulate(
        table_df.as_matrix(),
        headers=table_df.columns.values,
        tablefmt='orgtbl',
    )

def average_results(results, average_cm=True, roc_auc=True, average_keys=['accuracy', 'cohen_kappa', 'precision', 'recall', 'support', 'fscore']):
    if roc_auc:
        average_keys.append(['ROC_AUC', 'micro'])

    labels = None
    to_average = {}
    for key in average_keys:
        if type(key) == list:
            key = '_'.join(key)
        to_average[key] = []
    if average_cm:
        to_average['confusion_matrix'] = []

    print('Collecting averages')
    for c_name, c_results in results.items():
        for fold_result in c_results:
            scores = fold_result['scores']
            for key in average_keys:
                if type(key) == list:
                    score = scores
                    for s in key:
                        score = score[s]
                    key = '_'.join(key)
                else:
                    score = scores[key]
                to_average[key].append(score)
            if average_cm:
                to_average['confusion_matrix'].append(fold_result['confusion_matrix'])
            if labels is None:
                labels = scores['labels']

    average = {'labels': labels}
    print('Averaging')
    for key, values in to_average.items():
        average[key] = np.average(values, axis=0)
    return average

def up_down_regulated(dataset, sample_to_label, label_file, gene_names, predict_label, condisioned_label_name, normal_label=False, kursal=False, p_value=0.05, plot=False):
    gene_data = np.transpose(dataset)

    label_set = pd.read_csv(label_file, index_col=0, header=0)
    relevant_labels = label_set[[condisioned_label_name, predict_label]]
    normal_labels = relevant_labels[relevant_labels[condisioned_label_name] == normal_label]
    normal_labels = list(normal_labels[predict_label].unique())
    conditioned_labels = relevant_labels[relevant_labels[condisioned_label_name] != normal_label]
    conditioned_labels = list(conditioned_labels[predict_label].unique())

    index_sample_label = sample_to_label.reset_index()
    normal_index = index_sample_label[index_sample_label[predict_label].isin(normal_labels)].index.values
    tumorous_index = index_sample_label[index_sample_label[predict_label].isin(conditioned_labels)].index.values

    tumorous_samples = gene_data.take(tumorous_index, axis=1)
    normal_samples = gene_data.take(normal_index, axis=1)

    mean_exp_tumor = [x.mean() for x in tumorous_samples]
    mean_exp_normal = [x.mean() for x in normal_samples]

    regulation = np.subtract(mean_exp_tumor, mean_exp_normal)    

    gene_regulation = pd.Series(regulation, index=gene_names).sort_values()

    if kursal:
        print('Calculating Kurskal values')
        from scipy.stats import kruskal
        kw_samples = []
        for idx, x in enumerate(tumorous_samples):
            try:
                kw = kruskal(normal_samples[idx], x, nan_policy='propagate')
                #print(kw.pvalue)
            except Exception as e:
                #print('Error occured at gene {}: {}'.format(gene_names[idx], e))
                kw = np.nan
            kw_samples.append(kw)

        kw_p_samples = [x.pvalue if x is not np.nan and x.pvalue <= p_value else x for x in kw_samples]

        up_down_kw = []
        for idx, kw in enumerate(kw_p_samples):
            if kw is None:
                up_down_kw.append(kw)
            elif regulation[idx] >= 0:
                up_down_kw.append(True)
            else:
                up_down_kw.append(False)
            continue
        return pd.Series(up_down_kw, index=gene_names, name='up')

    if plot:
        zero = (regulation<0).sum()/len(regulation)
        sd = zero/2
        print(zero)
        zero = 0
        lowest = min(regulation)
        print(lowest)
        highest = max(regulation)

        regulation_heat = Data([Heatmap(
                z=[gene_regulation.values],
                x=gene_names,
                showscale = True,
                #colorscale = 'RdBu',
                #colorbar = [gene_regulation.values],
                colorscale = [[0, 'rgb(255,0,0)'], [0.5, 'rgb(0,0,0)'], [1, 'rgb(0,255,0)']]
            )])

        regulation_layout = Layout(
            yaxis = dict(showgrid=False, showline=False, showticklabels=False, ticks=''),
            height = 300,

        )

        regulation_plot = Figure(data=regulation_heat, layout=regulation_layout)
        iplot(regulation_plot)
        
def get_pathway_genes(p_id, pathway_df):
    p_row = pathway_df[pathway_df['id'] == p_id]
    ens_ids = p_row['genes'].values[0]
    return ensemble_to_symbol(ens_ids)
    
def ensemble_to_symbol(ens):
    mg = MyGeneInfo()

    gene_info = mg.getgenes(geneids=ens, fields='symbol', as_dataframe=True, df_index=False)
    gene_info = gene_info.drop_duplicates('query').reset_index()
    
    gene_symbol = gene_info['symbol'].values
    gene_id = gene_info.symbol.str.cat([gene_info['query']], sep='|', na_rep='?').values

    return gene_symbol, gene_id
