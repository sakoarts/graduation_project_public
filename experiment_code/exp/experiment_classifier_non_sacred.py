import numpy as np
import pandas as pd
import h5py
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

sys.path.append('../')
from tools import feature_importances, get_intersection, get_rank_per_calssifier, combine_ranks, rank_result_on_importance
from tools.metrics_classification import evaluate_model, pathway_enrichment, pathway_df_to_table

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# global variables
dataset_name = 'TCGA'
datasubset_name = '_coded'
data_fname = 'exp_{}{}_normalized.hdf5'.format(dataset_name, datasubset_name)
data_path = os.path.join('../data/normalized/', data_fname)

label_path = '../data/{}/exp_{}{}_labels_add.csv'.format(dataset_name, dataset_name, datasubset_name)

genes_to_select = list(np.load('C:/TEMP/last_genes_intersection_10000.npy'))
print(genes_to_select)
tt_split = 0.25

# Diffrent labels present in dataset
label_options = ['tumor', 'project', 'primary_site', 'subtype_tumor', 'site_tumor', 'tumor_stage', 'bmi_category', 'tumor_stage_float']
#sample_id, case_id, project, primary_site, tumor, subtype_tumor, age_at_diagnosis, alcohol_history, bmi, cigarettes_per_day, days_to_birth, days_to_death, days_to_last_follow_up, demographic_created_datetime, demographic_id, demographic_submitter_id, demographic_updated_datetime, diagnoses_created_datetime, diagnoses_submitter_id, diagnoses_updated_datetime, diagnosis_id, ethnicity, exposure_id, exposures_submitter_id, exposures_updated_datetime, gender, height, morphology, primary_diagnosis, race, site_of_resection_or_biopsy, state, tissue_or_organ_of_origin, tumor_stage, vital_status, weight, year_of_birth, year_of_death, years_smoked

# The subset of the data you would like to involve in training
subset_query_options = ['All', ['Kidney']]
subset_query = subset_query_options[0]

# The label type of the subset chosen to train on
subset_label = label_options[3]

# The label that you would like to predict
predict_label = label_options[3]

# The label on which the tt_split will be based
split_label = label_options[3]


def data():
    # Select genes and set row size variable
    data_h5 = h5py.File(data_path)
    rows = data_h5['gene_id'].shape[0]
    gene_names = data_h5['gene_id'][:].astype('str')

    # Read labels CSV
    label_set = pd.read_csv(label_path, index_col=0, header=0)

    # Account for All query
    global subset_query
    if subset_query is 'All':
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
        #indexes = [gene_index[x] for x in genes_to_select]
        indexes = []
        for g in genes_to_select:
            try:
                ind = gene_index[g]
                indexes.append(ind)
            except Exception as e:
                print(e)
                print('Gene {} is skipped'.format(g))

        X_train = np.take(X_train, indexes, axis=1)
        X_test = np.take(X_test, indexes, axis=1)
        gene_names = np.array(genes_to_select)

    dataset = np.concatenate((X_train, X_test))

    return X_train, X_test, y_train, y_test, unique_labels, gene_names, sample_to_label, dataset


def model(model, OvsR=False):
    if OvsR:
        model = OneVsRestClassifier(model, n_jobs=-1)

    return model

def train(classifier, n_genes, roc_auc=False):
    X_train, X_test, y_train, y_test, unique_labels, gene_names, sample_to_label, dataset = data()

    m = model(classifier, OvsR=False)

    m.fit(X_train, y_train)

    results = evaluate_model(m, X_test, y_test, unique_labels, plot=False, roc_auc=roc_auc)

    results['gene_importance'] = feature_importances(m, gene_names, n_genes=n_genes)

    return results

if __name__ == "__main__":
    classifiers = [
        RandomForestClassifier(n_estimators=100),
        LinearSVC(C=0.25),
    ]

    n_genes = 250
    n_folds = 5

    total_genes = {}
    for c in classifiers:
        class_genes = []
        class_acc = []
        classifier_name = c.__class__.__name__
        print('Running {} Classifier'.format(classifier_name))
        for n in range(n_folds):
            print('Fitting fold {}/{}'.format(n, n_folds))
            result = train(c, n_genes)

            class_genes.append(
                pd.DataFrame({'gene_id': result['gene_importance'][1], 'importance': result['gene_importance'][0]},
                             index=result['gene_importance'][2]))
            class_acc.append(result['accuracy'])
        print('Average accuracy: {}'.format(np.average(class_acc)))
        total_genes[classifier_name] = rank_result_on_importance(class_genes)


    intersection = get_intersection(total_genes)
    intersection_rank = get_rank_per_calssifier(total_genes, intersection)
    combined_ranking = combine_ranks(intersection_rank)
    top5 = combined_ranking[:5]
    for i, combi in enumerate(top5):
        print('{}. Gene {} with importance score {}'.format(i+1, combi[1], combi[-1][0]))

    selected_genes = [x[1] for x in combined_ranking]

    pathway_df = pathway_enrichment(selected_genes)
    if pathway_df is not None:
        print(pathway_df_to_table(pathway_df))

