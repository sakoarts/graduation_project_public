"""
This script woks on top of the dataset files created by the b_process_tcga.py so should be ran after that one
The goal is to combine all the serperate data files label sets per project into two complete files
"""


import os, sys
import pandas as pd
import h5py

sys.path.append('../')
from etl.query_tcga import get_all_projects

# destination files
data_path = '../data/TCGA'
out_file = 'exp_TCGA_coded_raw.hdf5'
out_labels = 'exp_TCGA_coded_labels.csv'
# will make use of hdf5 ability to have nested datasets, not supported by any of the other scripts
nested_structure = False

# Source file names
data_file = 'exp_data_coding.hdf5'
label_file = 'downloaded_fids_exp.csv'
case_label_file = 'case_labels_exp.csv'

projects = get_all_projects('exp')
projects = [x for x in projects if type(x[-1]) is int]

gene_ids = None
sample_label_header = None
sample_labels = []
with h5py.File(os.path.join(data_path, out_file), 'w') as hf:
    for project_row in projects:
        project = project_row[0]
        primary_site = project_row[2][0].replace(' ', '')
        subtype = project.split('-')[1]

        print('Handeling project {}'.format(project))

        project_path = '../data/TCGA/{}'.format(project)
        data_file_path = os.path.join(project_path, data_file)
        if not os.path.isfile(data_file_path):
            print(f'No such file: {data_file_path}')
            continue

        label_df = pd.read_csv(os.path.join(project_path, label_file), header=0, index_col=2)

        case_labels = pd.read_csv(os.path.join(project_path, case_label_file), header=0, index_col=0)

        data_h5 = h5py.File(data_file_path)

        if gene_ids is None:
            gene_ids = data_h5['gene_id'][:].astype('S')
            hf.create_dataset('gene_id', data=gene_ids)
        if sample_label_header is None:
            sample_header = ['sample_id', 'case_id', 'project', 'primary_site', 'tumor', 'subtype_tumor']
            case_labels_header = list(case_labels.columns.values)
            sample_label_header = sample_header + case_labels_header

        for sample in data_h5.keys():
            if sample == 'gene_id':
                continue
            tumor = str(label_df['tumor'][sample])
            if tumor not in ['True', 'False']:
                print('Error on sample {} tumor indicator {} invalid, skipping...'.format(sample, tumor))
                continue

            sample_data = data_h5[sample][:]

            if nested_structure:
                dataset_name = '{}/{}/{}'.format(primary_site, project, sample)
            else:
                dataset_name = sample
            hf.create_dataset(dataset_name, data=sample_data)

            case_id = label_df['case_id'][sample]
            sample_label = [sample, case_id, project, primary_site, tumor, '{}_{}'.format(subtype, tumor)]
            sample_case_labels = list(case_labels.loc[case_id].values)
            sample_label += sample_case_labels

            cur_case_labels_header = list(case_labels.columns.values)
            if case_labels_header != cur_case_labels_header:
                print('The case label headers of this sample are not as expected and thus cannot be combined')
                print('For now sample {} from project {} is excluded, make sure to fix this'.format(sample, project))
                print('Expected header values: {}'.format(case_labels_header))
                print('Observed header values: {}'.format(cur_case_labels_header))
                continue

            sample_labels.append(sample_label)

sample_labels_df = pd.DataFrame(sample_labels, columns=sample_label_header)

# remove columns where all values are empty
nan_indexes = sample_labels_df.index[sample_labels_df.isnull().all(1)].values
sample_labels_df.drop(sample_labels_df.columns[nan_indexes], axis=1, inplace=True)

sample_labels_df.to_csv(os.path.join(data_path, out_labels), index=False)
