"""
This script works on top of the file structure created by query_tcga.py and shoud be ran after that
Per project it takes all the raw sample files downloaded and combines them in one dataset file hdf5 and/or csv
It will also use the mygene (and biomart) api translate the ensemble id and determine the gene type
It will split the noncoding genes and coding genes based on that information
"""

import os, sys
import pandas as pd
from io import StringIO
from mygene import MyGeneInfo

sys.path.append('../')
from tools import to_hdf5_file
from etl.query_tcga import get_all_projects_by_filetype

# This dictionary controls which gene types end up in which data set, as of now the split is coding and noncoding
data_types = {
    'coding' : ['protein_coding'],
    'noncoding' : ['3prime_overlapping_ncRNA', 'IG_C_gene', 'IG_C_pseudogene',
       'IG_D_gene', 'IG_J_gene', 'IG_J_pseudogene', 'IG_V_gene',
       'IG_V_pseudogene', 'Mt_rRNA', 'Mt_tRNA', 'TEC', 'TR_C_gene',
       'TR_D_gene', 'TR_J_gene', 'TR_J_pseudogene', 'TR_V_gene',
       'TR_V_pseudogene', 'antisense_RNA', 'bidirectional_promoter_lncRNA',
       'lincRNA', 'macro_lncRNA', 'miRNA', 'misc_RNA', 'nan',
       'polymorphic_pseudogene', 'processed_pseudogene',
       'processed_transcript', 'pseudogene', 'rRNA',
       'ribozyme', 'sRNA', 'scRNA', 'scaRNA', 'sense_intronic',
       'sense_overlapping', 'snRNA', 'snoRNA',
       'transcribed_processed_pseudogene',
       'transcribed_unitary_pseudogene',
       'transcribed_unprocessed_pseudogene',
       'translated_processed_pseudogene', 'unitary_pseudogene',
       'unprocessed_pseudogene', 'vaultRNA']
}

# This function can be used to retreive data about genes from the biomart api, at the time this is not used
def biomart_get_gene_type(ensembl_ids, query_size=400):
    # import is done within function since it is only used here, and the function is not used at the time
    from biomart import BiomartServer

    if type(ensembl_ids) is not list:
        ensembl_ids = list(ensembl_ids)
    server = BiomartServer("http://useast.ensembl.org/biomart")
    server.verbose = False
    attributes = ['ensembl_gene_id', 'entrezgene', 'hgnc_symbol', 'gene_biotype']
    hsapiens_gene_ensembl  = server.datasets['hsapiens_gene_ensembl']
    len_ensembl_ids = len(ensembl_ids)
    print('Querying {} in slow biomart api to retreive correct gene type info'.format(len_ensembl_ids))

    query_status = 0
    total_gene_info = None
    while query_status % query_size == 0:
        cur_ensembl_ids = ensembl_ids[query_status : query_status + query_size]

        # Querying
        print('Querying {} ids with biomart API, status: {}/{}'.format(len(cur_ensembl_ids), query_status, len_ensembl_ids))
        try:
            response = hsapiens_gene_ensembl.search({
            'filters': {
                'ensembl_gene_id': cur_ensembl_ids
            },
            'attributes': attributes
            })
        except Exception as e:
            print('Error {} occurd, retrying...'.format(e))
            continue

        # parsing responce
        content = response.content.decode()
        content = StringIO(content)
        gene_info = pd.read_table(content, names=attributes)
        names = ['ensembl_id', 'entrez_id', 'gene_symbol', 'gene_type']
        gene_info.columns = names
        gene_info = gene_info.drop_duplicates(['ensembl_id'])
        gene_info = gene_info.set_index('ensembl_id')

        # adding missing Ensamble id's as empty rows
        gene_info = gene_info.reindex(cur_ensembl_ids)

        # append total gene info
        if total_gene_info is None:
            total_gene_info = gene_info
        else:
            total_gene_info = total_gene_info.append(gene_info)

        # set query status
        query_status = len(total_gene_info)

        # check for edge case where total ids is devisable by query size
        if query_status == len_ensembl_ids:
            break

    return total_gene_info


# parse Gene expression file to array
def read_exp(file):
    try:
        exp_file = pd.read_csv(file, sep='\t', names=['id', 'exp'])
    except Exception as e:
        print('Reading file {} failed due to error {}'.format(file, e))
        raise e
    return exp_file


# This funciton receives TCGA Ensamble ids, queries additional information about each gene from possibly multiple sources
# and returns the newly formated gene ids and the gene types
def ensemble_translator(ens_v, cache_path='../data/cache', get_gene_type_from_biomart_api=False):
    create_map = False
    map_path = os.path.join(cache_path, 'ens_id_type_mapping.csv')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        create_map = True
    else:
        if os.path.exists(map_path):
            map = pd.read_csv(map_path, index_col=0, header=0)
            if map.id.equals(ens_v):
                print('Local CSV cache is beeing used')
                return map['query'], map['type_of_gene']
        else:
            create_map = True

    ens = strip_after_dot(ens_v)

    mg = MyGeneInfo()
    mg.set_caching(cache_db=os.path.join(cache_path, 'mygene_cache'))

    gene_info = mg.getgenes(geneids=ens, fields='symbol,entrezgene,type_of_gene,ensembl.type_of_gene', as_dataframe=True, df_index=False)
    gene_info = gene_info.drop_duplicates('query').reset_index()
    # get type of gene string from ensembl dictionary
    gene_info['biomart_type_of_gene'] = gene_info['ensembl'].apply(lambda x: (type(x) is dict and x['type_of_gene']) or (type(x) is list and x[0]['type_of_gene']) or x)

    # get gene type
    if get_gene_type_from_biomart_api:
        query_ens_ids = gene_info['query'].values
        bio_frame = biomart_get_gene_type(query_ens_ids)
        bio_frame = bio_frame.reindex(query_ens_ids)
        gene_info['biomart_gene_type'] = bio_frame['gene_type'].values
        gene_type = gene_info['biomart_gene_type']
    else:
        #gene_type = gene_info['type_of_gene']
        gene_type = gene_info['biomart_type_of_gene']
    gene_type = gene_type.rename('type_of_gene')

    # build new gene_id
    gene_id = gene_info['query'].str.cat([strip_after_dot(gene_info.entrezgene.astype('str')), gene_info.symbol], sep='|', na_rep='?')

    if create_map:
        pd.DataFrame([ens_v, gene_id, gene_type]).T.to_csv(map_path)

    return gene_id, gene_type


# Helper function that strips the decimal form an Ensamble id, this decimal is used for versioning and is not compatible with the APIs
def strip_after_dot(ens):
    return ens.apply(lambda x: x.split('.')[0])


# This function walks over a path where raw TCGA data is stored, it parces the values and stores them in combined datasets
def walk_path_datatype(path, data_abv, csv=True, map_symbol_entrez_coding=True):
    dirs = os.listdir(path)
    exp_header = None
    gene_type = None
    exp_data = {}
    for idx, case in enumerate(dirs):
        case_path = os.path.join(path, case)
        if os.path.isfile(case_path):
            continue

        files = os.listdir(case_path)
        for file in files:
            fname_split = file.split('_')
            data_type = fname_split[0]
            if data_type != data_abv:
                continue
            fid = fname_split[1].split('.')[0]

            exp_file = os.path.join(case_path, file)
            exp_df = read_exp(exp_file)
            if exp_header is None:
                exp_header, gene_type = ensemble_translator(exp_df['id'])
                exp_header.name = 'gene_id'
            exp_data[fid] = exp_df.exp
        if idx % 50 == 0:
            print('Reading {}/{}'.format(idx, len(dirs) + 1))

    print('Creating dataframe')
    exp_data = pd.DataFrame.from_dict(exp_data)
    exp_data = exp_data.set_index(exp_header, drop=True)
    exp_data = exp_data.rename_axis('fid', axis='columns')

    # split on data type
    exp_data['gene_type'] = gene_type.values.astype(str)
    split_data = {}
    for key, types in data_types.items():
        split = exp_data[exp_data['gene_type'].isin(types)]
        split = split.drop('gene_type', axis=1)
        split_data[key] = split

    if map_symbol_entrez_coding:
        type = 'coding'
        coding_data = split_data[type]
        print('Mapping ids')
        ens_l, entrez_l, symbol_l = [], [], []
        for gene in coding_data.index.values:
            ens, entrez, symbol = gene.split('|')
            ens_l.append(ens)
            entrez_l.append(entrez)
            symbol_l.append(symbol)

        symbol_entrez = ['{}|{}'.format(sym, etz) for sym, etz in zip(symbol_l, entrez_l)]

        coding_data = coding_data.rename_axis('gene_id_old', axis='index')
        coding_data['gene_id'] = symbol_entrez
        coding_data = coding_data.groupby('gene_id').mean()
        split_data[type] = coding_data

    print('Writing files')
    for key, data in split_data.items():
        print('Type set: {}'.format(key))
        out_path = os.path.join(path, 'exp_data_{}'.format(key))
        if csv:
            data.to_csv(out_path + '.csv')
        to_hdf5_file(out_path + '.hdf5', data)


# This function orchistrates that each project path is processed by the functions above
def map_all_projects_datatype(data_abv, projects_path='../data/TCGA/', csv=True, map_symbol_entrez_coding=True, overwrite=False):
    data_projects = get_all_projects_by_filetype(data_abv)
    for idx, project in enumerate(data_projects):
        print('Creating datasets of {} project ({}/{})'.format(project, idx, len(data_projects)))
        project_path = os.path.join(projects_path, project)
        if not overwrite:
            if os.path.isfile(os.path.join(project_path, '{}_data.hdf5'.format(data_abv))):
                print('Skipping project {} hdf5 file for type {} already present'.format(project, data_abv))
                continue
        walk_path_datatype(project_path, data_abv, csv=csv, map_symbol_entrez_coding=map_symbol_entrez_coding)


# main function
if __name__ == "__main__":
    data_abv = 'exp'
    map_all_projects_datatype(data_abv, overwrite=True, csv=False)