"""
This script is meant to interact with the TCGA API
Its main goal is downloading data files and reqesting sample labels
The result will be a file structure containing these files
Currently it is setup to download all expression data files form all projects within the TCGA database
The script is protected to not crash on failed download attempts, and it saves its progress/success state
The flexibility of the script should allow you to download subsets of the TCGA data, multiple file types, and files formats
"""

import os
import tarfile
import zipfile
import gzip

import requests
import urllib
import pandas as pd
import numpy as np

# The TCGA project identifier for which data needs to be retrieved, should be set with set_project_and_folder function
project_id = None
# The path of the where a project that is beeing processes should use, should be set with set_project_and_folder funciton
project_path = None
# The root path where the file stucture will begin
out_path = '../data/TCGA'
# The different options for expression file formats availeble, FPKM is advised
exp_file_types = ['.htseq.counts.', '.FPKM.', '.FPKM-UQ.']
exp_file_type = exp_file_types[1]

# Dictionary mapping a file type abreviation to its official file type in the TCGA database
abv_data_type = {
    'exp' : 'Gene Expression Quantification',
    'meth' : 'Methylation Beta Value',
}
# Dictionary mapping a file type abreviation to its official file category in the TCGA database
abv_data_cat = {
    'exp' : 'Transcriptome Profiling',
    'meth' : 'DNA Methylation'
}

# The Case information categories of which labels should be downloaded per case
expand_case_labels = ['diagnoses', 'demographic', 'exposures']
# The header to the case label file, currently contains all labels availbe in above categories, if more labels are found, the program will give a warning
#case_label_header = ['age_at_diagnosis', 'alcohol_history', 'alcohol_intensity', 'bmi', 'cigarettes_per_day', 'classification_of_tumor', 'days_to_birth', 'days_to_death', 'days_to_last_follow_up', 'days_to_last_known_disease_status', 'days_to_recurrence', 'demographic_created_datetime', 'demographic_id', 'demographic_submitter_id', 'demographic_updated_datetime', 'diagnoses_created_datetime', 'diagnoses_submitter_id', 'diagnoses_updated_datetime', 'diagnosis_id', 'ethnicity', 'exposure_id', 'exposures_created_datetime', 'exposures_submitter_id', 'exposures_updated_datetime', 'gender', 'height', 'last_known_disease_status', 'morphology', 'primary_diagnosis', 'prior_malignancy', 'progression_or_recurrence', 'race', 'site_of_resection_or_biopsy', 'state', 'tissue_or_organ_of_origin', 'tumor_grade', 'tumor_stage', 'vital_status', 'weight', 'year_of_birth', 'year_of_death', 'years_smoked']
#case_label_header = ['age_at_diagnosis', 'age_at_index', 'age_at_onset', 'age_is_obfuscated', 'ajcc_clinical_m', 'ajcc_clinical_n', 'ajcc_clinical_stage', 'ajcc_clinical_t', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_stage', 'ajcc_pathologic_t', 'ajcc_staging_system_edition', 'alcohol_days_per_week', 'alcohol_drinks_per_day', 'alcohol_history', 'alcohol_intensity', 'anaplasia_present', 'anaplasia_present_type', 'ann_arbor_b_symptoms', 'ann_arbor_clinical_stage', 'ann_arbor_extranodal_involvement', 'ann_arbor_pathologic_stage', 'asbestos_exposure', 'best_overall_response', 'bmi', 'breslow_thickness', 'burkitt_lymphoma_clinical_variant', 'cause_of_death', 'cause_of_death_source', 'child_pugh_classification', 'cigarettes_per_day', 'circumferential_resection_margin', 'classification_of_tumor', 'coal_dust_exposure', 'cog_liver_stage', 'cog_neuroblastoma_risk_group', 'cog_renal_stage', 'cog_rhabdomyosarcoma_risk_group', 'country_of_residence_at_enrollment', 'days_to_best_overall_response', 'days_to_birth', 'days_to_death', 'days_to_diagnosis', 'days_to_last_follow_up', 'days_to_last_known_disease_status', 'days_to_recurrence', 'demographic_created_datetime', 'demographic_id', 'demographic_submitter_id', 'demographic_updated_datetime', 'diagnoses_created_datetime', 'diagnoses_submitter_id', 'diagnoses_updated_datetime', 'diagnosis_id', 'enneking_msts_grade', 'enneking_msts_metastasis', 'enneking_msts_stage', 'enneking_msts_tumor_site', 'environmental_tobacco_smoke_exposure', 'esophageal_columnar_dysplasia_degree', 'esophageal_columnar_metaplasia_present', 'ethnicity', 'exposure_duration', 'exposure_id', 'exposure_type', 'exposures_created_datetime', 'exposures_submitter_id', 'exposures_updated_datetime', 'figo_stage', 'figo_staging_edition_year', 'first_symptom_prior_to_diagnosis', 'gastric_esophageal_junction_involvement', 'gender', 'gleason_grade_group', 'gleason_grade_tertiary', 'gleason_patterns_percent', 'goblet_cells_columnar_mucosa_present', 'greatest_tumor_dimension', 'gross_tumor_weight', 'height', 'icd_10_code', 'igcccg_stage', 'inpc_grade', 'inpc_histologic_group', 'inrg_stage', 'inss_stage', 'international_prognostic_index', 'irs_group', 'irs_stage', 'ishak_fibrosis_score', 'iss_stage', 'largest_extrapelvic_peritoneal_focus', 'last_known_disease_status', 'laterality', 'lymph_node_involved_site', 'lymph_nodes_positive', 'lymph_nodes_tested', 'lymphatic_invasion_present', 'margin_distance', 'margins_involved_site', 'marijuana_use_per_week', 'masaoka_stage', 'medulloblastoma_molecular_classification', 'metastasis_at_diagnosis', 'metastasis_at_diagnosis_site', 'method_of_diagnosis', 'micropapillary_features', 'mitosis_karyorrhexis_index', 'mitotic_count', 'morphology', 'non_nodal_regional_disease', 'non_nodal_tumor_deposits', 'occupation_duration_years', 'ovarian_specimen_status', 'ovarian_surface_involvement', 'pack_years_smoked', 'papillary_renal_cell_type', 'percent_tumor_invasion', 'perineural_invasion_present', 'peripancreatic_lymph_nodes_positive', 'peripancreatic_lymph_nodes_tested', 'peritoneal_fluid_cytological_status', 'pregnant_at_diagnosis', 'premature_at_birth', 'primary_diagnosis', 'primary_gleason_grade', 'prior_malignancy', 'prior_treatment', 'progression_or_recurrence', 'race', 'radon_exposure', 'residual_disease', 'respirable_crystalline_silica_exposure', 'secondary_gleason_grade', 'secondhand_smoke_as_child', 'site_of_resection_or_biopsy', 'smoking_frequency', 'state', 'supratentorial_localization', 'synchronous_malignancy', 'time_between_waking_and_first_smoke', 'tissue_or_organ_of_origin', 'tobacco_smoking_onset_year', 'tobacco_smoking_quit_year', 'tobacco_smoking_status', 'tobacco_use_per_day', 'transglottic_extension', 'tumor_confined_to_organ_of_origin', 'tumor_depth', 'tumor_focality', 'tumor_grade', 'tumor_largest_dimension_diameter', 'tumor_regression_grade', 'tumor_stage', 'type_of_smoke_exposure', 'type_of_tobacco_used', 'vascular_invasion_present', 'vascular_invasion_type', 'vital_status', 'weeks_gestation_at_birth', 'weight', 'weiss_assessment_score', 'wilms_tumor_histologic_subtype', 'year_of_birth', 'year_of_death', 'year_of_diagnosis', 'years_smoked']
case_label_header = ['age_at_diagnosis', 'age_at_index', 'age_at_onset', 'age_is_obfuscated', 'ajcc_clinical_m', 'ajcc_clinical_n', 'ajcc_clinical_stage', 'ajcc_clinical_t', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_stage', 'ajcc_pathologic_t', 'ajcc_staging_system_edition', 'alcohol_days_per_week', 'alcohol_drinks_per_day', 'alcohol_history', 'alcohol_intensity', 'alcohol_type', 'ann_arbor_b_symptoms', 'ann_arbor_clinical_stage', 'ann_arbor_extranodal_involvement', 'ann_arbor_pathologic_stage', 'asbestos_exposure', 'best_overall_response', 'burkitt_lymphoma_clinical_variant', 'cause_of_death', 'cause_of_death_source', 'child_pugh_classification', 'cigarettes_per_day', 'classification_of_tumor', 'coal_dust_exposure', 'cog_liver_stage', 'cog_neuroblastoma_risk_group', 'cog_renal_stage', 'cog_rhabdomyosarcoma_risk_group', 'country_of_residence_at_enrollment', 'days_to_best_overall_response', 'days_to_birth', 'days_to_death', 'days_to_diagnosis', 'days_to_last_follow_up', 'days_to_last_known_disease_status', 'days_to_recurrence', 'demographic_created_datetime', 'demographic_id', 'demographic_submitter_id', 'demographic_updated_datetime', 'diagnoses_created_datetime', 'diagnoses_submitter_id', 'diagnoses_updated_datetime', 'diagnosis_id', 'eln_risk_classification', 'enneking_msts_grade', 'enneking_msts_metastasis', 'enneking_msts_stage', 'enneking_msts_tumor_site', 'environmental_tobacco_smoke_exposure', 'esophageal_columnar_dysplasia_degree', 'esophageal_columnar_metaplasia_present', 'ethnicity', 'exposure_duration', 'exposure_id', 'exposure_type', 'exposures_created_datetime', 'exposures_submitter_id', 'exposures_updated_datetime', 'figo_stage', 'figo_staging_edition_year', 'first_symptom_prior_to_diagnosis', 'gastric_esophageal_junction_involvement', 'gender', 'gleason_grade_group', 'gleason_grade_tertiary', 'gleason_patterns_percent', 'goblet_cells_columnar_mucosa_present', 'icd_10_code', 'igcccg_stage', 'inpc_grade', 'inpc_histologic_group', 'inrg_stage', 'inss_stage', 'international_prognostic_index', 'irs_group', 'irs_stage', 'ishak_fibrosis_score', 'iss_stage', 'last_known_disease_status', 'laterality', 'margin_distance', 'margins_involved_site', 'marijuana_use_per_week', 'masaoka_stage', 'medulloblastoma_molecular_classification', 'metastasis_at_diagnosis', 'metastasis_at_diagnosis_site', 'method_of_diagnosis', 'micropapillary_features', 'mitosis_karyorrhexis_index', 'mitotic_count', 'morphology', 'occupation_duration_years', 'ovarian_specimen_status', 'ovarian_surface_involvement', 'pack_years_smoked', 'papillary_renal_cell_type', 'peritoneal_fluid_cytological_status', 'pregnant_at_diagnosis', 'premature_at_birth', 'primary_diagnosis', 'primary_disease', 'primary_gleason_grade', 'prior_malignancy', 'prior_treatment', 'progression_or_recurrence', 'race', 'radon_exposure', 'residual_disease', 'respirable_crystalline_silica_exposure', 'satellite_nodule_present', 'secondary_gleason_grade', 'secondhand_smoke_as_child', 'site_of_resection_or_biopsy', 'sites_of_involvement', 'smokeless_tobacco_quit_age', 'smoking_frequency', 'state', 'supratentorial_localization', 'synchronous_malignancy', 'time_between_waking_and_first_smoke', 'tissue_or_organ_of_origin', 'tobacco_smoking_onset_year', 'tobacco_smoking_quit_year', 'tobacco_smoking_status', 'tobacco_use_per_day', 'tumor_confined_to_organ_of_origin', 'tumor_depth', 'tumor_focality', 'tumor_grade', 'tumor_regression_grade', 'type_of_smoke_exposure', 'type_of_tobacco_used', 'vital_status', 'weeks_gestation_at_birth', 'weiss_assessment_score', 'who_cns_grade', 'who_nte_grade', 'wilms_tumor_histologic_subtype', 'year_of_birth', 'year_of_death', 'year_of_diagnosis', 'years_smoked']

# The labels that will be in each label category and therefor will not be unique
recurrent_labels = ['updated_datetime', 'submitter_id', 'created_datetime']

# Takes a TCGA API endpoint and query and will download and return the corresponding dictionary
def endpoint_to_dict(endpoint, query='', params=None):
    endpoint_url = 'https://api.gdc.cancer.gov/{}/'.format(endpoint)
    url = endpoint_url + query
    response = try_request_xtimes(url, params)
    return dict(response.json())


# Based on a file type abreviation this funciton will start the process of downloading this file type for all project availble in TCGA
def download_all_files_of_all_projects_by_filetype(data_abv, overwrite_cases=False, verbose=100):
    data_projects = get_all_projects_by_filetype(data_abv)
    for idx, project in enumerate(data_projects):
        print('Downloading {} data from {} project ({}/{})'.format(abv_data_type[data_abv], project, idx,
                                                                   len(data_projects)))
        set_project_and_folder(project)
        cases = all_cases_data_type(data_abv, overwrite_cases, verbose=verbose)
        download_files_of_type_of_cases(data_abv, cases, verbose=verbose)


# Get all project codes that have files of the given file type
def get_all_projects_by_filetype(data_abv):
    projects = get_all_projects(data_abv=data_abv)
    return [x[0] for x in projects if type(x[-1]) is int]


# Gets all availeble project codes
def get_all_projects(data_abv=None):
    query = '?from=0&size=1000' #&sort=project.project_id:asc'
    if data_abv is not None:
        query += '&expand=summary.data_categories'
    resp_dict = endpoint_to_dict('projects', query)

    p_data = resp_dict['data']['hits']
    projects = []
    for p in p_data:
        project = [p['project_id'], p['name'], p['primary_site']]
        if data_abv is not None:
            file_type = abv_data_cat[data_abv]
            for cat in p['summary']['data_categories']:
                if cat['data_category'] == file_type:
                    project.append(cat['case_count'])

        projects.append(project)

    return projects


# Takes type id tag and returns a boolean representing if the sample is tumerous or not
def sample_type_id_is_tumor(type_id):
    # return true if first char is 1, false if 0
    if type_id:
        try:
            tumor_tag = int(type_id[0])
        except Exception as e:
            print('No valid type id (not 0 or 1) found: {}'.format(type_id))
            return None
    else:
        print('WARNING uncertain type, type_id is None'.format(type_id))
        return None
    if tumor_tag == 1:
        return False
    elif tumor_tag == 0:
        return True
    else:
        print('WARNING uncertain type id (not 0 or 1) found: {}'.format(type_id))
        return None


# Function retrieving all case identifiers within the project that have data of a certain datatype
def all_cases_data_type(data_abv, overwrite_cases=False, verbose=100):
    cases_file = '{}/cases_{}.csv'.format(project_path, data_abv)
    labels_file = os.path.join(project_path, 'case_labels_{}.csv'.format(data_abv))
    if not overwrite_cases and os.path.exists(cases_file):
        return pd.read_csv(cases_file, index_col=0)

    endpoint = 'cases'
    data_type = abv_data_type[data_abv]
    json_input = '{"op":"and","content":[{"op":"in","content":{"field":"project.project_id","value":"' + project_id + '"}},{"op":"=","content":{"field":"files.data_type","value":"' + data_type + '"}}]}'
    expand_string = ','.join(expand_case_labels)
    params = {'filters': json_input, 'size': 1500, 'expand': expand_string}

    json_out = endpoint_to_dict(endpoint, params=params)

    cases, data, = [], []
    for h in json_out["data"]["hits"]:
        case_id = h["case_id"]
        cases.append(case_id)

        d, head = [], []
        for l in expand_case_labels:
            try:
                info = h[l]
            except KeyError as e:
                if verbose > 25:
                    print('Case {} is missing information of type {}'.format(case_id, e))
                continue
            label = info[0] if type(info) is list else info
            for key, val in label.items():
                if key in recurrent_labels:
                    key = '{}_{}'.format(l, key)
                head.append(key)
                if val == 'not reported' or val is None:
                    val = np.nan
                d.append(val)


        d_series = pd.DataFrame(list(zip(head, d)), columns=['header', 'values']).sort_values('header')
        d_series = d_series.drop_duplicates('header').reset_index(drop=True)

        cur_header = list(d_series['header'].values)
        if cur_header != case_label_header:
            if set(cur_header) < set(case_label_header):
                if verbose > 25:
                    print('Case {} has an incomplete label set, missing labels are left empty'.format(case_id))
                d_series = d_series.set_index('header')
                try:
                    d_series = d_series.reindex(case_label_header, fill_value=np.nan)
                except ValueError as e:
                    print('Duplicate header values found with case {}, error: {}'.format(case_id, e))
                    print('Probably caused by multiple label file have equally named attributes')
                    print('Adding the duplicate value to the recurrent_labels parameter at the top of this script will solve your issue')
                    print(np.unique(d_series.index.values,return_counts=True))
                    exit()
            else:
                print('Case {} has labels that are not expected to be in the label set'.format(case_id))
                print('Fix the case_label_header parameter at the top of this script')
                print('Expected header values: {}'.format(case_label_header))
                print('Observed header values: {}'.format(cur_header))
                exit()
        data.append(d_series['values'].values)

    labels_df = pd.DataFrame(data, columns=case_label_header, index=cases)
    labels_df.index.names = ['case_id']
    labels_df.to_csv(labels_file)

    cases_pd = pd.DataFrame(cases, columns=['case_id'])
    cases_pd.to_csv(cases_file)

    return cases_pd


# downloads one file based on TCGA file id, it will retry a given amout of times when it fails
def downloadFile(fid, x=10):
    i = 0
    while i < x:
        try:
            response = urllib.request.urlretrieve("https://api.gdc.cancer.gov/data/" + fid)
            break
        except Exception as e:
            print('Error while retrieving file, attempt {}/{}'.format(i,x))
            print(e)
            i += 1
    else:
        print('Download still failed after {} times, skipping file {} for now'.format(x, fid))


    return response


# Gives TCGA file identifier of a file having a certain dataType belonging to a certain case ID
# NOTE only tested with datatypes  "Gene Expression Quantification" and "Methylation Beta Value"
def getFileIdOfDataType(dataType, caseId):
    endpoint = 'files'
    json_input_exp = '{"op":"and","content":[{"op":"in","content":{"field":"cases.case_id","value":"' + caseId + '"}},{"op":"=","content":{"field":"files.data_type","value":"' + dataType + '"}}]}'
    params = {'filters': json_input_exp, 'size': 1500, 'expand' : 'cases.samples'}

    json_out = endpoint_to_dict(endpoint, params=params)

    file_ids = []
    for h in json_out["data"]["hits"]:
        if dataType in 'Gene Expression Quantification':
            if exp_file_type in h["file_name"]:
                fid = h["file_id"]
                if 'sample_type_id' in h['cases'][0]['samples'][0]:
                    sample_type_id = h['cases'][0]['samples'][0]['sample_type_id']
                else:
                    sample_type_id = 'unkown'
                file_ids.append([fid, sample_type_id])
            else:
                print(f'Skipping {h["file_name"]} not selected exp_file_type')
        else:
            file_ids.append(h["file_id"])

    return file_ids


# writes a file to a location based on a case id and a file name
def writeFile(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        if (type(file) is str):
            file = open(file, 'rb')
        with open(file_path, 'wb') as fw:
            fw.write(file.read())
    except Exception as e:
        print('Error occured while writing: {}'.format(e))
        os.remove(file_path)


# checks if given file is compressed with zip, gzip or tar, OPTIONAL it returns the unzipted file
def check_if_compressed(file, decompress=False):
    gzip_head = b'\x1f\x8b\x08'
    if gzip_head in open(file, 'rb').read(40):
        if decompress:
            return gzip.open(file)
        else:
            return True
    elif tarfile.is_tarfile(file):
        if decompress:
            return tarfile.open(file)
        else:
            return True
    elif zipfile.is_zipfile(file):
        if decompress:
            return zipfile.open(file)
        else:
            return True
    else:
        if decompress:
            return file
        else:
            return False


# Based on TCGA fid the file is downloaded and written to appropriate file structure
def downWriteFID(fid, case, data_abv):
    global project_path
    path = os.path.join(project_path, case)
    fname = '{}_{}.txt'.format(data_abv, fid)
    file_path = os.path.join(path, fname)

    if (os.path.exists(file_path)):
        if (os.path.getsize(file_path) < 20):
            os.remove(file_path)
        else:
            return False
    try:
        file = downloadFile(fid)
    except Exception as e:
        print('{} not downloaded due to error {}'.format(fid, e))
        return True

    temp_file_path = file[0]

    try:
        decompressed = False
        while not decompressed:
            writeFile(check_if_compressed(temp_file_path, decompress=True), file_path)
            temp_file_path = file_path
            if not check_if_compressed(temp_file_path):
                decompressed = True

    except Exception as e:
        print('Unable to de-compress file {} error: {}'.format(fid, e))
        return True
    return False


# takes a set of case ids and a file type, and downloads all files availeble for that case corresponding to the file type
def download_files_of_type_of_cases(data_abv, cases, verbose=100):
    data_type = abv_data_type[data_abv]
    case_len = len(cases.index)
    fids_file = '{}/downloaded_fids_{}.csv'.format(project_path, data_abv)
    if os.path.exists(fids_file):
        fids = pd.read_csv(fids_file, index_col=0)
    else:
        fids = None
    temp_fids = []
    for index, row in cases.iterrows():
        case_id = row[0]
        if fids is not None and any(fids.case_id == case_id):
            if verbose > 99:
                print('Found case {}'.format(case_id))
            found_case = fids.loc[fids['case_id'] == case_id]
            idx = found_case.index[0]
            found_fid =  found_case.values.tolist()[0][1]
            if 'error_' not in found_fid:
                continue
            else:
                print('An error occured on last download attempt, retrying...')
            fids.drop(idx, inplace=True)

        print("Retrieving case info {}/{} of case {}".format(index + 1, case_len, case_id))
        fid_tumor_list = getFileIdOfDataType(data_type, case_id)


        for fid_tumor in fid_tumor_list:
            fid = fid_tumor[0] if type(fid_tumor) == list else fid_tumor
            tumor = sample_type_id_is_tumor(fid_tumor[1])
            if tumor is None:
                print('Skipping {} due to unvalid tumor indicator'.format(fid))
                continue
            print('Downloading file {}'.format(fid))
            if (downWriteFID(fid, case_id, data_abv)):
                fid = 'error_{}'.format(fid)

            temp_fids.append([case_id, fid, tumor])

        if index % 10 == 1 or index == len(cases) - 1:
            pd_fids = pd.DataFrame(temp_fids, columns=['case_id', 'fid_{}'.format(data_abv), 'tumor'])
            if fids is None:
                fids = pd_fids
            else:
                fids = pd.concat([fids, pd_fids], ignore_index=True)
            fids.drop_duplicates('fid_{}'.format(data_abv), inplace=True)
            fids.to_csv(fids_file)
            temp_fids = []
    if verbose > 50:
        print('Finished downloading files of {} type: {}'.format(project_id, data_type))
    error_cases = error_files_exist(fids)
    if error_cases:
        print('The following list of cases/files have not been downloaded due to errors')
        print(error_cases)
        print('Rerun the script and it will try to download them again')
    else:
        if verbose > 50:
            print('No errors exist so all files are downloaded')


# Checks the status of the case array for if an error had occured
def error_files_exist(cases):
    error_cases = []
    for index, row in cases.iterrows():
        case_id = row[0]
        fid = row[1]
        if 'error_' in fid:
            error_cases.append([case_id, fid, index])

    return error_cases


# Execustes ar requested query based on endpoint and parameters, tries a given amount of times
def try_request_xtimes(end_point, parameters=None, x=10, stream=None):
    i = 0
    while i < x:
        try:
            response = requests.get(end_point, params=parameters, stream=stream)
        except Exception as e:
            print(e)
            i += 1
            continue
        if response.status_code == 200:
            break
        print('Got a server error {} times'.format(i + 1))
        i += 1
    if response.status_code != 200:
        print('Got 10 server errors in a row, exiting...')
        print('Got response status {}'.format(response.status_code))
        print(response.reason())

    return response


# sets the global project variable used by all functions
def set_project_and_folder(project):
    global project_id
    global project_path
    global out_path
    project_id = project
    project_path = os.path.join(out_path, project)

    if not os.path.exists(project_path):
        os.makedirs(project_path)

# main function
if __name__ == "__main__":
    data_abv = 'exp'
    download_all_files_of_all_projects_by_filetype(data_abv, overwrite_cases=True, verbose=20)


