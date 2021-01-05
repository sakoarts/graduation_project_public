"""
This script will add some labels based on the existing labels that can be used in training
"""

import os
import pandas as pd
import numpy as np

label_path = '../data/TCGA/'
label_in_file = 'exp_TCGA_coded_labels.csv'
label_out_file = 'exp_TCGA_coded_labels_add.csv'

# adds a float representation of kown stage labels and a label for late or early stage diagnosis
def new_stage_label(df):
    print('Adding tumor_stage_float and stage_diagnosis labels')
    stage_series = df['tumor_stage']

    stages = ['i', 'i/ii nos', 'ii', 'iii', 'iii/iv', 'iiib', 'iiib/v', 'is', 'iv', 'iv/v', 'stage 0', 'stage 2b',
              'stage 3', 'stage 4', 'stage 4s', 'stage i', 'stage ia', 'stage ib', 'stage ii', 'stage iia', 'stage iib',
              'stage iic', 'stage iii', 'stage iiia', 'stage iiib', 'stage iiic', 'stage iv', 'stage iva', 'stage ivb',
              'stage ivc', 'stage x']
    floats = [1, 1.5, 2, 3, 3.5, 3.5, 3.5, 1, 4, 4.5, 0, 2.5, 3, 4, 4, 1, 1.25, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5,
              3.75, 4, 4.25, 4.5, 4.75, 5]
    stage_float = pd.Series(floats, index=stages)

    float_series = stage_series.map(stage_float)
    stage_diagnosis = float_series.apply(lambda x: 'early' if x < 3 else 'late')

    df['tumor_stage_float'] = float_series
    df['stage_diagnosis'] = stage_diagnosis

    return df


# adds a bmi category label based on bmi value
def new_bmi_label(df):
    print('Adding bmi_category label')
    bmi_series = df['bmi']

    bmi_category_series = bmi_series.apply(bmi_to_label)

    df['bmi_category'] = bmi_category_series

    return df


# bmi translator function
def bmi_to_label(bmi):
    # Source: https://www.webmd.com/a-to-z-guides/body-mass-index-bmi-for-adults
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'healthy'
    elif bmi < 30:
        return 'overweight'
    elif bmi >= 30:
        return 'obese'
    else:
        return np.nan


# adds a long or short survivor category label and a days survived value
def new_survivor_label(df):
    print('Adding survivor and days survived labels')
    df = df.apply(serie_to_survivor, axis=1)
    return df


# survivor construction function
def serie_to_survivor(serie):
    long_threshold_days = 5 * 365

    vital_status = serie['vital_status']
    # days from initial diagnosis to last known vital status
    days_to_last_follow_up = serie['days_to_last_follow_up']
    # days from initial diagnosis to death
    days_to_death = serie['days_to_death']

    survivor = np.nan
    days_survived = np.nan
    if vital_status == 'alive':
        days_survived = days_to_last_follow_up
        if days_survived >= long_threshold_days:
            survivor = 'long'
        else:
            survivor = 'yet_unkown'
    elif vital_status == 'dead':
        days_survived = days_to_death
        if days_survived >= long_threshold_days:
            survivor = 'long'
        else:
            survivor = 'short'

    serie['survivor'] = survivor
    serie['days_survived'] = days_survived

    return serie


# main function
if __name__ == "__main__":
    print('Reading raw labels')
    label_set = pd.read_csv(os.path.join(label_path, label_in_file), header=0, index_col=0)

    # add stage labels
    label_set = new_stage_label(label_set)
    # add bmi category label
    label_set = new_bmi_label(label_set)
    # add survivor labels
    label_set = new_survivor_label(label_set)

    print('Writing enriched labels')
    label_set.to_csv(os.path.join(label_path, label_out_file))