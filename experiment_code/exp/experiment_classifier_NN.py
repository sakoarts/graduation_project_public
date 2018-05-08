import h5py
import numpy as np
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

sys.path.append('../')
from tools.generator import DataGeneratorClassifier
from tools.callbacks import MetricsToCSV

# set keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow' # 'cntk' # 'theano' #

from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.callbacks import TensorBoard

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

# DEBUG parameters
DEBUG = True
if DEBUG:
    save_model = False
    tensorboard = False
else:
    save_model = True
    tensorboard = True

# create experiment:
ex = Experiment('PAN_classifier_NN')

# add file observer
observer_path = '../runs/DEBUG' if DEBUG else '../runs'
ex.observers.append(FileStorageObserver.create(basedir=os.path.join(observer_path, ex.path)))

if not DEBUG:
    # add mongo observer
    with open('../tools/.mongo', 'r') as f:
        auth_url = f.read()
        ex.observers.append(MongoObserver.create(url=auth_url, db_name='graduation'))


@ex.config
def my_config():
    # Data set parameters
    dataset_name = 'TCGA'
    datasubset_name = '_coded'
    data_fname = 'exp_{}{}_normalized.hdf5'.format(dataset_name, datasubset_name)
    data_path = os.path.join('../data/normalized/', data_fname)

    # label set parameter
    label_path = '../data/{}/exp_{}{}_labels_add.csv'.format(dataset_name, dataset_name, datasubset_name)

    # can contain a list of genes used in training and testing, all data from other genes will be disregarded
    genes_to_select = None

    # Train test split
    tt_split = 0.25

    # True or False, if True, balanced classweights will be used
    class_weights = True

    # Neural Network parameters
    layer_sizes = [200, 20]
    batch_size = 8
    samples_per_epoch = 8
    nb_epoch = 500

    # Diffrent labels present in dataset
    label_options = ['tumor', 'project', 'primary_site', 'subtype_tumor', 'site_tumor', 'tumor_stage', 'bmi_category', 'tumor_stage_float']
    #sample_id, case_id, project, primary_site, tumor, subtype_tumor, age_at_diagnosis, alcohol_history, bmi, cigarettes_per_day, days_to_birth, days_to_death, days_to_last_follow_up, demographic_created_datetime, demographic_id, demographic_submitter_id, demographic_updated_datetime, diagnoses_created_datetime, diagnoses_submitter_id, diagnoses_updated_datetime, diagnosis_id, ethnicity, exposure_id, exposures_submitter_id, exposures_updated_datetime, gender, height, morphology, primary_diagnosis, race, site_of_resection_or_biopsy, state, tissue_or_organ_of_origin, tumor_stage, vital_status, weight, year_of_birth, year_of_death, years_smoked

    # The subset of the data you would like to involve in training
    subset_query_options = ['All', ['Kidney'], ['Eye']]
    subset_query = subset_query_options[1]

    # The label type of the subset chosen to train on
    subset_label = label_options[2]

    # The label that you would like to predict
    predict_label = label_options[3]

    # The label on which the tt_split will be based
    split_label = label_options[3]

@ex.capture
def data(data_path, label_path, subset_label, predict_label, split_label, batch_size, subset_query, tt_split, class_weights, _seed, genes_to_select):
    # Select genes and set row size variable
    if genes_to_select is None:
        data_h5 = h5py.File(data_path)
        rows = data_h5['gene_id'].shape[0]
    else:
        data_h5 = h5py.File(data_path)
        all_genes = data_h5['gene_id'][:].astype(str)
        genes_overlap = np.in1d(all_genes, genes_to_select)
        genes_to_select = np.where(genes_overlap == True)[0]
        rows = len(genes_to_select)

    # Read labels CSV
    label_set = pd.read_csv(label_path, index_col=0, header=0)

    # Account for All query
    if subset_query == 'All':
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
        train_k, test_k = train_test_split(split_keys, test_size=tt_split, random_state=_seed)
        train_keys += train_k
        test_keys += test_k

    # Get all labels and set related variables
    labels = subset_set[predict_label]
    unique_labels = np.unique(labels)
    number_of_classes = len(unique_labels)

    # One hot encode the labels and create sample_to_label set
    onehot_encoder = LabelBinarizer().fit(unique_labels)
    onehots = onehot_encoder.transform(labels)
    sample_to_label = pd.DataFrame(list(zip(labels.values, onehots)), index=labels.index, columns=['label', 'onehot'])

    # Construct generators
    training_generator = DataGeneratorClassifier(data_path, sample_to_label=sample_to_label, keys=train_keys, number_of_classes=number_of_classes, rows=rows, batch_size=batch_size, class_weights=class_weights, genes_to_select=genes_to_select).generate()
    validation_generator = DataGeneratorClassifier(data_path, sample_to_label=sample_to_label, keys=test_keys, number_of_classes=number_of_classes, rows=rows, batch_size=batch_size, class_weights=class_weights, genes_to_select=genes_to_select).generate()

    return training_generator, validation_generator, rows, number_of_classes


@ex.capture
def model(rows, batch_size, layer_sizes, number_of_classes):
    # This is our input placeholder
    input = Input(batch_shape=(batch_size, rows), name='input')

    # Dense layers
    dense = input
    for l in range(len(layer_sizes)):
        dense = Dense(layer_sizes[l], activation='relu', name='dense_{}'.format(l))(dense)

    # comprise to class output
    dense_out = Dense(number_of_classes, activation='relu', name='dense_out')(dense)

    # Softmax activation output layer
    output = Activation('softmax', name='output')(dense_out)

    # Construct model
    m = Model(inputs=[input], outputs=[output])

    return m


@ex.automain
def train(nb_epoch, samples_per_epoch):
    training_generator, validation_generator, rows, number_of_classes = data()
    out_path = ex.observers[0].dir

    m = model(rows=rows, number_of_classes=number_of_classes)

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = []
    callbacks.append(MetricsToCSV(os.path.join(out_path, 'metrics.csv')),)
    if tensorboard:
        callbacks.append(TensorBoard(log_dir=out_path, write_graph=True))

    m.summary()

    history = m.fit_generator(generator=training_generator,
                              steps_per_epoch=samples_per_epoch,
                              epochs=nb_epoch,
                              validation_data=validation_generator,
                              validation_steps=samples_per_epoch,
                              callbacks=callbacks)

    if save_model:
        m.save(os.path.join(out_path, 'model.h5'))

    results = {}
    for key in history.history.keys():
        results[key] = history.history[key][-1]

    return results