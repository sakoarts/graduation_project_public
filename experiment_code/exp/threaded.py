"""
This script is a controller script for running experiments in parralleel in multiple threads
It only works with script set up for this which currently is only experiment_classifier_threaded.py
"""

import sys
import numpy as np
import time
import subprocess
import json
from gevent import Timeout
from gevent.timeout import Timeout as TimeoutException
from gevent import monkey
monkey.patch_all()

from multiprocessing.dummy import Pool

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

sys.path.append('../')

experiment_file = 'experiment_regressor_threaded.py'

python_interpeter = '/usr/bin/anaconda3/bin/python3'

run = 0

# Run a single thread
def run_thread(classifier_name, classifier_params, fold, n_genes, run_delay, timeout_sec):
    try:
        try:
            time.sleep((np.random.random_sample())+run_delay*10)
            timeout = Timeout(timeout_sec)
            timeout.start()
            print('Starting {} Classifier fold {} run: {}'.format(classifier_name, fold, run_delay))
            out = subprocess.call([python_interpeter, './{}'.format(experiment_file), str(classifier_name), str(classifier_params), str(n_genes)])
            print('Done with classifier {} fold {} run: {}'.format(classifier_name, fold, run_delay))

            if out == 0:
                return 'Success_{}_{}_{}'.format(classifier_name, fold, run_delay)
            elif out == 1:
                return 'Error_{}_{}_{}'.format(classifier_name, fold, run_delay)
            return out
        except TimeoutException as te:
            print('Set timeout of {} has been reached in '.format(te))
            print('Classifier {}, fold {}, run {}'.format(classifier_name, fold, run_delay))
            print('Probably caused by a locked thread')
            return 'Timeout_{}_{}_{}'.format(classifier_name, fold, run_delay)
    except Exception as e:
        print('Exception {} occured during threading of:'.format(e))
        print('Classifier {}, fold {}, run {}'.format(classifier_name, fold, run_delay))
        return 'Catched Exception: {}, Classifier: {}, fold: {}, run: {}'.format(e, classifier_name, fold, run_delay)

if __name__ == "__main__":
    # the classifiers
    classifiers = [
        LinearSVR(),
        #RandomForestRegressor(),
    ]
    # their parameters, should be in the correct order
    classifiers_params = [
        {'C': 0.25},
        #{'n_estimators' : 100},
    ]

    # number of genes selected per run
    n_genes = 250
    # number of folds per classifier
    n_folds = 2500
    # number of threads
    n_threads = 18
    # time after which a thread is killed
    timeout_sec = 7200

    args = []
    for ic, c in enumerate(classifiers):
        c_name = c.__class__.__name__
        for idx, f in enumerate(range(n_folds)):
            run_delay = idx % n_threads
            c_dict = json.dumps(classifiers_params[ic])
            args.append((c_name, c_dict, f, n_genes, run_delay, timeout_sec))

    pool = Pool(n_threads)

    print('Running {} algorithms, {} folds each'.format(len(classifiers), n_folds))
    results = pool.starmap(run_thread, args)
    pool.close()
    pool.join()
    print(results)