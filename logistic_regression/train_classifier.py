#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np
import physionetchallenge2018_lib as phyc
import matplotlib
from pylab import find
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def init():
    # Create the 'models' subdirectory and delete any existing model files
    try:
        os.mkdir('models')
    except OSError:
        pass
#    for f in glob.glob('models/*_model.pkl'):
#        os.remove(f)

def preprocess_record(record_name):
    header_file = record_name + '.hea'
    signal_file = record_name + '.mat'
    arousal_file = record_name + '-arousal.mat'

    # Get the signal names from the header file
    signal_names, Fs, n_samples = phyc.import_signal_names(header_file)
    signal_names = list(np.append(signal_names, 'arousals'))

    # Convert this subject's data into a pandas dataframe
    this_data = phyc.get_subject_data(arousal_file, signal_file, signal_names)

    # ----------------------------------------------------------------------
    # Generate the Features for the classificaition model - variance of SaO2
    # ----------------------------------------------------------------------

    # For the baseline, let's only look at how SaO2 might predict arousals

    SaO2 = this_data.get(['SaO2']).values
    arousals = this_data.get(['arousals']).values

    # We select a window size of 60 seconds with no overlap to compute
    # the features
    step        = Fs * 60
    window_size = Fs * 60

    # Initialize the matrices that store our training data
    X_subj = np.zeros([((n_samples) // step), 1])
    Y_subj = np.zeros([((n_samples) // step), 1])

    # Extract the variance of the SaO2 in 60 second windows as a feature
    for idx, k in enumerate(range(0, (n_samples-step+1), step)):
        X_subj[idx, 0] = np.var(np.transpose(SaO2[k:k+window_size]), axis=1)
        Y_subj[idx]    = np.max(arousals[k:k+window_size])

    # Ignore records that do not contain any arousals
    if not np.any(Y_subj):
        sys.stderr.write('no arousals found in %s\n' % record_name)
        return

    # ---------------------------------------------------------------------
    # Train a (multi-class) Logistic Regression classifier
    # ---------------------------------------------------------------------
    my_classifier = LogisticRegression()
    my_classifier.fit(X_subj, np.ravel(Y_subj))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Save this algorithm for submission to Physionet Challenge:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model_file = 'models/%s_model.pkl' % os.path.basename(record_name)
    joblib.dump(my_classifier, model_file)

def finish():
    pass

if __name__ == '__main__':
    init()
    for record in sys.argv[1:]:
        preprocess_record(record)
    finish()
