#!/usr/bin/env python3
"""
Created on Thu Mar 29 13:47:45 2018

AUTHORS: Mohammad M. Ghassemi
       : Benjamin E. Moody

PURPOSE: This script prepares an entry for the physionet 2018 Challenge

REQUIREMENTS: We assume that you have downloaded the data from
              https://physionet.org/physiobank/database/challenge/2018/#files
"""
import numpy as np
import os
import sys
import physionetchallenge2018_lib as phyc
from score2018 import Challenge2018Score
from pylab import find
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from zipfile import ZipFile, ZIP_DEFLATED
import gc

import train_classifier as T
import run_my_classifier as R

# -----------------------------------------------------------------------------
# Generate the data to train the classifier
# -----------------------------------------------------------------------------
def train():
    T.init()

    # Generate a data frame that points to the challenge files
    tr_files, te_files = phyc.get_files()

    # For each subject in the training set...
    for i in range(0, np.size(tr_files, 0)):
        gc.collect()
        print('Preprocessing training subject: %d/%d'
              % (i + 1, np.size(tr_files, 0)))
        record_name = tr_files.header.values[i][:-4]
        T.preprocess_record(record_name)

    T.finish()

# -----------------------------------------------------------------------------
# Run the classifier on each training subject, and compute the mean performance
# -----------------------------------------------------------------------------
def score_training_set():
    # Generate a data frame that points to the challenge files
    tr_files, te_files = phyc.get_files()

    score = Challenge2018Score()
    for i in range(0, np.size(tr_files, 0)):
        gc.collect()
        sys.stdout.write('Evaluating training subject: %d/%d'
                         % (i + 1, np.size(tr_files, 0)))
        sys.stdout.flush()
        record_name = tr_files.header.values[i][:-4]
        predictions = R.classify_record(record_name)

        arousals = phyc.import_arousals(tr_files.arousal.values[i])
        arousals = np.ravel(arousals)

        score.score_record(arousals, predictions, record_name)
        auroc = score.record_auroc(record_name)
        auprc = score.record_auprc(record_name)
        print(' AUROC:%f AUPRC:%f' % (auroc, auprc))

    print()
    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()
    print('Training AUROC Performance (gross): %f' % auroc_g)
    print('Training AUPRC Performance (gross): %f' % auprc_g)
    print()

# -----------------------------------------------------------------------------
# Run the classifier on each test subject, and save the predictions
# for submission
# -----------------------------------------------------------------------------
def evaluate_test_set():
    # Generate a data frame that points to the challenge files
    tr_files, te_files = phyc.get_files()

    for i in range(0, np.size(te_files, 0)):
        gc.collect()
        print('Evaluating test subject: %d/%d' % (i+1, np.size(te_files, 0)))
        record_name = te_files.header.values[i][:-4]
        output_file = os.path.basename(record_name) + '.vec'
        predictions = R.classify_record(record_name)
        np.savetxt(output_file, predictions, fmt='%.3f')

# -----------------------------------------------------------------------------
# Build a zip file for submission to the Challenge
# -----------------------------------------------------------------------------
def package_entry():
    with ZipFile('entry.zip', 'w', ZIP_DEFLATED) as myzip:
        for dirName, subdirList, fileList in os.walk('.'):
            for fname in fileList:
                if ('.vec' in fname[-4:] or '.py' in fname[-3:]
                        or '.pkl' in fname[-4:] or '.txt' in fname[-4:]):
                    myzip.write(os.path.join(dirName, fname))

if __name__ == '__main__':
    train()
    score_training_set()
    evaluate_test_set()
    package_entry()
