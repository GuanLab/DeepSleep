## parse annotation
import xml.etree.ElementTree as ET
import mne
import numpy as np
import sys
import os
import scipy.io
import cv2

path1='/ssd/hongyang/2018/physionet/data/shhs/'

size= 2**20
scale_pool=5
freq=125

# all shhs1 records
id_all=[]
f=open('id_shhs1.txt','r')
for line in f:
    id_all.append(line.rstrip())
f.close()

# only freq=125Hz (so that we can avg5) & channel names match
ch_names=['SaO2', 'H.R.', 'EEG(sec)', 'ECG', 'EMG', 'EOG(L)', 'EOG(R)', 'EEG', 'SOUND', 'AIRFLOW']
#['SaO2', 'H.R.', 'EEG(sec)', 'ECG', 'EMG', 'EOG(L)', 'EOG(R)', 'EEG', 'AIRFLOW', 'THOR RES', 'ABDO RES', 'POSITION', 'LIGHT', 'NEW AIR', 'OX stat']
id_test=[]
for the_id in id_all:
    edf = mne.io.read_raw_edf(path1 + the_id + '.edf', verbose=False)
    print(the_id, edf.info['sfreq'])
    #if (edf.info['sfreq'] % 25 == 0):
    if (edf.info['sfreq'] == freq) & (edf.ch_names[:10] == ch_names) & (edf.n_times <= size*scale_pool):
        id_test.append(the_id)

f=open('id1.txt', 'w')
for the_id in id_test:
    f.write('%s\n' % the_id)
f.close()

########################################


size= 2**20
scale_pool=10
freq=250

# all shhs2 records
id_all=[]
f=open('id_shhs2.txt','r')
for line in f:
    id_all.append(line.rstrip())
f.close()

# only freq=250Hz (so that we can avg10) & channel names match
ch_names=['SaO2', 'PR', 'EEG(sec)', 'ECG', 'EMG', 'EOG(L)', 'EOG(R)', 'EEG', 'AIRFLOW', 'THOR RES', 'ABDO RES', 'POSITION', 'LIGHT', 'OX STAT']
id_test=[]
for the_id in id_all:
    edf = mne.io.read_raw_edf(path1 + the_id + '.edf', verbose=False)
    print(the_id, edf.info['sfreq'])
    #if (edf.info['sfreq'] % 25 == 0):
    if (edf.info['sfreq'] == freq) & (edf.ch_names == ch_names) & (edf.n_times <= size*scale_pool):
        id_test.append(the_id)

f=open('id2.txt', 'w')
for the_id in id_test:
    f.write('%s\n' % the_id)
f.close()


