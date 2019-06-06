#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import cv2
import time
import scipy.io
import glob
import unet
from keras import backend as K
import tensorflow as tf
import keras
import cv2 
from scipy import signal
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def import_arousals(file_name): # target
    import h5py
    import numpy as np
    f = h5py.File(file_name, 'r')
    arousals = np.transpose(np.array(f['data']['arousals']))
    return arousals

def import_signals(file_name): # feature
    return scipy.io.loadmat(file_name)['val']

def score_record(truth, predictions, input_digits=None):
    if input_digits is None: # bin resolution
        input_digits = 3 
    scale=10**input_digits
    pos_values = np.zeros(scale + 1, dtype=np.int64)
    neg_values = np.zeros(scale + 1, dtype=np.int64)

    b = scale+1
    r = (-0.5 / scale, 1.0 + 0.5 / scale)
    all_values = np.histogram(predictions, bins=b, range=r)[0]
    if np.sum(all_values) != len(predictions):
        raise ValueError("invalid values in 'predictions'")

    pred_pos = predictions[truth > 0]
    pos_values = np.histogram(pred_pos, bins=b, range=r)[0]
    pred_neg = predictions[truth == 0]
    neg_values = np.histogram(pred_neg, bins=b, range=r)[0]

    return (pos_values, neg_values)

def calculate_auc(pos_values,neg_values): # auc & auprc; adapted from score2018.py

    tp = np.sum(pos_values)
    fp = np.sum(neg_values)
    tn = fn = 0 
    tpr = 1 
    tnr = 0 
    if tp == 0 or fp == 0:
        # If either class is empty, scores are undefined.
        return (float('nan'), float('nan'))
    ppv = float(tp) / (tp + fp) 
    auroc = 0 
    auprc = 0 

    for (n_pos, n_neg) in zip(pos_values, neg_values):
        tp -= n_pos
        fn += n_pos
        fp -= n_neg
        tn += n_neg
        tpr_prev = tpr 
        tnr_prev = tnr 
        ppv_prev = ppv 
        tpr = float(tp) / (tp + fn) 
        tnr = float(tn) / (tn + fp) 
        if tp + fp > 0:
            ppv = float(tp) / (tp + fp) 
        else:
            ppv = ppv_prev
        auroc += (tpr_prev - tpr) * (tnr + tnr_prev) * 0.5 
        auprc += (tpr_prev - tpr) * ppv_prev
    return (auroc, auprc)

def pool_avg_2(input,if_mask=False):
    index1=np.arange(0,input.shape[1],2)
    index2=np.arange(1,input.shape[1],2)
    if (len(index2)<len(index1)):
        index2=np.concatenate((index2,[input.shape[1]-1]))
    output = (input[:,index1] + input[:,index2]) / float(2)
    if (if_mask): # -1 position are masked by -1, not avg
        mask = np.minimum(input[:,index1],input[:,index2])
        output[mask<0]=-1
    return output


###### PARAMETER ###############

#channel=8
size=2**16
num_channel=60
write_vec=True # whether generate .vec prediction file
overlap=0.5 # overlop/stride
size_edge=int(100) # chunck edges to be excluded
ss=10 # for sorenson dice
reso_digits=3 # auc resolution
batch=1
num_pool=8
name_model='weights_' + sys.argv[1] + '.h5'

num_augtest=5

################################

scale_pool=2**num_pool
scale=10**reso_digits
positives_all = np.zeros(scale + 1, dtype=np.int64)
negatives_all = np.zeros(scale + 1, dtype=np.int64)
auc_auprc=open('auc_auprc_' + sys.argv[1] + '.txt','w')
eva=open('eva_global_' + sys.argv[1] + '.txt','w')


if __name__ == '__main__':
    model0 = unet.get_unet()
    model0.load_weights(name_model)

    dice_all=np.empty([0])
    auc_all=np.empty([0])
    auprc_all=np.empty([0])

    path1='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/full_8m_anchor555/'
    new_path='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/new_arousal/'
    all_test_files=open('whole_test.dat','r')
    for filename in all_test_files:
        filename=filename.strip()
        tmp=filename.split('/')
        the_id=tmp[-1]
        print(the_id)

        index=np.array([3,6,7])
        image_ori = np.load(path1 + the_id + '.npy')[index,:]
        arousal_file = new_path + the_id + '-arousal.mat'
        arousal = import_arousals(arousal_file)

        mask=np.ones(arousal.shape)
        mask[arousal<0]=0

        d2=arousal.shape[1]

        # STFT
        _, _, image_stft = signal.stft(image_ori, window='hamming', fs=200, nperseg=256) # 3,129,65537(2**16+1)
        image_stft = np.log(np.abs(image_stft)+1e-7)

        # filter-bank
        filterbank = np.array([0,0.2,0.4,0.6,0.8,1,1,0.8,0.6,0.4,0.2,0])
        image_stft_filter=np.apply_along_axis(lambda x: np.convolve(x,filterbank,mode='same'), axis=1, arr=image_stft)
        image_stft_filter=image_stft_filter/len(filterbank) # np.convolve is sum, not mean

        # filter-bank - 20 features only
        image_stft_filter=image_stft_filter[:,np.arange(6,128,6)[0:20],:] # 3,20,65537

        # reshape - 20*3 channels
        image=np.reshape(image_stft_filter,(60,65537))[:,1:]
        image_batch=[]
        for k in np.arange(0,65520,48):
            image_batch.append(image[:,k:(k+48)].T)
        image_batch=np.array(image_batch)
#        input_pred=np.reshape(image.T,(batch,size,num_channel))
#        output = model0.predict(input_pred)
        print(image_batch)
        output = model0.predict(image_batch)
        output_new=np.reshape(output,(65520))
        #output_new=np.reshape(output,(size*batch))

        print(output_new)
        output_part1=np.repeat(output_new[np.arange(0,65520,2)],2**8)
        output_part2=np.repeat(output_new[np.arange(1,65520,2)],2**8)        

        # deal with half-overlapping
        output_final=np.zeros(65520*128+128)
        output_count=np.zeros(65520*128+128)
        output_final[:65520*128] = output_final[:65520*128] + output_part1
        output_count[:65520*128] = output_count[:65520*128] + 1
        output_final[128:] = output_final[128:] + output_part2
        output_count[128:] = output_count[128:] + 1

        output_final=output_final/output_count        
        output_all=output_final[0:d2]
        print(output_part1.shape)
        print(output_final.shape, output_all.shape)
        print(output_all)

        sum_base=np.multiply(output_all,mask).sum() + np.multiply(arousal,mask).sum()
        sum_val_cut=2*np.multiply(np.multiply(output_all,mask),np.multiply(arousal,mask)).sum()
        ratio=(float(sum_val_cut)+ss)/(float(sum_base)+ss)
        dice_all=np.concatenate((dice_all,np.reshape(ratio,(1,))))

       # eva=open('eva_global.txt','a')
        eva.write('%s' % the_id) 
        eva.write('\t%.4f' % ratio)
        eva.write('\n')
        eva.flush()
       # eva.close()

        positives, negatives = score_record(arousal.flatten(),output_all.flatten(),reso_digits)
        positives_all += positives
        negatives_all += negatives
        auc, auprc = calculate_auc(positives, negatives)
        auc_all=np.concatenate((auc_all,np.reshape(auc,(1,))))
        auprc_all=np.concatenate((auprc_all,np.reshape(auprc,(1,))))

#        auc_auprc=open('auc_auprc.txt','a')
        auc_auprc.write('%s' % the_id)
        auc_auprc.write('\t%.6f' % auc)
        auc_auprc.write('\t%.6f' % auprc)
        auc_auprc.write('\n')
        auc_auprc.flush()
#        auc_auprc.close()
        print(auc,auprc)

        if(write_vec):
            vec = open('vec/' + the_id + '.vec', 'w')
            for item in output_all:
                vec.write("%.6f\n" % item)
            vec.close()
        pass

    all_test_files.close()
    auc, auprc = calculate_auc(positives_all, negatives_all)
    auc_auprc.write('%s' % 'overall')
    auc_auprc.write('\t%.6f' % auc)
    auc_auprc.write('\t%.6f' % auprc)
    auc_auprc.write('\n')
    auc_auprc.write('%s' % 'avg_individual')
    auc_auprc.write('\t%.6f' % np.nanmean(auc_all))
    auc_auprc.write('\t%.6f' % np.nanmean(auprc_all))
    auc_auprc.write('\n')
    auc_auprc.close()

    eva.write('%s' % 'overall')
    eva.write('\t%.4f' % dice_all.mean())
    eva.write('\n')
    eva.close()





