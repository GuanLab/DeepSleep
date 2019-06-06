from __future__ import print_function

import os
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import cv2
import scipy.io
from scipy import signal
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

batch_size=5
ss = 10

def scaleImage (image,scale):
    [x,y]= image.shape
    x1=x
    y1=int(round(y*scale))
    image=cv2.resize(image.astype('float32'),(y1,x1)) # check this for multiple channnels!!
    new=np.zeros((x,y))
    if (y1>y):
        start=int(round(y1/2-y/2))
        end=start+y
        new=image[:,start:end]
    else:
        new_start=int(round(y-y1)/2)
        new_end=new_start+y1
        new[:,new_start:new_end]=image
    return new

def label_major_vote(input_data,scale_pool):
    size_new=int(input_data.shape[1]/scale_pool)
    input_data=input_data.reshape(size_new,scale_pool).T
    input_data=input_data.astype(int) + 1 # bincount need non-negative, int dtype
    counts=np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=0, arr=input_data)
    major=np.apply_along_axis(lambda x: np.argmax(x), axis=0, arr=counts) - 1 
    major=major.reshape(1,len(major))
    return major

import unet
import random
model = unet.get_unet()
#model.load_weights('weights_' + sys.argv[1] + '.h5')
model.summary()

from datetime import datetime
import random
path1='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/full_8m_anchor555/'
path2='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/full_8m_label/'
new_path='/fs/project/PAS1294/osu10386/hongyang/2018/physionet/data/new_arousal/'
all_ids=open('whole_train.dat','r')
all_line=[]
for line in all_ids:
    all_line.append(line.rstrip())
all_ids.close()

#random.seed(datetime.now())
random.seed(int(sys.argv[1]))
random.shuffle(all_line)
partition_ratio=0.8
train_line=all_line[0:int(len(all_line)*partition_ratio)]
test_line=all_line[int(len(all_line)*partition_ratio):len(all_line)]
random.seed(datetime.now())

def generate_data(train_line, batch_size, if_train):
    """Replaces Keras' native ImageDataGenerator."""
##### augmentation parameters ######
    if_time=False
    max_scale=1.15
    min_scale=1
    if_mag=True
    max_mag=1.15
    min_mag=0.9
    if_flip=False
####################################
    i = 0
    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(train_line):
                i = 0
                random.shuffle(train_line)
            sample = train_line[i]
            i += 1
            the_id=sample.split('/')[-1]

            index=np.array([3,6,7])
            image = np.load(path1 + the_id + '.npy')[index,:]
            label = np.load(path2 + the_id + '.npy')

            # STFT
            _, _, image_stft = signal.stft(image, window='hamming', fs=200, nperseg=256) # 3,129,65537(2**16+1)
            image_stft = np.log(np.abs(image_stft)+1e-7) # np.log(0) results in -inf..

            # filter-bank
            filterbank = np.array([0,0.2,0.4,0.6,0.8,1,1,0.8,0.6,0.4,0.2,0])
            image_stft_filter=np.apply_along_axis(lambda x: np.convolve(x,filterbank,mode='same'), axis=1, arr=image_stft)
            image_stft_filter=image_stft_filter/len(filterbank) # np.convolve is sum, not mean
            
            # filter-bank - 20 features only
            image_stft_filter=image_stft_filter[:,np.arange(6,128,6)[0:20],:] # 3,20,65537

            # reshape - 20*3 channels
            image_final=np.reshape(image_stft_filter,(60,65537))[:,1:]
            tmp=np.repeat(label_major_vote(label,2**7),2)
            tmp=np.concatenate((tmp[1:],tmp[:1]))[:131040]
            label_final=label_major_vote(tmp.reshape((1,len(tmp))),2*48)

            #index=np.arange(0,65520,48)
            num_seq=10
            index=np.arange(0,1360,num_seq) #65280/48=1360
            random.shuffle(index)
            for k in index: # 48 -> 30-second
                if np.sum(label_final[:,k:(k+num_seq)]!=-1) > 0:
                    image_batch.append(image_final[:,(k*48):(k*48+num_seq*48)].T) # dim 60*480
                    label_batch.append(label_final[:,k:(k+num_seq)].T) # dim 1*10

        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
#        print(image_batch.shape,label_batch.shape)
        yield image_batch, label_batch

#model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=False)
name_model='weights_' + sys.argv[1] + '.h5'
callbacks = [
#    keras.callbacks.TensorBoard(log_dir='./',
#    histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    verbose=0,save_weights_only=False,monitor='val_loss')
    #verbose=0,save_weights_only=False,monitor='val_loss',save_best_only=True)
    ]

model.fit_generator(
    generate_data(train_line, batch_size,True),
    steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=5,
    validation_data=generate_data(test_line,batch_size,False),
    validation_steps=int(len(test_line) // batch_size),callbacks=callbacks)

