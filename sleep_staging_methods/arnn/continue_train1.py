from __future__ import print_function

import os
import sys
from skimage.transform import resize
from skimage.io import imsave
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
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

batch_size=1
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

import unet1
import random
model = unet1.get_unet()
model.load_weights('weights_' + sys.argv[1] + '.h5')
#model.summary()

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

            index1=np.arange(7)
            np.random.shuffle(index1)
            index2=np.arange(2)+8
            np.random.shuffle(index2)
            index3=np.array([7,10,12])
            index=np.concatenate((index1[0:1],index2[0:1],index3))

            image = np.load(path1 + the_id + '.npy')[index,:]
            label = np.load(path2 + the_id + '.npy')

#            rrr=random.random()
#            if (rrr>0.5):
#                image = np.load(path3 + the_id + '.npy')[0:11,:]
#                label = np.load(path4 + the_id + '.npy')
#            else:
#                image = np.load(path1 + the_id + '.npy')[0:11,:]
#                label = np.load(path2 + the_id + '.npy')

            if (if_train==1):
                rrr=random.random()
                rrr_scale=rrr*(max_scale-min_scale)+min_scale
                rrr=random.random()
                rrr_mag=rrr*(max_mag-min_mag)+min_mag
                rrr_flip=random.random()
                if(if_time):
                    image=scaleImage(image,rrr_scale)
                    label=scaleImage(label,rrr_scale)
                if(if_mag):
                    image=image*rrr_mag
                if(if_flip & (rrr_flip>0.5)):
                    image=cv2.flip(image,1)
                    label=cv2.flip(label,1)
                shift=int(random.random()*1500000)
                image=np.roll(image,shift,axis=1)
                label=np.roll(label,shift,axis=1)

            image_batch.append(image.T)
            label_batch.append(label.T)

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
    steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=25,
    validation_data=generate_data(test_line,batch_size,False),
    validation_steps=int(len(test_line) // batch_size),callbacks=callbacks)

