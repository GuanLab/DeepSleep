from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda,BatchNormalization,LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf
from keras.layers import ZeroPadding1D

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size= 4096*1024
channel=5
batch_size=32
ss=10


def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
#    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    mask=K.greater_equal(y_true_f,-0.5)
#    out = -(y_true_f * K.log(y_pred_f)*mask + (1.0 - y_true_f) * K.log(1.0 - y_pred_f)*mask)
#    out=K.mean(out)
    losses = -(y_true_f * K.log(y_pred_f) + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    losses = tf.boolean_mask(losses, mask)
    masked_loss = tf.reduce_mean(losses)
    return masked_loss

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((size, channel)) #4096*1024
    print(inputs.shape)

    conv01 = BatchNormalization()(Conv1D(15, 7, activation='relu', padding='same')(inputs))
    conv01 = BatchNormalization()(Conv1D(15, 7, activation='relu', padding='same')(conv01))
    pool01 = MaxPooling1D(pool_size=2)(conv01) #4096*512

    conv0 = BatchNormalization()(Conv1D(18, 7, activation='relu', padding='same')(pool01))#+8
    conv0 = BatchNormalization()(Conv1D(18, 7, activation='relu', padding='same')(conv0))
    pool0 = MaxPooling1D(pool_size=4)(conv0) #4096*128

    conv1 = BatchNormalization()(Conv1D(21, 7, activation='relu', padding='same')(pool0))#+8
    conv1 = BatchNormalization()(Conv1D(21, 7, activation='relu', padding='same')(conv1))
    pool1 = MaxPooling1D(pool_size=4)(conv1) #4096*32

#    conv2 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(pool1))#+16
#    conv2 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(conv2))
#    pool2 = MaxPooling1D(pool_size=2)(conv2) #4096*8

    conv3 = BatchNormalization()(Conv1D(30, 7, activation='relu', padding='same')(pool1))#+16
    conv3 = BatchNormalization()(Conv1D(30, 7, activation='relu', padding='same')(conv3))
    pool3 = MaxPooling1D(pool_size=4)(conv3) #4096*8

#    conv4 = BatchNormalization()(Conv1D(112, 7, activation='relu', padding='same')(pool3))#+32
#    conv4 = BatchNormalization()(Conv1D(112, 7, activation='relu', padding='same')(conv4))
#    pool4 = MaxPooling1D(pool_size=2)(conv4) #4096*2

    conv5 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(pool3))#+32
    conv5 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(conv5))
    pool5 = MaxPooling1D(pool_size=4)(conv5) #4096*2

#    conv6 = BatchNormalization()(Conv1D(208, 7, activation='relu', padding='same')(pool5))#+64
#    conv6 = BatchNormalization()(Conv1D(208, 7, activation='relu', padding='same')(conv6))
#    pool6 = MaxPooling1D(pool_size=2)(conv6) #2048

    conv7 = BatchNormalization()(Conv1D(120, 7, activation='relu', padding='same')(pool5))#+64 
    conv7 = BatchNormalization()(Conv1D(120, 7, activation='relu', padding='same')(conv7))
    pool7 = MaxPooling1D(pool_size=4)(conv7) #2048

#    conv8 = BatchNormalization()(Conv1D(400, 7, activation='relu', padding='same')(pool7))#+128
#    conv8 = BatchNormalization()(Conv1D(400, 7, activation='relu', padding='same')(conv8))
#    pool8 = MaxPooling1D(pool_size=2)(conv8) #512

    conv9 = BatchNormalization()(Conv1D(240, 7, activation='relu', padding='same')(pool7))#+128
    conv9 = BatchNormalization()(Conv1D(240, 7, activation='relu', padding='same')(conv9))
    pool9 = MaxPooling1D(pool_size=4)(conv9) #512

    conv10 = BatchNormalization()(Conv1D(480, 7, activation='relu', padding='same')(pool9))#+496
    conv10 = BatchNormalization()(Conv1D(480, 7, activation='relu', padding='same')(conv10))
    #lstm0 = CuDNNLSTM(1900,return_sequences=True)(conv10)

    up11 = concatenate([Conv1DTranspose(conv10,240, 4, strides=4, padding='same'), conv9], axis=2)
    conv11 = BatchNormalization()(Conv1D(240, 7, activation='relu', padding='same')(up11))
    conv11 = BatchNormalization()(Conv1D(240, 7, activation='relu', padding='same')(conv11)) #1024

#    up12 = concatenate([Conv1DTranspose(conv11,400, 2, strides=2, padding='same'), conv8], axis=2)
#    conv12 = BatchNormalization()(Conv1D(400, 7, activation='relu', padding='same')(up12))
#    conv12 = BatchNormalization()(Conv1D(400, 7, activation='relu', padding='same')(conv12)) #1024

    up13 = concatenate([Conv1DTranspose(conv11,120, 4, strides=4, padding='same'), conv7], axis=2)
    conv13 = BatchNormalization()(Conv1D(120, 7, activation='relu', padding='same')(up13))
    conv13 = BatchNormalization()(Conv1D(120, 7, activation='relu', padding='same')(conv13)) #4096

#    up14 = concatenate([Conv1DTranspose(conv13,208, 2, strides=2, padding='same'), conv6], axis=2)
#    conv14 = BatchNormalization()(Conv1D(208, 7, activation='relu', padding='same')(up14))
#    conv14 = BatchNormalization()(Conv1D(208, 7, activation='relu', padding='same')(conv14)) #4096

    up15 = concatenate([Conv1DTranspose(conv13,60, 4, strides=4, padding='same'), conv5], axis=2)
    conv15 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(up15))
    conv15 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(conv15)) #4096*4

#    up16 = concatenate([Conv1DTranspose(conv15,112, 2, strides=2, padding='same'), conv4], axis=2)
#    conv16 = BatchNormalization()(Conv1D(112, 7, activation='relu', padding='same')(up16))
#    conv16 = BatchNormalization()(Conv1D(112, 7, activation='relu', padding='same')(conv16)) #4096*4

    up17 = concatenate([Conv1DTranspose(conv15,30, 4, strides=4, padding='same'), conv3], axis=2)
    conv17 = BatchNormalization()(Conv1D(30, 7, activation='relu', padding='same')(up17))
    conv17 = BatchNormalization()(Conv1D(30, 7, activation='relu', padding='same')(conv17)) #4096*16

#    up18 = concatenate([Conv1DTranspose(conv17,60, 2, strides=2, padding='same'), conv2], axis=2)
#    conv18 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(up18))
#    conv18 = BatchNormalization()(Conv1D(60, 7, activation='relu', padding='same')(conv18)) #4096*16
    
    up19 = concatenate([Conv1DTranspose(conv17,21, 4, strides=4, padding='same'), conv1], axis=2)
    conv19 = BatchNormalization()(Conv1D(21, 7, activation='relu', padding='same')(up19))
    conv19 = BatchNormalization()(Conv1D(21, 7, activation='relu', padding='same')(conv19)) #4096*64

    up20 = concatenate([Conv1DTranspose(conv19,18, 4, strides=4, padding='same'), conv0], axis=2)
    conv20 = BatchNormalization()(Conv1D(18, 7, activation='relu', padding='same')(up20))
    conv20 = BatchNormalization()(Conv1D(18, 7, activation='relu', padding='same')(conv20)) #4096*64

    up21 = concatenate([Conv1DTranspose(conv20,15, 2, strides=2, padding='same'), conv01], axis=2)
    conv21 = BatchNormalization()(Conv1D(15, 7, activation='relu', padding='same')(up21))
    conv21 = BatchNormalization()(Conv1D(15, 7, activation='relu', padding='same')(conv21)) #4096*256

    conv22 = Conv1D(1, 1, activation='sigmoid')(conv21) 

    model = Model(inputs=[inputs], outputs=[conv22])

    model.compile(optimizer=Adam(lr=1e-4,beta_1=0.9, beta_2=0.999,decay=1e-5), loss=crossentropy_cut, metrics=[dice_coef])

    return model


