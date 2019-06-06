from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda,BatchNormalization,LSTM,GRU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf
from keras.layers import ZeroPadding1D

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#size= 2**16
size= 32
channel=96
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
    inputs = Input((size, channel)) #2**16
    print(inputs.shape)

    conv1 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(inputs))
    conv1 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv1))

    conv1 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv1))#+8
    conv1 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv1))

    conv1 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv1))#+8
    conv1 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv1))

    pool1 = MaxPooling1D(pool_size=2)(conv1) #2**1

    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(pool1))
    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv2))

    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv2))#+8
    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv2))

    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv2))#+8
    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv2))

    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv2))#+8
    conv2 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv2))

    pool2 = MaxPooling1D(pool_size=2)(conv2) 

    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(pool2))
    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))

    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))#+8
    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))

    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))#+8
    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))

    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))#+8
    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))

    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))#+8
    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))

    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))#+8
    conv3 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv3))

    pool3 = MaxPooling1D(pool_size=2)(conv3) 

    conv4 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(pool3))#+8
    conv4 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv4))

    conv4 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv4))#+8
    conv4 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv4))

    conv4 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv4))#+8
    conv4 = BatchNormalization()(Conv1D(96, 3, activation='relu', padding='same')(conv4))

    pool4 = MaxPooling1D(pool_size=2)(conv4) 

    lstm0 = GRU(96,return_sequences=True)(pool4)

    conv22 = Conv1D(1, 1, activation='sigmoid')(lstm0) 

    model = Model(inputs=[inputs], outputs=[conv22])

    model.compile(optimizer=Adam(lr=1e-3,beta_1=0.9, beta_2=0.999,decay=1e-5), loss=crossentropy_cut, metrics=[dice_coef])

    return model


