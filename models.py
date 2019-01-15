from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import numpy as np

import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Softmax, UpSampling2D, ZeroPadding2D, concatenate, add
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, SGD, Adam
from keras.initializers import glorot_normal, RandomNormal, Zeros, Constant
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.regularizers import l2
from keras import losses

import cv2

#Parameters
INPUT_CHANNELS = 1
NUMBER_OF_CLASSES = 1

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def create_weighted_binary_crossentropy(zero_weight = 0.4, one_weight=0.6):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def get_model(batch_size=8, width=None, height=None):

    inputs = Input((height, width, INPUT_CHANNELS))

    #base = get_fcn_vgg16_32s_modified(inputs, NUMBER_OF_CLASSES) # good
    #base = get_fcn_vgg16_32s_jayb(inputs, NUMBER_OF_CLASSES) # good
    #base = get_fcn_vgg16_8s_modified(inputs, NUMBER_OF_CLASSES) # bad
    #base = get_segnet_vgg16(inputs, NUMBER_OF_CLASSES)
    base = get_unet(inputs, NUMBER_OF_CLASSES)
    # base = get_simplenet(inputs, NUMBER_OF_CLASSES)

    # sigmoid
    reshape = Reshape((-1, NUMBER_OF_CLASSES))(base)
    act = Activation('sigmoid')(reshape)

    model = Model(inputs=inputs, outputs=act)

    #optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = Adadelta()
    optimizer = Adam(lr = 1e-5)

    # model.compile(optimizer=optimizer, loss=jaccard_distance_loss, metrics=['accuracy'])
    #model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizer, loss=dice_coef, metrics=['accuracy'])    
    #model.compile(optimizer=optimizer, loss=create_weighted_binary_crossentropy(), metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss=[focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])


    print(model.summary())

    return model


# def get_model(batch_size=8, width=None, height=None):

#     input_tensor = Input((height, width, INPUT_CHANNELS))

#     # create the base model


#     base_model = VGG16(input_tensor=input_tensor, weights=None, include_top=False, pooling=None)
#     # base_model = Xception(input_tensor=input_tensor, weights=None, include_top=False, pooling=None)


#     x = base_model.output

#     # Convolutional layers transfered from fully-connected layers
#     x = Conv2D(512, (7, 7), activation='relu', padding='same', name='fc1')(x)
#     x = Dropout(0.33)(x)
#     x = Conv2D(512, (1, 1), activation='relu', padding='same', name='fc2')(x)
#     x = Dropout(0.33)(x)
#     #classifying layer
#     x = Conv2D(NUMBER_OF_CLASSES, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1))(x)

#     x = Conv2DTranspose(NUMBER_OF_CLASSES, kernel_size=(64, 64), strides=(32, 32), activation='linear', padding='same')(x)

#     reshape = Reshape((-1, NUMBER_OF_CLASSES))(x)
#     predictions = Activation('sigmoid')(reshape)

#     # this is the model we will train
#     model = Model(inputs=base_model.input, outputs=predictions)

#     # first: train only the top layers (which were randomly initialized)
#     # i.e. freeze all convolutional InceptionV3 layers
#     # for layer in base_model.layers:
#     #     layer.trainable = False

#     model.compile(optimizer=Adadelta(), loss='binary_crossentropy')

#     print(model.summary())

#     return model


def get_fcn_vgg16_32s_modified(inputs, n_classes):


    # x = BatchNormalization()(inputs)

    # # Block 1
    # x = Conv2D(16, (9, 9), activation='relu', padding='same', name='block1_conv1')(x)
    # x = Conv2D(16, (9, 9), activation='relu', padding='same', name='block1_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # # Block 2
    # x = Conv2D(32, (7, 7), activation='relu', padding='same', name='block2_conv1')(x)
    # x = Conv2D(32, (7, 7), activation='relu', padding='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # x = Conv2D(256, (3, 3), activation='relu', padding="same")(x)

    # x = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(4, 4), activation='linear', padding='same')(x)

    # return x


    # ----------------------------------------------------------------------------------------------

    leak = .001
    dropout = 0.2

    x = BatchNormalization()(inputs)

    # Block 1
    x = Conv2D(32, (7, 7), activation='relu', padding='same', name='block1_conv0') (x)
    x = Conv2D(32, (7, 7), activation='relu', padding='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv0')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # x = Dropout(dropout)(x)

    # # Block 4
    # x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
    # x = LeakyReLU(alpha=leak)(x)
    # x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)
    # x = LeakyReLU(alpha=leak)(x)
    # x = Conv2D(256, (3, 3), padding='same', name='block4_conv3')(x)
    # x = LeakyReLU(alpha=leak)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # Block 5
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = BatchNormalization()(x)
    # # x = Dropout(dropout)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


    x = Conv2D(256, (3, 3), activation='relu', padding="same")(x)

    # x = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), activation='linear', padding='same')(x)
    # x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), activation='linear', padding='same')(x)
    x = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(4, 4), activation='linear', padding='same')(x)
    return x



def custom_gabor(shape, dtype=None):
    total_ker = np.zeros(shape, dtype=dtype)
    for i in range(shape[2]): # channels
        n_kernels = shape[3]
        for j in range(n_kernels): # kernels
            # gk = gabor_kernel(frequency=0.2, bandwidth=0.1)
            tmp_filter = cv2.getGaborKernel(ksize=(shape[0], shape[1]), sigma=1, theta=6.28 * j / n_kernels, lambd=0.5, gamma=0.3, psi=(3.14) * 0.5,
                           ktype=cv2.CV_64F)
            total_ker[:, :, i, j] = tmp_filter
    #         filter = []
    #         for row in tmp_filter:
    #             filter.append(np.delete(row, -1))
    #         kernels.append(filter)
    #             # gk.real
    #     total_ker.append(kernels)
    # np_tot = np.array(total_ker)
    return total_ker # np_tot

def get_fcn_vgg16_32s_jayb(inputs, n_classes):

    x = BatchNormalization()(inputs)
    #x = inputs

    reg = 0.01

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=True, kernel_initializer=custom_gabor, use_bias=True, kernel_regularizer=l2(reg), name='block1_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    ksize = 4

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    ksize = 8

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block3_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    ksize = 16

    # Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block4_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block4_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    ksize = 32

    # # Block 5
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block5_conv1')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block5_conv2')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(reg), name='block5_conv3')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # ksize = 64

    x = Conv2D(512, (3, 3), activation='relu', padding="same")(x)

    # x = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), activation='linear', padding='same')(x)

    ssize = int(ksize/2)
    x = Conv2DTranspose(n_classes, kernel_size=(ksize, ksize), strides=(ssize, ssize), activation='linear', padding='same')(x)

    return x



def get_fcn_vgg16_8s_modified(inputs, n_classes):

    x = BatchNormalization()(inputs)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = BatchNormalization()(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = BatchNormalization()(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = BatchNormalization()(x)

    block_3 = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = BatchNormalization()(x)

    block_4 = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), activation='relu', padding="same")(x)

    block_5 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(x)

    sum_1 = add([block_4, block_5])
    sum_1 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(sum_1)

    sum_2 = add([block_3, sum_1])

    x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), activation='linear', padding='same')(sum_2)

    return x


def get_unet(inputs, n_classes):

    x = BatchNormalization()(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = custom_gabor)(x)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.25)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.25)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)

    return conv10

def get_segnet_vgg16(inputs, n_classes):

    x = BatchNormalization()(inputs)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Up Block 1
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # Up Block 2
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # Up Block 3
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Up Block 4
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Up Block 5
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(n_classes, (1, 1), activation='linear', padding='same')(x)

    return x


def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    return


# https://github.com/EricAlcaide/SimpleNet-Keras/blob/master/simplenet.py
def get_simplenet(inputs, n_classes, s = 2, act = 'relu', drop = 0.2, weight_decay = 1e-2):

    x = BatchNormalization()(inputs)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal(), input_shape=x_train.shape[1:]))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 4
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    # First Maxpooling
    x = MaxPooling2D(pool_size=(2,2), strides=s)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # # First Maxpooling
    # model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    # model.add(Dropout(0.2))


    # Block 5
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 6
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 7
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    # Second Maxpooling
    x = MaxPooling2D(pool_size=(2,2), strides=s)(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # # Second Maxpooling
    # model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))


    # Block 8
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 9
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)
    # Third Maxpooling
    x = MaxPooling2D(pool_size=(2,2), strides=s)(x)

    # model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))
    # Third Maxpooling
    # model.add(MaxPooling2D(pool_size=(2,2), strides=s))


    # Block 10
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # model.add(BatchNormalization())
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 11
    x = Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal())(x)
    x = Activation(act)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal()))
    # model.add(Activation(act))
    # model.add(Dropout(0.2))

    # Block 12
    x = Conv2D(256, (1, 1), padding='same', kernel_initializer=glorot_normal())(x)
    x = Activation(act)(x)
    # Fourth Maxpooling
    x = MaxPooling2D(pool_size=(2,2), strides=s)(x)
    x = Dropout(drop)(x)

    # model.add(Conv2D(256, (1,1), padding='same', kernel_initializer=glorot_normal()))
    # model.add(Activation(act))
    # # Fourth Maxpooling
    # model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    # model.add(Dropout(0.2))


    # Block 13
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = Activation(act)(x)
    # Fifth Maxpooling
    x = MaxPooling2D(pool_size=(2,2), strides=s)(x)

    # model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # model.add(Activation(act))
    # # Fifth Maxpooling
    # model.add(MaxPooling2D(pool_size=(2,2), strides=s))

    ksize = 64
    ssize = int(ksize/2)

    x = Conv2DTranspose(n_classes, kernel_size=(ksize, ksize), strides=(ssize, ssize), activation='linear', padding='same')(x)
    return x

    # Final Classifier
    # model.add(Flatten())
    # model.add(Dense(num_classes, activation='softmax'))

    # return model
