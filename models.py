
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Softmax, UpSampling2D, ZeroPadding2D, concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model, Sequential

def get_fcn_vgg16_32s_modified(inputs, n_classes):
    
    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # x = Dropout(0.35)(x)

    # Block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Dropout(0.35)(x)

    # Block 5
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Dropout(0.35)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding="same")(x)

    #     
    x = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), activation='linear', padding='same')(x)
    
    return x
