
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

import keras.backend as K
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Softmax, UpSampling2D, ZeroPadding2D, concatenate, add
from keras.layers import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, SGD
from keras.layers import LeakyReLU
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception

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


def get_model(batch_size=8, width=None, height=None):

    inputs = Input((height, width, INPUT_CHANNELS))

    base = get_fcn_vgg16_32s_modified(inputs, NUMBER_OF_CLASSES) # good
    #base = get_fcn_vgg16_8s_modified(inputs, NUMBER_OF_CLASSES) # bad
    #base = get_segnet_vgg16(inputs, NUMBER_OF_CLASSES)
    #base = get_unet(inputs, NUMBER_OF_CLASSES)

    # sigmoid
    reshape = Reshape((-1, NUMBER_OF_CLASSES))(base)
    act = Activation('sigmoid')(reshape)

    model = Model(inputs=inputs, outputs=act)

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = Adadelta()

    model.compile(optimizer=optimizer, loss=jaccard_distance_loss, metrics=['accuracy'])

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
    dropout = 0.25
    x = BatchNormalization()(inputs)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)
    x = LeakyReLU(alpha=leak)(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = LeakyReLU(alpha=leak)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


    # Block 2
    x = Conv2D(64, (5, 5), padding='same', name='block2_conv1')(x)
    x = LeakyReLU(alpha=leak)(x)
    x = Conv2D(64, (5, 5), padding='same', name='block2_conv2')(x)
    x = LeakyReLU(alpha=leak)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(128, (9, 9), padding='same', name='block3_conv1')(x)
    x = LeakyReLU(alpha=leak)(x)
    x = Conv2D(128, (9, 9), padding='same', name='block3_conv2')(x)
    # x = LeakyReLU(alpha=leak)(x)
    # x = Conv2D(128, (3, 3), padding='same', name='block3_conv3')(x)
    x = LeakyReLU(alpha=leak)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

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
    x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), activation='linear', padding='same')(x)

    return x






    # ------------------------------------------------------------------------------------

    # x = BatchNormalization()(inputs)

    # # Block 1
    # x = Conv2D(32, (9, 9), activation='relu', padding='same', name='block1_conv1')(inputs)
    # x = Conv2D(32, (9, 9), activation='relu', padding='same', name='block1_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # # Block 2
    # x = Conv2D(64, (7, 7), activation='relu', padding='same', name='block2_conv1')(x)
    # x = Conv2D(64, (7, 7), activation='relu', padding='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # # Block 3
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # Block 4
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # Block 5
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(256, (3, 3), activation='relu', padding="same")(x)

    # x = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), activation='linear', padding='same')(x)

    # return x

def get_fcn_vgg16_8s_modified(inputs, n_classes):

    x = BatchNormalization()(inputs)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    block_3 = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    block_4 = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding="same")(x)

    block_5 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(x)

    sum_1 = add([block_4, block_5])
    sum_1 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(sum_1)

    sum_2 = add([block_3, sum_1])

    x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), activation='linear', padding='same')(sum_2)

    return x


def get_unet(inputs, n_classes):

    x = BatchNormalization()(inputs)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='linear')(conv9)

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