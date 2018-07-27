""" Jay Borseth
    2018.07.25
"""

from __future__ import print_function

import os
# force to run on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import cv2
import h5py  # for saving the model
import io
import keras
import keras.backend as K
import numpy as np
from time import time
from datetime import datetime  # for filename conventions
from keras.optimizers import Adam, Adadelta
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Softmax, UpSampling2D, ZeroPadding2D, concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from generator import batch_generator
import models
import sys

#Parameters
INPUT_CHANNELS = 1
NUMBER_OF_CLASSES = 1
BATCH_SIZE = 1
VALIDATION_BATCH_SIZE = 1
SCALE_SIZE = (1920, 1920)  # W, H
epochs = 200
patience = 60

loss_name = "binary_crossentropy"


def get_model(batch_size=BATCH_SIZE, width=None, height=None):

    inputs = Input((height, width, INPUT_CHANNELS))

    base = models.get_fcn_vgg16_32s_modified(inputs, NUMBER_OF_CLASSES)

    # sigmoid
    reshape = Reshape((-1, NUMBER_OF_CLASSES))(base)
    act = Activation('sigmoid')(reshape)

    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adadelta(), loss=loss_name)

    #print(model.summary())

    return model


# Create a function to allow for different training data and other options
def train_model_batch_generator(image_dir=None,
                                label_dir=None,
                                job_dir='./tmp/semantic_segmenter',
                                model_out_name="model.h5",
                                **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    bg = batch_generator(
        image_dir,
        label_dir,
        training_split=0.8,
        scale_size=SCALE_SIZE,
        augment=True,
        rotate=True)

    model = get_model(BATCH_SIZE, width=bg.width, height=bg.height)
    checkpoint_name = 'model_weights_' + loss_name + '.h5'

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(
            checkpoint_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=0),
    ]

    history = model.fit_generator(
        bg.training_batch(BATCH_SIZE),
        validation_data=bg.validation_batch(VALIDATION_BATCH_SIZE),
        epochs=epochs,
        steps_per_epoch=bg.steps_per_epoch,
        validation_steps=1,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    # history = model.fit_generator(
    #     train_generator,
    #     validation_data=validate_generator,
    #     epochs=epochs,
    #     steps_per_epoch=48 / BATCH_SIZE,
    #     validation_steps=8 / BATCH_SIZE,
    #     verbose=1,
    #     shuffle=False,
    #     callbacks=callbacks)
    #callbacks=[tensorboard])

    # Save the model locally
    model.save(model_out_name)


def visualy_inspect_result(image_dir, label_dir):

    bg = batch_generator(image_dir, label_dir, scale_size=SCALE_SIZE)

    model = get_model(BATCH_SIZE, width=bg.width, height=bg.height)
    model.load_weights('model.h5')

    img, mask = bg.get_random_validation()

    y_pred = model.predict(img[None, ...].astype(np.float32))[0]

    print('y_pred.shape', y_pred.shape)

    y_pred = y_pred.reshape((bg.height, bg.width, NUMBER_OF_CLASSES))

    print('np.min(y_pred)', np.min(y_pred))
    print('np.max(y_pred)', np.max(y_pred))

    cv2.imshow('img', img)
    cv2.imshow('mask 1', mask)
    cv2.imshow('mask object 1', y_pred[:, :, 0])
    cv2.waitKey(0)


def make_predition_movie(image_dir, label_dir):
    ''' make a movie of both training and validation data with the mask as a red overlay '''

    bg = batch_generator(
        image_dir,
        label_dir,
        scale_size=SCALE_SIZE,
        augment=True,
        rotate=False)

    model = get_model(BATCH_SIZE, width=bg.width, height=bg.height)

    model.load_weights('model.h5')

    vid_name = "crack_detect.mp4"

    video_scale = 4
    vid_scaled = (int(bg.width / video_scale), int(bg.height / video_scale))

    vid_fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out = cv2.VideoWriter(vid_name, vid_fourcc, 30.0, vid_scaled)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index in range(bg.image_count):
        img, mask8 = bg.get_image_and_mask(index, source='all')
        img = img.reshape((bg.height, bg.width, NUMBER_OF_CLASSES))

        y_pred = model.predict(img[None, ...].astype(np.float32))[0]
        y_pred = y_pred.reshape((bg.height, bg.width, NUMBER_OF_CLASSES))

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = np.zeros_like(img)
        mask[:, :, 2] = y_pred[:, :, 0] * 255

        alpha = 0.25
        cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)

        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        cv2.putText(img, bg.images[index], (22, 22), font, 0.3,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, bg.images[index], (20, 20), font, 0.3, (255, 255, 255), 1,
                    cv2.LINE_AA)

        vid_out.write(img)
        #cv2.imshow('img',img)
        # key = cv2.waitKey(0)
        # if key == 27: # esc
        #     break
    #cv2.destroyAllWindows()
    vid_out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-dir',
        default=r"..\data\crack_detect\original\crack_images\*.tif",
        help='local path to image data')
    parser.add_argument(
        '--label-dir',
        default=r"..\data\crack_detect\original\crack_annotations\*.png",
        help='local path to label data')
    parser.add_argument(
        '--job-dir',
        default='tmp',
        help='Cloud storage bucket to export the model and store temp files')

    args = parser.parse_args()
    arguments = args.__dict__

    train_model_batch_generator(**arguments)
    make_predition_movie(args.image_dir, args.label_dir)
