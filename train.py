""" Jay Borseth
    2018.07.25

    tensorboard --logdir=logs
"""

from __future__ import print_function

import argparse
import io
import os
import sys
from datetime import datetime  # for filename conventions
from time import time   

# force to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import cv2
import h5py
#import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf
from tensorflow import keras
import tensorflow_core.python.keras.backend as K
from tensorflow_core.python.keras.optimizers import Adadelta, SGD, Adam
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import random as rn

# repeatable
np.random.seed(42)
rn.seed(12345)
#tf.set_random_seed(1234)

import models
from generator import batch_generator

#print(tf.VERSION)
print(tf.keras.__version__)



# 2018.12.30 added three all white image to data ..._0_21_106.tif

#Parameters
INPUT_CHANNELS = 1
NUMBER_OF_CLASSES = 1
NAME = "CrackDetect_{}".format(datetime.now().isoformat(timespec='seconds')).replace(':', '-')

#HyperParameters
BATCH_SIZE = 1
VALIDATION_BATCH_SIZE = 8 # use all validation images if None
OUTPUT_SIZE = (256, 256)  # (H, W)
SCALE_SIZE = (1920, 1920) # (H, W)
EPOCHS = 500
PATIENCE = 100
CLASS_WEIGHT = {0: 95.0, 1: 5.0}

# Create a function to allow for different training data and other options
def train_model_batch_generator(image_dir=None,
                                label_dir=None,
                                model_out_name="model.h5",
                                **args):

    bg = batch_generator(
        image_dir,
        label_dir,
        batch_size= BATCH_SIZE,
        validation_batch_size=VALIDATION_BATCH_SIZE,
        training_split=0.8,
        output_size=OUTPUT_SIZE, # (H, W)
        scale_size=SCALE_SIZE, # (H, W)
        include_only_crops_with_mask = True,
        augment=True)

    model = models.get_model(BATCH_SIZE, width=OUTPUT_SIZE[1], height=OUTPUT_SIZE[0])

    # frozen_layer = model.get_layer('block1_conv0')

    checkpoint_name = NAME + '.h5'

    # class LearningRateTracker(keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs={}):
    #         optimizer = self.model.optimizer
    #         lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
    #         print('\nLR: {:.6f}\n'.format(lr))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=0.0001)

    def unfreeze(model, epoch):
        if epoch == 20:
            model.layers[2].trainable = True
            optimizer = Adam(lr = 1e-4)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    unfreeze_layers = keras.callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, logs: unfreeze(model, epoch)
          
    )

    class LRTensorBoard(TensorBoard):
        ''' add the learning rate to tensorboard '''
        def __init__(self, log_dir, histogram_freq=1, write_grads=True, write_images=True):  # add other arguments to __init__ if you need
            super().__init__(log_dir=log_dir, histogram_freq=1, write_grads=True, write_images=True)

        def on_epoch_end(self, epoch, logs=None):
            logs.update({'lr': K.eval(self.model.optimizer.lr)})
            super().on_epoch_end(epoch, logs)
            
    tensorboard = TensorBoard(
        log_dir="logs\\{}".format(NAME),
        histogram_freq=1,
        write_grads=True,
        write_images=False
        )

    callbacks = [
        tensorboard,
        reduce_lr,
        #unfreeze_layers,
        # LearningRateTracker(),
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0),
        ModelCheckpoint(
            checkpoint_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=0),
    ]

    X_test, Y_test = bg.validation_batch()

    history = model.fit_generator(
        bg.training_batch(),
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        steps_per_epoch=bg.steps_per_epoch,
        validation_steps=bg.validation_steps,
        verbose=1,
        shuffle=False,
        # class_weight=CLASS_WEIGHT,
        callbacks=callbacks)

    # Save the model locally
    model.save(model_out_name)


def visualy_inspect_result(image_dir, label_dir):

    bg = batch_generator(image_dir, label_dir,
        batch_size=1,
        validation_batch_size=1,
        output_size=OUTPUT_SIZE)

    model = models.get_model(BATCH_SIZE, width=bg.width, height=bg.height)
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


def make_predition_movie(image_dir, label_dir, weights=None, fraction_cracks=None):
    ''' make a movie of predictions.
    if label_dir != None, draw ground truth as blue,
    if min_crack_fraction != None, only add frames to movie if cracks > fraction.
    '''

    # bg = batch_generator(
    #     image_dir,
    #     label_dir,
    #     batch_size=1,
    #     validation_batch_size=1,
    #     output_size=OUTPUT_SIZE, # (H, W)
    #     scale_size=SCALE_SIZE, # (H, W)
    #     include_only_crops_with_mask = True,
    #     augment=True)


    bg = batch_generator(
        image_dir,
        label_dir,
        batch_size=1,
        validation_batch_size=1,
        output_size=(1920, 1920), # (H, W)
        scale_size=None, # (H, W)
        include_only_crops_with_mask = False,
        augment=False)

    model = models.get_model(1, width=bg.width, height=bg.height)

    if weights:
        model.load_weights(weights)
    else:
        model.load_weights(NAME + ".h5")

    vid_name = NAME + ".mp4"

    video_scale = 4
    vid_scaled = (int(bg.width / video_scale), int(bg.height / video_scale))

    vid_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(vid_name, vid_fourcc, 2.0, vid_scaled)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for loops in range(1):
        for index in range(bg.image_count):

            img, mask8 = bg.get_image_and_mask(index, source='all', augment=False)
            img = img.reshape((bg.height, bg.width, NUMBER_OF_CLASSES))

            y_pred = model.predict(img[None, ...].astype(np.float32))[0]
            y_pred = y_pred.reshape((bg.height, bg.width)) # , NUMBER_OF_CLASSES))
            #print (y_pred.min(), y_pred.max(), y_pred.sum(), y_pred.shape)
            
            if fraction_cracks is not None:
                fraction = np.count_nonzero(y_pred > 0.5)
                fraction /= (bg.height * bg.width)
                if fraction < fraction_cracks:
                    continue



            y_pred8 = y_pred * 255 
            y_pred8 = y_pred8.astype(np.uint8)
            #print (y_pred8.min(), y_pred8.max(), y_pred8.sum(), y_pred8.shape)

            img = img[:,:,0].astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # mask = np.zeros_like(img)
            # mask[:,:,0] = y_pred8

            # alpha = 0.25
            # img = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)

            # fill with the ground truth
            if mask8 is not None:
                mask = np.zeros_like(img)
                mask[:,:,2] = mask8
                alpha = 0.25
                cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)

                # notmask8 = cv2.bitwise_not(mask8)
                # img[:,:,0] = cv2.bitwise_or(img[:,:,0], mask8)
                # img[:,:,1] = cv2.bitwise_and(img[:,:,1], notmask8)
                # img[:,:,2] = cv2.bitwise_and(img[:,:,2], notmask8)


            # draw the prediction contour
            ret,thresh = cv2.threshold(y_pred8,127,255,0)
            im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0,255,0), 8)

            font_scale = 1.6
            fname = os.path.basename(bg.images[index])
            cv2.putText(img, fname, (20, 44), font, font_scale,
                        (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(img, fname, (20, 40), font, font_scale, (255, 255, 255), 6,
                        cv2.LINE_AA)

            if video_scale != 1:
                img = cv2.resize(img, vid_scaled)

            vid_out.write(img)
            if index % 10 == 0:
                print (index)

            # cv2.imshow('img',img)
            # key = cv2.waitKey(0)
            # if key == 27: # esc
            #     break

    #cv2.destroyAllWindows()
    vid_out.release()

class RunningStats:

    # https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())


def calc_mean(image_dir, label_dir, weights=None, fraction_cracks=None, loops = 100):
    ''' calc mean and stddev for the dataset.'''

    bg = batch_generator(
        image_dir,
        label_dir,
        batch_size=1,
        validation_batch_size=1,
        output_size=(1920, 1920), # (H, W)
        scale_size=None, # (H, W)
        include_only_crops_with_mask = False,
        augment=True)

    stats = RunningStats()

    for loop in range(loops):
        for index in range(bg.image_count):
            img, mask8 = bg.get_image_and_mask(index, source='all', augment=False)
            mean = np.mean(img)
            stats.push(mean)

    print (stats.mean(), stats.standard_deviation())

    # 187.4111992431641 7.840043441207606 (10, augment)
    # 187.41119924316425 7.836514274501137 (100, augment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--command',
        default="movie",
        help='train or movie')
    parser.add_argument(
        '-i', '--image_dir',
        default=r"../data/crack_detect/original/crack_images/*.tif",
        help='local path to image data')
    parser.add_argument(
        '-l', '--label_dir',
        default=r"../data/crack_detect/original/crack_annotations/*.png",
        help='local path to label data')
    parser.add_argument(
        '-w', '--weights',
        default=None,
        help='weights file')
    parser.add_argument(
        '-f', '--fraction_cracks',
        default=None,
        help='fraction of image labeled crack to be included in movie')

    args = parser.parse_args()
    if args.label_dir == 'None':
        args.label_dir = None
    if args.fraction_cracks != None:
        args.fraction_cracks = float(args.fraction_cracks)
    arguments = args.__dict__


    if args.command == 'train':
        train_model_batch_generator(**arguments)
    if args.command == 'train' or args.command == 'movie':
        make_predition_movie(args.image_dir, args.label_dir, args.weights, args.fraction_cracks)
    if args.command == 'calc_mean':
        calc_mean(args.image_dir, args.label_dir, args.weights, args.fraction_cracks)
