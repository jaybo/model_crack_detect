""" Jay Borseth
    2018.07.25
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
import keras
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

import models
from generator import batch_generator


#Parameters
INPUT_CHANNELS = 1
NUMBER_OF_CLASSES = 1
NAME = "CrackDetect_{}".format(int(time()))

#HyperParameters
BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 4
OUTPUT_SIZE = (256, 256)  # (H, W)
SCALE_SIZE = (1920, 1920) # (H, W)
EPOCHS = 300
PATIENCE = 80


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

    checkpoint_name = 'model_weights_' + NAME + '.h5'

    tensorboard = keras.callbacks.TensorBoard(
        log_dir="logs/{}".format(NAME), 
        #histogram_freq=1,
        write_images=True)

    callbacks = [
        tensorboard,
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0),
        ModelCheckpoint(
            checkpoint_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=0),
    ]

    history = model.fit_generator(
        bg.training_batch(),
        validation_data=bg.validation_batch(),
        epochs=EPOCHS,
        steps_per_epoch=bg.steps_per_epoch,
        validation_steps=bg.validation_steps,
        verbose=1,
        shuffle=False,
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


def make_predition_movie(image_dir, label_dir, weights=None):
    ''' make a movie of both training and validation data with the mask as a red overlay '''

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
        model.load_weights('model.h5')

    vid_name = "crack_detect.mp4"

    video_scale = 4
    vid_scaled = (int(bg.width / video_scale), int(bg.height / video_scale))

    vid_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(vid_name, vid_fourcc, 15.0, vid_scaled)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for loops in range(1):
        for index in range(bg.image_count):

            img, mask8 = bg.get_image_and_mask(index, source='all', augment=False)
            img = img.reshape((bg.height, bg.width, NUMBER_OF_CLASSES))

            y_pred = model.predict(img[None, ...].astype(np.float32))[0]
            y_pred = y_pred.reshape((bg.height, bg.width)) # , NUMBER_OF_CLASSES))
            #print (y_pred.min(), y_pred.max(), y_pred.sum(), y_pred.shape)
            y_pred8 = y_pred * 255
            y_pred8 = y_pred8.astype(np.uint8)
            #print (y_pred8.min(), y_pred8.max(), y_pred8.sum(), y_pred8.shape)

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            mask = np.zeros_like(img)
            mask[:,:,0] = y_pred8

            # alpha = 0.25
            # img = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)

            font_scale = 0.8
            cv2.putText(img, bg.images[index], (22, 22), font, font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, bg.images[index], (20, 20), font, font_scale, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # outline contour
            # ret,thresh = cv2.threshold(y_pred8,128,255,0)

            # fill with the ground truth
            if mask8 is not None:
                alpha = 0.5
                merge = np.zeros_like(img)
                merge[:,:,0] = mask8
                img = cv2.addWeighted(merge, alpha, img, 1 - alpha, 0, img)

            # draw the prediction contour
            ret,thresh = cv2.threshold(y_pred8,128,255,0)
            im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0,255,0), 6)

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

    args = parser.parse_args()
    if args.label_dir == 'None':
        args.label_dir = None
    arguments = args.__dict__


    if args.command == 'train':
        train_model_batch_generator(**arguments)
    if args.command == 'train' or args.command == 'movie':
        make_predition_movie(args.image_dir, args.label_dir, args.weights)
