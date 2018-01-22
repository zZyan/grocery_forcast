from __future__ import division
from numpy import random as nprand
import sys
import os
import h5py

import pandas as pd 
import time

from keras.models import Model
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Bidirectional, LSTM, Input, GRU
from keras.layers import Concatenate, Reshape, Merge, Dropout

from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import keras.backend as K

log_dir = 'logs/'
log_batch = log_dir + 'log_batch_'
log_epoch = log_dir + 'log_epoch.txt'
exp_dir = 'exps/'
save_model_path = exp_dir + 'final_nnet_'


class LSTMModel(object):
    def __init__(self, learning_rate = 1e-4, batch_size=100, num_steps=100, optimizer='adam', loss='hinge', stopping_criterion=1e-4,
        nb_of_sequental_features=1, feature_nb=275, days_predict = 10, timesteps = 5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_steps
        self.optimizer = optimizer
        self.loss = loss
        self.avg_loss = 0
        self.avg_acc = 0
        self.stopping_criterion = 1e-4

        self.feature_nb = feature_nb
        self.days_predict = days_predict
        self.nb_of_sequental_features = nb_of_sequental_features
        self.timesteps = timesteps

        self.build_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):

        sequential_input = Input(shape=(self.timesteps, self.nb_of_sequental_features))
        feature_input = Input(shape=(self.feature_nb,))
        lstm_layer = GRU(128, return_sequences=True)(sequential_input)
        lstm_layer = Dropout(0.5)(lstm_layer)
        # lstm_layer = LSTM(128, return_sequences=True)(lstm_layer)
        lstm_layer = GRU(16, return_sequences=False)(lstm_layer)

        merged = Concatenate(axis = -1)([lstm_layer, feature_input])

        blend = Dense(64, activation='relu')(merged)
        # blend = Dense(blending_units_2nd_layer, activation='relu')(blend)
        # output number of days 
        output = Dense(20)(blend)
        self.model = Model(inputs=[sequential_input, feature_input], outputs=output)

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	    # print("Model Summary")
	    # model.summary()
	    # return model


    def run_single_step(self, X_batch, feature_batch, y_batch):
       metrics = self.model.fit([X_batch, feature_batch], y_batch, epochs=1, verbose=0)
       return metrics

    def report_training_stats(self, metrics, batch_num, epoch, max_batch):
        if batch_num == 0:
            with open(log_epoch, 'a') as f:
                f.write("EPOCH #%d TRAINING - AVG LOSS : %0.4f, AVG ACC : %0.2f\n\t\tTIME TAKEN" \
                        "- %f seconds\n" % (epoch, self.avg_loss / max_batch,
                                            self.avg_acc / max_batch, time.time() - self.start_time))
            self.start_time = time.time()
            self.avg_loss = 0
            self.avg_acc = 0

        acc = metrics.history["acc"][0]
        loss = metrics.history["loss"][0]
        self.avg_loss += loss
        self.avg_acc += acc
        # Write avg_stats to file only every 100 batches
        if batch_num % 100 == 0:
            log_file = log_batch + str(epoch) + '.txt'
            with open(log_file, 'a') as f:
                f.write("AVG LOSS : %0.4f, ACC : %0.2f after training %d batches\n"
                        % ((self.avg_loss / batch_num), (self.avg_acc / batch_num), batch_num))


    def report_validation_stats(self, metrics, epoch):
        log_file = log_batch + str(epoch) + '.txt'
        with open(log_file, 'a') as f:
            f.write("TESTING - LOSS : %0.4f, ACC : %0.2f\n" % (metrics[0], metrics[1]))


    def save_model(self, epoch_num):
        full_path = save_model_path + str(epoch_num) + '.h5'
        self.model.save(full_path)

    def stop_training(self, epoch_num):
        if (epoch_num <= 5):  # Run minimum 5 epochs
            return False
        elif ((self.avg_loss[epoch_num - 1] - self.avg_loss[epoch_num]) < self.stopping_criterion):
            return True