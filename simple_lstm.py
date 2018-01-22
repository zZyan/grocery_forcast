from __future__ import division
from numpy import random as nprand
import sys
import os

import pandas as pd 
import time

from keras.models import Model
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Bidirectional, LSTM, Merge
from keras.layers import Concatenate, Reshape

from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import keras.backend as K

import numba

from model_lstm import LSTMModel


def read_all():
	# read main data for sales
	df = pd.read_feather('data/train_pivot.feat')

	uid = df.uid
	df.drop('uid', axis=1, inplace = True)
	df.fillna(method='ffill', axis=1, inplace=True)
	# df.uid = uid

	# for i in range(df)

	# process additional features 
	store_df = pd.read_csv('data/stores.csv')
	store_df = store_df[['store_nbr', 'type', 'cluster']]

	item_df = pd.read_csv('data/items.csv')


	store_nbr = [int(i.split('_')[0]) for i in uid]
	item_nbr = [int(i.split('_')[1]) for i in uid]


	df['item_nbr'] = item_nbr
	df['store_nbr'] = store_nbr

	df = df.merge(store_df, on = 'store_nbr', how = 'left')
	df = df.merge(item_df, on = 'item_nbr', how = 'left')

	return df



def get_encoded_features(df, features):
	'''
	param: df: dataframe 
	param: featurese: list of features to be one-hot encoded
	return: feature_df: one hot encoded features 
	'''

	def one_hot_encode(df, column):
	    one_hot = pd.get_dummies(uid_df[feature], drop_first=False)
	    return (one_hot - one_hot.mean()) / one_hot.std()

	uid_df = df[features]

	# encode feather 
	for feature in features:
		one_hot = one_hot_encode(uid_df, feature)

		name = [feature + '_'+ str(c) for c in one_hot.columns]
		one_hot.columns = name

		uid_df = pd.concat([uid_df, one_hot], axis = 1)
		uid_df.drop(feature, axis = 1, inplace = True)

	return uid_df


@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


@numba.jit(nopython=True)
def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len/lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_365 = single_autocorr(series, lag)
            c_364 = single_autocorr(series, lag-1)
            c_366 = single_autocorr(series, lag+1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.NaN
    return corr #, support


# raw_year_autocorr = batch_autocorr(df.values, 365, starts, ends, 1.5, args.corr_backoffset)
# year_unknown_pct = np.sum(np.isnan(raw_year_autocorr))/len(raw_year_autocorr)  # type: float

# # Quarterly autocorrelation
# raw_quarter_autocorr = batch_autocorr(df.values, int(round(365.25/4)), starts, ends, 2, args.corr_backoffset)
# quarter_unknown_pct = np.sum(np.isnan(raw_quarter_autocorr)) / len(raw_quarter_autocorr)  # type: float



# train_span = 365
# # timesteps = 20
# validate_span = 20
# test_span = 10



import random

def generate_batch(data, features, num_items = 1, encode_span=100, predict_span=20, timesteps = 5):
	# random.seed(step_no)
	# data = data.squeeze()

	X_batch_concat = []
	y_batch_concat = []
	feature_batch_concat = []

	time_limit = data.shape[0]-encode_span-predict_span-timesteps


	while len(X_batch_concat) < num_items:
		# print (n)
		if time_limit > 0:
			start_time = random.randrange(time_limit)
		else:
			start_time = 0
		
		item_ind = random.randrange(data.shape[1])

		# print('item: ', item_ind, ' start_time:', start_time)

		feature_batch = features.iloc[item_ind, :].values.reshape(1, -1)
		feature_batch  = np.tile(feature_batch, (encode_span, 1))
		# K.tile(feature_batch, [encode_span, 1])
		feature_batch_concat.append(feature_batch)

        # start_time = 0
		end_time = start_time + encode_span

		shifted_train = []

		for i in range(timesteps + predict_span):
			series = data[start_time:end_time, item_ind,:]
			shifted_train.append(np.concatenate(series, axis=0))
			start_time += 1
			end_time += 1

		shifted_train = np.array(shifted_train)
		shifted_train = shifted_train.transpose()


		X_batch = shifted_train[:,:-predict_span]
		# X_batch = X_batch.reshape(X_batch.shape + (1,))
		y_batch = shifted_train[:, -predict_span:]

		X_batch_concat.append(X_batch)
		y_batch_concat.append(y_batch)

		start_time = 0
		end_time = 0

	X_batch_concat = np.concatenate(X_batch_concat)
	y_batch_concat = np.concatenate(y_batch_concat)
	feature_batch_concat = np.concatenate(feature_batch_concat)

	X_batch_concat = X_batch_concat.reshape(X_batch_concat.shape + (1,))


	return X_batch_concat, feature_batch_concat, y_batch_concat


# def generate_validation_dict(validate, features, validate_sample = 100, encode_span=100, predict_span=20, timesteps = 5):
# 	num_items = features.shape[0]

# 	loss = 0

# 	X_dict = dict()
# 	feature_dict = dict()
# 	y_dict = dict()

# 	for s in range(validate_sample):
# 		item_ind = random.randrange(num_items)

# 		feature_batch = features.iloc[item_ind, :].values.reshape(1, -1)
# 		feature_batch  = np.tile(feature_batch, (encode_span, 1))

# 		shifted_val = []
# 		start_time = 0
# 		end_time = encode_span

# 		for i in range(timesteps + predict_span):
# 			series = validate[start_time:end_time, item_ind,:]
# 			shifted_val.append(np.concatenate(series, axis=0))
# 			start_time += 1
# 			end_time += 1
		
# 		shifted_val = np.array(shifted_val)
# 		shifted_val = shifted_val.transpose()

# 		X_batch = shifted_val[:,:-predict_span]
# 		X_batch = X_batch.reshape(X_batch.shape + (1,))
# 		y_batch = shifted_val[:, -predict_span:]
# 		X_dict[s] = X_batch
# 		y_dict[s] = y_batch
# 		feature_dict[s] = feature_batch

# 	return X_dict, feature_dict, y_dict



def train(learning_rate=1e-4, batch_size=50, num_epoch=50, num_steps=50000):

	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	set_session(sess)
	lstm = LSTMModel(learning_rate=1e-4, batch_size=50)
	# data_obj = DataGeneration(queue_batch=100, batch_size=batch_size, inp_dim=inp_dim, sess=sess)

	df = read_all()
	features = ['type', 'cluster', 'family','class','perishable']
	uid_df = get_encoded_features(df, features)

	df = df.drop(features, axis = 1)

	features = uid_df

	# may add sequencial features as well 
	encode_span = 100
	predict_span = 20

	train_data = df.iloc[:,range(df.shape[1]-predict_span)]
	train_data = train_data.values.transpose()


	train_data = train_data.reshape(train_data.shape + (1,))

	timesteps = 5

	# shape: [number of items, number of days]
	validate = df.iloc[:, range(train_data.shape[0]-encode_span-timesteps, train_data.shape[0] + predict_span)]
	validate = validate.values.transpose()
	validate = validate.reshape(validate.shape + (1,))

	# normalize
	for i in range(train_data.shape[0]):
		temp_mean = train_data[i, :].mean()
		temp_std = train_data[i, :].std()
		train_data[i, :] = (train_data[i, :] - temp_mean)/temp_std
		validate[i:,] = (validate[i, :] - temp_mean)/temp_std

	# num_epoch = 10
	# num_steps = 200


	for epoch in range(num_epoch):
	    lstm.start_time = time.time()

	    X_val, feature_val, y_val= generate_batch(validate, features, 100)

	    # initialize per_epoch variables
	    for batch in range(num_steps):
	        X_batch, feature_batch, y_batch = generate_batch(train_data, features)
	        # Train on single batch
	        train_metrics = lstm.model.fit([X_batch, feature_batch], y_batch)
	        if batch % 100 == 0:  ## Evaluate model on test set every 10000 batches
	        	test_metrics = lstm.model.evaluate([X_val, feature_val], y_val, verbose=0)

	        	print('avg_loss: ', test_metrics[0], ' avg_acc: ', test_metrics[1])

	            # test_metrics = lstm.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
	            # lstm.report_validation_stats(test_metrics, epoch)

	        lstm.report_training_stats(train_metrics, batch, epoch, num_steps)

	    lstm.save_model(epoch)  ## Save model
	    ### Empty input queues
	    # data_obj.empty_input_queues()

	    if lstm.stop_training(epoch) is True:
	        break  ## Stop training if loss is not decreasing


	# X_test, y_test = generate_validation_set(test_set_size, inp_dim)

	sess.close()


if __name__ == '__main__':
    model = train(learning_rate=1e-4, batch_size=10, num_epoch=50,
                  num_steps=500)

