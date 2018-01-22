import pandas as pd
import numpy as np

stores_df = pd.read_csv('data/stores.csv')
##
holiday_df = pd.read_csv('data/holidays_events.csv')
# can
oil_df = pd.read_csv('data/oil.csv')
items_df = pd.read_csv('data/items.csv')

# train_df = pd.read_csv('data/train.csv')
# on promotion - a lot of null - might be predictive - need to count -> most of data
test_df = pd.read_csv('data/test.csv')
# get 1/5 of the training data
train_df_sub = pd.read_csv('data/train.csv', nrows=2000000)


# dataframe for timestamp
timestamp_df = pd.DataFrame(np.unique(train_df_sub['date']), columns=['date'])
# combine with holiday info
# check
pd.isnull(oil_df).any

holiday17 = holiday_df[holiday_df['timestamp'].dt.year==2017]

