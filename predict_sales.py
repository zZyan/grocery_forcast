import pandas as pd
import numpy as np
import sklearn
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from collections import Counter

# %matplotlib inline


### Load training data
## CAUTION!! : load full data
# load complete train data
types_dict = {'id': 'int32',
             'item_nbr': 'int32',
             'store_nbr': 'int8',
             'unit_sales': 'float32',
             'onpromotion': 'bool'}

train_df_full = pd.read_csv('data/train.csv',low_memory=True,  parse_dates=['date'],dtype=types_dict, nrows=10000000)

## save in feather for easy reload 
# train_df.to_feather('data/train.feat')
# train_df = pd.read_feather('data/train.feat')

train_df_sub = train_df_ori[train_df_ori['month']==8]
validate_df = train_df_sub[train_df_sub.year==2017]
train_df = train_df_sub[(train_df_sub['year']==2016)]


holiday_df = pd.read_csv('data/holidays_events.csv',parse_dates=['date'],dtype= {'transferred':np.bool,'type':'category','locale':'category'})
# holiday_df = holiday_df.query('transferred == False')
holiday_df=holiday_df.drop(['transferred'],axis = 1)

national_holiday_df = holiday_df[holiday_df.locale == 'National']
regional_holiday_df = holiday_df[holiday_df.locale == 'Regional']
local_holiday_df = holiday_df[holiday_df.locale == 'Local']

store = pd.read_csv('data/stores.csv')

items_df = pd.read_csv('data/items.csv',dtype = {'family':'category','perishable':bool,'class':'category'})

oil_df = pd.read_csv('data/oil.csv')
oil_df = oil_df[~oil_df.dcoilwtico.isnull()]
# drop missing value because not all date has this oil price anyway


def join_feature(df):
    df['dow'] = df['date'].dt.dayofweek
    df['doy'] = df['date'].dt.dayofyear
    
    df.loc[(df.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
    df['unit_sales_cal'] =  df['unit_sales'].apply(pd.np.log1p) #logarithm conversion

    # join store info
    df = pd.merge(df, store, on=['store_nbr'], how='left')
    
    # join holiday info
    df = pd.merge(df,national_holiday_df[['date', 'type']],on = 'date', how='left', suffixes=['', '_national'])
    df = pd.merge(df, local_holiday_df[['date','type','locale_name']], 
                     left_on = ['date','city'], right_on=['date','locale_name'], 
                     how='left', suffixes=['', '_city'])
    
    df = df.drop(['locale_name'], axis=1)
    
    # join item info
    df = df.merge(items_df, on='item_nbr', how='left')
    
    return df


def engineer_feature(df, mode='train', le_dict={}):
    
    if mode != 'test':
        df.loc[(df.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
        df['unit_sales_cal'] =  df['unit_sales'].apply(pd.np.log1p) #logarithm conversion

    to_drop = ['onpromotion']
    # to_drop += ['date', 'store_nbr', 'item_nbr']
    # to_dummy = ['perishable']
    df = df.drop(to_drop, axis = 1)
    
    to_dummy = []
    df = pd.get_dummies(df, columns=to_dummy)
    
    to_numerical = ['city', 'state', 'family', 'type_national','type_city']
    
    
    # save the label encoder for testing data

    for col in to_numerical:
        print (('Converting ', col)

        if (df[col].dtype != 'object') and (df[col].dtype == 'category') and (df[col].isnull().any().any()):
            df[col] = df[col].cat.add_categories(['NA'])
            df[col] = df[col].fillna('NA')

        new_col = col +'_coded'
        
        if mode == 'train':
            le = preprocessing.LabelEncoder()
            le_dict[col] = le.fit(df[col])
        
        df[new_col] = le_dict[col].transform(df[col])

    to_drop = to_numerical
    df.drop(to_drop, axis=1, inplace=True)
    
    df['perishable'] = df['perishable'].map({0: 1, 1: 1.25})
    
    return df, le_dict


train_eng, le_dict = engineer_feature(train_merged, 'train')
validate_eng, _ = engineer_feature(validate_merged, 'validate', le_dict)