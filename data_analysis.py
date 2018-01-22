import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import gc

#For statistical tests
import scipy.stats as st

#For formula notation (similar to R)
import statsmodels.formula.api as smf

import seaborn as sns
sns.set(style = 'whitegrid', color_codes = True)

train_df = pd.read_feather('data/train_1year.feat')
train_df.loc[train_df['unit_sales']<0,'unit_sales']=0


# load store info
store_df = pd.read_csv('data/stores.csv')
# load holiday info
holiday_df = pd.read_csv('data/holidays_events.csv',parse_dates=['date'])
# load item info
items_df = pd.read_csv('data/items.csv',dtype = {'family':'category','perishable':bool,'class':'category'})
# load oil info
oil_df = pd.read_csv('data/oil.csv')
#load transacrion
trans_df = pd.read_csv("data/transactions.csv")


# for plotting

strain = train_df.sample(frac=0.01)

### plot date features 
fig1, axiss = plt.subplots(4,1,figsize=(30,30))
sns.barplot(x='month',y='unit_sales',data=strain,ax=axiss[0])
sns.barplot(x='quater',y='unit_sales',data=strain,ax=axiss[1])
sns.barplot(x='day',y='unit_sales', data=strain,ax=axiss[2])
sns.barplot(x='dow',y='unit_sales',data=strain,ax=axiss[3])

fig1.show()

### join item
train_item = train_df.merge(items_df, how='inner')

perishable = train_item[train_item['perishable']]
non_perishable = train_item[~train_item['perishable']]

fi2, axis = plt.subplots(1,1,figsize=(15,15))
sns.distplot(perishable['unit_sales'], bins=range(0, 81, 1), kde=False, color='red',label='perishable')
sns.distplot(non_perishable['unit_sales'], bins=range(0, 81, 1), kde=False, color='blue',
            axlabel='unit_sales',label='non-perishable')
plt.legend()


fig, axis  = plt.subplots(1,1,figsize=(15,15))
sns.countplot(x='family',hue='perishable',data=train_item, ax=axis)

del train_item
gc.collect();


### join holiday 
national_holiday_df = holiday_df[holiday_df.locale == 'National']
regional_holiday_df = holiday_df[holiday_df.locale == 'Regional']
local_holiday_df = holiday_df[holiday_df.locale == 'Local']


train_df= train_df.merge(store_df, on='store_nbr', how='left')
train_holiday = pd.merge(train_df,national_holiday_df[['date', 'type']],on = 'date', how='left', suffixes=['', '_national'])
train_holiday = pd.merge(train_holiday, local_holiday_df[['date','type','locale_name']], 
                 left_on = ['date','city'], right_on=['date','locale_name'], 
                 how='left', suffixes=['', '_city'])

train_holiday = train_holiday.drop(['locale_name'], axis=1)

train_holiday['type_national'] = train_holiday['type_national'].fillna('NA')
train_holiday['type_city'] = train_holiday['type_city'].fillna('NA')

strain = train_holiday.sample(frac=0.01)

figure, (axis1,axis2) = plt.subplots(1,2, figsize=(15,30))
sns.barplot(x='type_national',y='unit_sales',data=strain, ax=axis1)
sns.barplot(x='type_city',y='unit_sales',data=strain, ax=axis2)
 
del train_holiday
gc.collect();


### regression on oil price
train_df = train_df.join(oil_df.set_index('date'), on='date')

train_df['dcoilwtico'] = train_df['dcoilwtico'].fillna(method='ffill')

train_df_mean= train_df.groupby('date').mean().reset_index()

lm1 = smf.ols(formula = 'unit_sales ~ dcoilwtico',train_df_mean).fit()
lm1 = smf.ols(formula = 'unit_sales ~ dcoilwtico',data=train_df_mean).fit()

train_df_mean['unit_sales_calc'] = train_df_mean['unit_sales'].apply(pd.np.log1p)
lm2 = smf.ols(formula = 'unit_sales_calc ~ dcoilwtico',data=train_df_mean).fit()


fig, (axis1, axis2) = plt.subplots(2,1,figsize=(15,4))

ax1 = train_df_mean['unit_sales_calc'].plot(legend=True,ax=axis1,marker='o',title="Transformed Average Sales")
ax2 = train_df_mean['dcoilwtico'].plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Oil price")

# features to use:

cols = ['dow', 'month','perishable','class','family','store_nbr','item_nbr','type_national']
