print('Importing libs..')
#Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
import os
import calendar
import re
from sklearn import datasets
import random
import pickle
import datetime

import seaborn as sns
import matplotlib.pyplot as plt


# THESE DATA CAN BE USED FREELY PROVIDED THAT THE FOLLOWING SOURCE IS ACKNOWLEDGED:
# ROYAL NETHERLANDS METEOROLOGICAL INSTITUTE
# RD: 24-hour sum of precipitation in tenths of a millimeter from 08:00 UTC previous day to 08:00 UTC current day.
# SX: Snow cover code number at 08:00 UTC.



print('Reading historical_rain_data.csv')
filename='files/historical_rain_data.csv'
df=pd.read_csv(filename)

print('Data preparation in process...')
#data preparation
selected_cols=['STN', 'YYYYMMDD', 'DDVEC', 'FG', 'TG', 'TN', 'TX',
          'DR', 'RH', 'RHX', 'RHXH',
          'PG', 'PX', 'PN','UX', 'UN']

print(df.head(10))

stations_df=pd.read_csv('files/station_list_klimatologie.csv')
df=pd.merge(df,stations_df,how='left',on='STN')
df['station_name']=df['station_name'].fillna('unknown')
print(df.head(10))

col_names_dict = {
    "STN": "station",
    "station_name": "station_name",
    "YYYYMMDD": "date",
    "DDVEC": "wind_direction",
    'FG':"mean_wind_speed",
    "TG":'mean_temp',
    "TN":'min_temp',
    "TX":'max_temp',
    'DR':'rain_duration',
    'RH':'rain_amount_mm',
    'RHX':'max_hourly_rain_mm',
    'RHXH':'time_of_max_rain',
    'PG':'mean_pressure',
    'PX':'max_pressure',
    'PN':'min_pressure',
    'UX':'max_humidity',
    'UN':'min_humidity'
}

#filter out not selected columns
df=df[[x for x in df.columns if x in col_names_dict]]

#Convert data to the expected format by LGBM model
df.columns = df.columns.str.strip()
df=df.rename(columns=col_names_dict)

df['rain_amount_mm']=df['rain_amount_mm']/10
df.loc[df.rain_amount_mm <=0, 'rain_amount_mm'] = 0

df['max_hourly_rain_mm']=df['max_hourly_rain_mm']/10
df.loc[df.max_hourly_rain_mm <=0, 'max_hourly_rain_mm'] = 0

df['mean_temp']=df['mean_temp']/10
df['min_temp']=df['min_temp']/10
df['max_temp']=df['max_temp']/10
df['mean_pressure']=df['mean_pressure']/10
df['min_pressure']=df['min_pressure']/10
df['max_pressure']=df['max_pressure']/10
df['rain_duration']=df['rain_duration']/10


df['year']=np.floor((df['date']/10000)).astype('int')
df['month']=(np.floor(df['date']/100)-np.floor((df['date']/10000))*100).astype('int')

#previous day rain in mm
df['next_day_rain_mm']=df['rain_amount_mm'].shift(periods=-1).fillna(0)

#seasons
#map one column to another
conditions = [(df['month'].isin ([12,1,2])),
              (df['month'].isin ([3,4,5])),
              (df['month'].isin ([6,7,8])),
            (df['month'].isin ([9,10,11]))
              ]
choices = ['Winter', 'Spring', 'Summer','Fall']
df['season'] = np.select(conditions, choices,default='na')
df['month_name'] =df['month'].apply(lambda x: calendar.month_name[x])

# Define features and target
df.columns
features = ['wind_direction','month_name','season',
            'mean_wind_speed',
            'mean_temp','min_temp','max_temp',
            'mean_pressure','max_pressure', 'min_pressure',
            'max_humidity', 'min_humidity',
            'rain_duration', 'rain_amount_mm','max_hourly_rain_mm', 'time_of_max_rain'
            ]
target = 'next_day_rain_mm'

df[features].dtypes
for col in df.columns:
    col_type = df[col].dtype
    if col_type == 'object' or col_type.name == 'string':
        df[col] = df[col].astype('category')
print(df.dtypes)

######
print('Creating train and test set..')

#####form train-test set
lgbm_params = {
    'n_estimators': 10,  # 100, opt 1000
    'max_depth': 6,  # 6, opt 14
    'learning_rate': 0.5,  # 0.3
    'reg_alpha': 0.5,  # none
    'reg_lambda': 0,  # none,
    # 'monotone_constraints': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #This is used when we want monotonic constraints for example for regression wrt a feature
}

# Define features and target

# Create X and y
df.index=np.arange(len(df))

X = df[features].copy()  # Features table
X.index=df.index.copy()
y = df[target]
y.index=df.index.copy()


# Split X and y randomly
X_train=X[X.index<=len(X)*0.8].copy()
X_test=X[X.index>len(X)*0.8].copy()

Y_train=y[y.index<=len(y)*0.8].copy()
Y_test=y[y.index>len(y)*0.8].copy()

Y_train_clf=pd.Series(np.where((Y_train<=0.1),0,1),index=Y_train.index,name='rain_occurrence')
Y_test_clf=pd.Series(np.where((Y_test<=0.1),0,1),index=Y_test.index,name='rain_occurrence')

# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.20,random_state=42)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
Y_train_clf.shape
Y_test_clf.shape

print('Training classification and regression models..')
# Fit model using predetermined parameters
# lgbr = lgb.LGBMRegressor(**lgbm_params)  # lgbr.get_params()
lgb_reg = lgb.LGBMRegressor()  # lgbr.get_params()
lgb_reg.fit(X_train, Y_train, eval_set=(X_test, Y_test), feature_name='auto', categorical_feature='auto')

lgbr_clf = lgb.LGBMClassifier()  # lgbr.get_params()
lgbr_clf.fit(X_train, Y_train_clf, eval_set=(X_test, Y_test_clf), feature_name='auto', categorical_feature='auto')
lgbr_clf.best_score_

# #Plot importance
# print('feature importance by gain')
# lgb.plot_importance(lgbr,importance_type='gain',figsize=(6,20),max_num_features=55)
# lgbr.feature_importances_
# plt.show()
#
# print('feature importance by split')
# lgb.plot_importance(lgbr,importance_type='split',figsize=(6,20),max_num_features=55)
# lgbr.feature_importances_
# plt.show()

print('Making predcitions..')
# make predictions
pred_test = lgb_reg.predict(X_test)
pred_test=np.where(pred_test<=0,0,pred_test)
pred_train = lgb_reg.predict(X_train)

pred_test_clf = lgbr_clf.predict(X_test)
pred_train_clf = lgbr_clf.predict(X_train)

pred_test_probs=lgbr_clf.predict_proba(X_test)[:,1]
pred_train_probs=lgbr_clf.predict_proba(X_train)[:,1]

print('Creating comprehensive DataFrames..')
# predictions as df using index of X_test
pred_test_df = pd.DataFrame({'pred_rainfall':pred_test,
                             'pred_rain_occurrence':pred_test_clf,
                             'rain_probability':pred_test_probs
                             },index=X_test.index)
pred_train_df = pd.DataFrame({'pred_rainfall':pred_train,
                              'pred_rain_occurrence':pred_train_clf,
                              'rain_probability':pred_train_probs
                              },index=X_train.index)

#Accuracy on training and test set
test_df=pd.concat([X_test,Y_test,Y_test_clf, pred_test_df], axis=1)
train_df=pd.concat([X_train,Y_train,Y_train_clf, pred_train_df], axis=1)


test_df[['rain_occurrence','pred_rain_occurrence']].value_counts(normalize=True)
train_df[['rain_occurrence','pred_rain_occurrence']].value_counts(normalize=True)

#error
test_df['error']=test_df['pred_rainfall']-test_df['rain_amount_mm']
train_df['error']=train_df['pred_rainfall']-train_df['rain_amount_mm']

test_df['error'].mean()
test_df['pred_rainfall'].mean()

test_df['sample_type']='test'
train_df['sample_type']='train'


df_all=pd.concat([train_df,test_df])
df_all=pd.merge(df['date'],df_all,how='left',left_index=True,right_index=True)

path='files/train_test_set_comprehensive.csv'
df_all.to_csv(path,index=False)

model_id=str(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
print('Model id is:',model_id)

rainfall_models=[lgb_reg,lgbr_clf,model_id]
path='files/rainfall_models.pickle'
print("Dumping models as artefacts to:", path)
with open(path, 'wb') as handle:
    pickle.dump(rainfall_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

#CREATE PLOTS

#plot heatmap
kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}
df.columns


#CONFUSION MATRIX TEST
preds=df_all[df_all.sample_type=='test']['pred_rain_occurrence'].copy()
actuals=df_all[df_all.sample_type=='test']['rain_occurrence'].copy()


cf_matrix = confusion_matrix(actuals, preds)
tmp = pd.DataFrame(cf_matrix)

x_labs=tmp.columns.to_list()
y_labs=tmp.index.to_list()

fig=sns.heatmap(tmp, cmap='Reds', xticklabels=x_labs, yticklabels=y_labs, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('Confusion matrix TEST set\n model')
print("fig created")
path='images/confusion_matrix_test.png'
plt.savefig(path)
print("fig saved to: ", path)
plt.close('all')

#CONFUSION MATRIX TRAIN
preds=df_all[df_all.sample_type=='train']['pred_rain_occurrence'].copy()
actuals=df_all[df_all.sample_type=='train']['rain_occurrence'].copy()


cf_matrix = confusion_matrix(actuals, preds)
tmp = pd.DataFrame(cf_matrix)

x_labs=tmp.columns.to_list()
y_labs=tmp.index.to_list()

fig=sns.heatmap(tmp, cmap='Blues', xticklabels=x_labs, yticklabels=y_labs, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('Confusion matrix TRAIN set\n model')
print("fig created")
path='images/confusion_matrix_train.png'
plt.savefig(path)
print("fig saved to: ", path)
plt.close('all')



