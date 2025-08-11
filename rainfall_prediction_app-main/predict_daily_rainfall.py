import pandas as pd
import re
import io
import os
import requests
from requests.exceptions import HTTPError
import numpy as np
import datetime
import calendar

#set date to 15 days ago
end_date=int((datetime.date.today()-datetime.timedelta(days=1)).strftime('%Y%m%d'))
start_date=int((datetime.date.today()-datetime.timedelta(days=35)).strftime('%Y%m%d'))

data = {
    'start': start_date,
    'end': end_date,
    'vars': 'ALL',
    'stns': 'ALL',
}

URL = "https://daggegevens.knmi.nl/klimatologie/daggegevens"

try:
    response = requests.post(URL, data=data)
    response.raise_for_status()
except HTTPError as http_err:
    print(f"HTTP error occurred : {http_err}")
except Exception as err:
    print(f"Other error occurred: {err}")
else:
    print("Success with status code",response.status_code)


##Parse response as df

s=str(response.content)# response as string
s=s[s.rindex('#')+1:] #find last # and delete the first part
s=s.replace('\\n','\n') #replace //n with /n
s=s.replace(' ','') #delete space
s=s.replace("'",'') #delete '
df=pd.read_csv(io.StringIO(s), sep=",")
type(df)



#data preparation
print(df.head(10))

stations_df=pd.read_csv('files/station_list_klimatologie.csv')
df=pd.merge(df,stations_df,how='left',on='STN')
df['station_name']=df['station_name'].fillna('unknown')
print(df.head(10))

selected_cols=['STN','station_name', 'YYYYMMDD', 'DDVEC', 'FG', 'TG', 'TN', 'TX',
          'DR', 'RH', 'RHX', 'RHXH',
          'PG', 'PX', 'PN','UX', 'UN']



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


print("loading rainfall models")
import pickle
with open('files/rainfall_models.pickle', 'rb') as handle:
    reg_model,clf_model,model_id = pickle.load(handle)

print("Making predcitions")
print(df[features])
print("rain occurrence predcition..")
rain_occurrence_prediction=clf_model.predict(df[features])
print("rain amount predcition..")
rain_amount_mm_prediction=reg_model.predict(df[features])
print("rain occurrence probability..")
chance_of_rain_prediction=clf_model.predict_proba(df[features])[:,1]

df_test = pd.DataFrame({'chance_of_rain_prediction':chance_of_rain_prediction,
                        'rain_occurrence_prediction':rain_occurrence_prediction,
                        'rain_amount_mm_prediction':rain_amount_mm_prediction
                        })
df=pd.concat([df,df_test],axis=1)
# df=df.sort_values('date',ascending=False).drop_duplicates(subset=['station'])
path='files/daily_prediction.csv'
print('saving to path:',path)
df['rain_amount_mm_prediction']=df['rain_amount_mm_prediction']*df['rain_occurrence_prediction']
df['pred_run_on']=str(datetime.datetime.now())
df['used_model']=model_id
print(df)
df.to_csv(path,mode='w',header=True,index=False)
