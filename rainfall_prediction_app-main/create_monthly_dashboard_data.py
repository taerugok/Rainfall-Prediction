

print('Importing libs..')
#Import libraries

import numpy as np
import pandas as pd
import os
import calendar
import datetime
import io
import requests
from requests.exceptions import HTTPError

#set training end date to 1 day ago
end_date=(datetime.date.today()-datetime.timedelta(days=1))

#start_date is 10 years before end_date
start_date=end_date-datetime.timedelta(days=3650)

end_date=int(end_date.strftime('%Y%m%d'))
start_date=int(start_date.strftime('%Y%m%d'))

data = {
    'start': start_date,
    'end': end_date,
    'vars': 'ALL',
    'stns': '240:344:370:380:269',
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
selected_cols=['STN', 'YYYYMMDD', 'DDVEC', 'FG', 'TG', 'TN', 'TX',
          'DR', 'RH', 'RHX', 'RHXH',
          'PG', 'PX', 'PN','UX', 'UN']

col_names_dict = {
    "STN": "station",
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

stations_df=pd.read_csv('files/station_list_klimatologie.csv')
df=pd.merge(df,stations_df,how='left',on='STN')
df['station_name']=df['station_name'].fillna('unknown')

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
df['day']=(df['date']-(np.floor(df['date']/100)*100)).astype('int')
df['daymonthyear']=df['date']
df['yearmonth']=df['year']*100+df['month']
df['date']=df['date'].astype('string')

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

df['monthly_rain_mm']=df.groupby(['year','month','station'])[['rain_amount_mm']].transform('sum')

df['monthly_max_temp']=df.groupby(['year','month','station'])[['max_temp']].transform('max')
df['monthly_min_temp']=df.groupby(['year','month','station'])[['min_temp']].transform('min')
df.to_csv("files/monthly_dashboard_df.csv")
