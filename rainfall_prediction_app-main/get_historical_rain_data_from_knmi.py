import pandas as pd
import re
import io
import os
import requests
from requests.exceptions import HTTPError

data = {
    'start': os.environ['TRAIN_START_DATE'],
    'end': os.environ['TRAIN_END_DATE'],
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
print(df.dtypes)
print(df.shape)

path='files/historical_rain_data.csv'
df.to_csv(path,index=False)
