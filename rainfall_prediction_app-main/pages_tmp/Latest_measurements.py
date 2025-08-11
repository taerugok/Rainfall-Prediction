import streamlit as st
import pandas as pd
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='volkan-ai',
    layout="wide",
    page_icon=':volcano:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.
st.cache_data.clear()
def get_df(filename):
    print('Reading df')
    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/filename
    tmp_df = pd.read_csv(DATA_FILENAME)
    print ('df read with shape ',tmp_df.shape,' and type ',type(tmp_df))
    return tmp_df


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.image('images/el-chalten.jpg','El Chalten, Patagonia')
st.sidebar.header("About")
with st.sidebar:
    st.image('images/profile_round.png',width=170,caption="https://www.linkedin.com/in/volkangumuskaya/")
    
'''
# Daily and monthly meteorological outlook for the cities in Netherlands

This is an example project to demonstrate MLOps, data visualisation and DS capabilities. The prediction algorithm is a simple model in the sole purpose of demonstration purposes. 
As such, it is not a full-blown weather model that aims to provide industry standard predictions. 
More can be found in the [github repo here](https://github.com/volkangumuskaya/rainfall_prediction_app/blob/main/README.md)

The data is provided via the [KNMI API](https://daggegevens.knmi.nl/klimatologie/daggegevens)
'''

# Add some spacing
''
''

###Reading latest measurements
filename='files/latest_measurements.csv'
df = pd.read_csv(filename)


cols=['stationname',
      'Time',
      'Temperature',
      'Rainfall_Duration_last_hour_minutes',
      'Amount_Rainfall_last_Hour_in_mm',
      'Total_cloud_cover_percentage',
      'Air_pressure_in_hPa',
      'Wind_Speed_kmh',
      'Wind_Direction'
      ]

df=df[cols].copy()
col_names_dict = {
    "stationname": "stationname",
    "Rainfall_Duration_last_hour_minutes": "Rainfall duration (mins) ",
    "Amount_Rainfall_last_Hour_in_mm": "Rainfall amount (mm)",
    'Wind_Direction':"Wind Direction",
    "Wind_Speed_kmh":'Windspeed (km/h)',
    "Air_pressure_in_hPa":'Pressure (hPA)',
    "Total_cloud_cover_percentage":'Cloud cover (%)',
    'Temperature':'Temperature (C)'
}

df=df.rename(columns=col_names_dict)

st.header('Latest measurements', divider='red')

for i in range(0, len(df)):
    st.subheader(df.iloc[i]['stationname'], divider='gray')
    st.text(df.iloc[i]['Time'])
    cols = st.columns(3)
    for j,k in zip(range(2, len(df.columns)), range(0, len(df.columns)-2)):
        col = cols[k % len(cols)]
        with col:
            st.metric(
                label=df.columns[j],
                value=df.iloc[i][j]
            )
