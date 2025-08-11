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
# -----------------------------------------------------------------------------

#Predictions part
''
st.header('Rainfall amounts', divider='gray')
''
filename='files/daily_prediction.csv'
df = pd.read_csv(filename)

df=df.fillna(0)
df = df.replace('',0)
min_value = df['date'].min()
max_value = df['date'].max()
print("min",min_value,'max',max_value)

from_year, to_year = st.select_slider(
    "Select a range of date",
    options=df.date.sort_values().unique(),
    value=(min_value, max_value),
)
''
stations = df['station_name'].sort_values().unique()

if not len(stations):
    st.warning("Select at least one station")

selected_stations = st.multiselect(
    'Which station would you like to view?',
    stations,
    ["Eindhoven","Rotterdam"])

''

# Filter the data
filtered_df = df[
    (df['station_name'].isin(selected_stations))
    & (df['date'] <= to_year)
    & (from_year <= df['date'])
]


''
filtered_df['date']=filtered_df['date'].astype('str')

''
st.bar_chart(
    filtered_df,
    x='date',
    y='next_day_rain_mm',
    color='station_name',
    y_label='Rainfall amount',
    stack=False
)
''

selected_stat = st.selectbox(
    'Which station to be inspected in detail vis-a-vis predictions?',
    stations
)

#plot detailed
df = df[
    (df['station_name']==selected_stat)
    & (df['date'] <= to_year)
    & (from_year <= df['date'])
].copy()
df['date']=df['date'].astype('str')
df['error']=df['rain_amount_mm_prediction']-df['next_day_rain_mm']
y_max=np.ceil(max(df.rain_amount_mm_prediction.max(),df.next_day_rain_mm.max())/20)*20
y_min=-y_max

kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}

#Start creating plotly plot
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=df.date, y=df.error,
           # text=round(metrics.R2, 2),
           marker_color='dodgerblue', opacity=0.7,
           name="Error"),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=df.date, y=df.next_day_rain_mm,
               mode='markers', name="Next day rain"),
    secondary_y=True
)

fig.add_trace(
    go.Scatter(x=df.date, y=df.rain_amount_mm_prediction,
               mode='markers', name="Rain prediction"),
    secondary_y=True
)


fig.update_traces(marker=dict(size=5.5,
                              line=dict(width=0.3, color='black')),
                  selector=dict(mode='markers'))
fig.update_layout(
    title="Detailed predictions, actuals and errors for selected station",
    xaxis_title="Date",
    yaxis_title="Rain amount",
    legend_title="Legend",
)
fig.update_layout(xaxis_type='category')
fig.update_xaxes(tickangle=270)
fig.update_yaxes(range=[y_min,y_max], secondary_y=False)
fig.update_yaxes(range=[y_min,y_max],title_text="Rain amount", secondary_y=True)
st.plotly_chart(fig)
