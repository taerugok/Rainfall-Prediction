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
#####################
## Monthly dashboards
#####################
st.header('Monthly plots', divider='gray')
st.subheader('Rainfall plots', divider='gray')
filename='files/monthly_dashboard_df.csv'
df = pd.read_csv(filename)
min_year = df['year'].min()+1
max_year = df['year'].max()
print("min",min_year,'max',max_year)

stations = df['station_name'].sort_values().unique()

selected_station = st.selectbox(
    'Which station would you like to view?',
    stations)

from_year, to_year = st.select_slider(
    "Select a range of years for monthly plots",
    options=df[df.year>=min_year].year.sort_values().unique(),
    value=(min_year, max_year),
)


#plot
def plot_monthly_rain_per_years(selected_stations,fromyear,toyear,df_f):
    cols=['station','station_name','year','month','month_name','monthly_rain_mm']
    tmp=df_f[(df_f.station_name==selected_stations)&(df_f.year>=fromyear)&(df_f.year<=toyear)].drop_duplicates(subset=['year','month'])[cols]
    tmp_mean=tmp.groupby(['month','month_name','station'])[['monthly_rain_mm']].mean().reset_index()
    max_monthly_rain_overall = np.ceil(tmp.monthly_rain_mm.max() / 20) * 20
    min_monthly_rain_overall = 0
    x_categories = tmp.sort_values('month').month.to_list()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for yr in tmp.year.unique():
        sel_tmp=tmp[tmp.year==yr]
        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_rain_mm,
                       line=dict(width=1.1),marker_size=7,
                   mode='lines+markers',
                   name=str(yr)),
            secondary_y=False
        )


    fig.add_trace(
        go.Scatter(x=tmp_mean.month_name, y=tmp_mean.monthly_rain_mm,
               line = dict(dash='dot',width=2,color='black'),
               mode='lines+markers',
               name="Average from "+str(fromyear)+" to "+str(toyear)),
        secondary_y=True
    )

    fig.update_xaxes(type="category",categoryorder='array', categoryarray= x_categories)
    fig.update_layout(
        title="Total rain per month in the years "+str(fromyear)+'-'+str(toyear)+" in "+ selected_stations,
        xaxis_title="Month",
        yaxis_title="Monthly total rain (mm)",
        legend_title="Years",
    )
    fig.update_yaxes(title_text="Monthly total rain (mm)", secondary_y=True)
    fig.update_yaxes(range=[min_monthly_rain_overall, 250], secondary_y=True)
    fig.update_yaxes(range=[min_monthly_rain_overall, 250], secondary_y=False)
    return fig

figure2=plot_monthly_rain_per_years(selected_station,from_year,to_year,df)
st.plotly_chart(figure2,width=1400,)
