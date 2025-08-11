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
    page_icon=':rainbow:', # This is an emoji shortcode. Could be a URL too.
    # page_icon="images/weather_icon.png"
)

import base64
def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

# Load and display sidebar image
img_path = "images/logo3_transparent.png"
img_base64 = img_to_base64(img_path)
if img_base64:
    st.sidebar.markdown(
        f'<img src="data:images/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )

# st.logo('images/el-chalten.jpg')
# st.set_page_icon("images/weather_icon.png")
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


st.sidebar.header("About",divider='orange')
with st.sidebar:
    st.image('images/profile_round.png',width=200,caption="https://www.linkedin.com/in/volkangumuskaya/")
    
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
df = get_df(filename)


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

#Show measurements only for selected station
st.header('Latest measurements', divider=True)

stations = df['stationname'].sort_values().unique()
selected_station = st.selectbox(
    'Select station for latest measurements',
    stations)


for i in range(0, len(df)):
    if(df.iloc[i]['stationname']==selected_station):
        st.subheader(df.iloc[i]['stationname'], divider=True)
        st.text(df.iloc[i]['Time'])
        cols = st.columns(3)
        for j,k in zip(range(2, len(df.columns)), range(0, len(df.columns)-2)):
            col = cols[k % len(cols)]
            with col:
                st.metric(
                    label=df.columns[j],
                    value=df.iloc[i][j]
                )

#####################
## Monthly dashboards
#####################
st.header('Monthly plots', divider=True)
st.subheader('Rainfall plots', divider=True)
del(df)
filename='files/monthly_dashboard_df.csv'
df = get_df(filename)
min_year = df['year'].min()+1
max_year = df['year'].max()
print("min",min_year,'max',max_year)

stations = df['station_name'].sort_values().unique()

selected_station = st.selectbox(
    'Which station would you like to view?',
    stations)
dummy=max_year-2
from_year, to_year = st.select_slider(
    "Select a range of years for monthly plots",
    options=df[df.year>=min_year].year.sort_values().unique(),
    value=(dummy, max_year),
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

st.subheader('Max - min temperature plots', divider=True)
selected_station_t = st.selectbox(
    'Which station would you like to view for temperature plots?',
    stations)

from_year_t, to_year_t = st.select_slider(
    "Select a range of years for monthly temp plots",
    options=df[df.year>=min_year].year.sort_values().unique(),
    value=(min_year, max_year),
)

#plot
def plot_max_min_temps(selected_stations,fromyear,toyear,df_f,selected_palet_max,selected_palet_min):
    cols=['station','station_name','year','month','month_name','monthly_rain_mm','monthly_max_temp','monthly_min_temp']
    tmp=df_f[(df_f.station_name==selected_stations)&(df_f.year>=fromyear)&(df_f.year<=toyear)].drop_duplicates(subset=['year','month'])[cols]
    tmp_min=tmp.groupby(['month','month_name','station'])[['monthly_max_temp']].mean().reset_index()
    tmp_max=tmp.groupby(['month','month_name','station'])[['monthly_min_temp']].mean().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df_len=len(tmp.year.unique())
    interval_=int(79/(df_len-1))
    x_categories = tmp.sort_values('month').month.to_list()
    for yr,color_index in zip(tmp.year.unique(),range(20, interval_*df_len, interval_)):
        sel_tmp=tmp[tmp.year==yr]
        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_max_temp,
                       line=dict(width=1.7),
                       marker_size=6, marker_color=selected_palet_max[color_index],
                   mode='lines+markers',
                   name=str(yr)),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_min_temp,
                       line=dict(width=1.7),
                       marker_size=6, marker_color=selected_palet_min[color_index],
                       mode='lines+markers',
                       name=str(yr),showlegend=False),
            secondary_y=True
        )

    fig.add_trace(
        go.Scatter(x=tmp_min.month_name, y=tmp_min.monthly_max_temp,
               line = dict(dash='dot',width=1.2,color='black'),
               mode='lines+markers',
               name="Average min from "+str(fromyear)+" to "+str(toyear)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=tmp_max.month_name, y=tmp_max.monthly_min_temp,
               line = dict(dash='dot',width=1.2,color='black'),
               mode='lines+markers',
               name="Average max from "+str(fromyear)+" to "+str(toyear)),
        secondary_y=False
    )
    fig.add_hline(y=0, line_width=1.0, line_color="black")

    fig.update_xaxes(type="category",categoryorder='array', categoryarray= x_categories)
    fig.update_layout(
        title="Max and min temp per month in the years "+str(fromyear)+'-'+str(toyear)+" in "+ selected_stations,
        xaxis_title="Month",
        yaxis_title="Celcius degree",
        legend_title="Years",
    )
    fig.update_yaxes(title_text="Celcius degree", secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=False)
    return fig

from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale
colors_ = np.linspace(1, 10, 100)
discrete_colors_mx = sample_colorscale('Reds', minmax_scale(colors_))
discrete_colors_mn = sample_colorscale('Blues', minmax_scale(colors_))

figure=plot_max_min_temps(selected_stations=selected_station_t,
                          fromyear=from_year_t,toyear=to_year_t,
                          df_f=df,
                          selected_palet_max=discrete_colors_mx,selected_palet_min=discrete_colors_mn
                         )
st.plotly_chart(figure)

#Predictions part
filename='files/daily_prediction.csv'
df = get_df(filename)
if(len(df)>1):
        
    ''
    st.header('Rainfall amounts', divider=True)
    ''
    
    
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
