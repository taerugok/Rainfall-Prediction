import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

df=pd.read_csv('files/monthly_dashboard_df.csv')

#plot
def plot_monthly_rain_per_years(selected_stations,fromyear,toyear,df_f):
    cols=['station','station_name','year','month','month_name','monthly_rain_mm']
    tmp=df_f[(df_f.station.isin(selected_stations))&(df_f.year>=fromyear)&(df_f.year<=toyear)].drop_duplicates(subset=['year','month'])[cols]
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
        title="Total rain per month in the years "+str(fromyear)+'-'+str(toyear),
        xaxis_title="Month",
        yaxis_title="Monthly total rain (mm)",
        legend_title="Years",
    )
    fig.update_yaxes(title_text="Monthly total rain (mm)", secondary_y=True)
    fig.update_yaxes(range=[min_monthly_rain_overall, max_monthly_rain_overall], secondary_y=True)
    fig.update_yaxes(range=[min_monthly_rain_overall, max_monthly_rain_overall], secondary_y=False)
    return fig

figure2=plot_monthly_rain_per_years([370],2019,2024,df)
path='images/monthly_rain.png'
pio.write_image(figure2, path,width=1600, height=900)
print("fig saved to: ", path)


#plot
def plot_max_min_temps(selected_stations,fromyear,toyear,df_f,selected_palet):
    cols=['station','station_name','year','month','month_name','monthly_rain_mm','monthly_max_temp','monthly_min_temp']
    tmp=df_f[(df_f.station.isin(selected_stations))&(df_f.year>=fromyear)&(df_f.year<=toyear)].drop_duplicates(subset=['year','month'])[cols]
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
                       marker_size=6, marker_color=selected_palet[color_index],
                   mode='lines+markers',
                   name=str(yr)),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_min_temp,
                       line=dict(width=1.7),
                       marker_size=6, marker_color=selected_palet[color_index],
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
        title="Max and min temp per month in the years "+str(fromyear)+'-'+str(toyear),
        xaxis_title="Month",
        yaxis_title="Celcius degrees",
        legend_title="Years",
    )
    fig.update_yaxes(title_text="Celcius degrees", secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=False)
    return fig

from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale
colors_ = np.linspace(1, 10, 100)
discrete_colors = sample_colorscale('Reds', minmax_scale(colors_))
figure=plot_max_min_temps(selected_stations=[370],
                          fromyear=2018,toyear=2024,
                          df_f=df,
                          selected_palet=discrete_colors)
path='images/monthly_temps.png'
pio.write_image(figure, path,width=1600, height=900)
print("fig saved to: ", path)




