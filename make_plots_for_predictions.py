import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#plot heatmap
print('Reading daily_prediction.csv')
filename='files/daily_prediction.csv'
df=pd.read_csv(filename)

y_max=np.ceil(max(df.rain_amount_mm_prediction.max(),df.next_day_rain_mm.max())/20)*20
y_min=-y_max

kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}
df.columns


df['error']=df['rain_amount_mm_prediction']-df['next_day_rain_mm']
df['date']=df['date'].astype('str')
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=df.date, y=df.error,
           # text=round(metrics.R2, 2),
           marker_color='dodgerblue', opacity=0.9,
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


fig.update_traces(marker=dict(size=10,
                              line=dict(width=1, color='black')),
                  selector=dict(mode='markers'))
fig.update_layout(
    title="Rain prediction for Eindhoven",
    xaxis_title="Date",
    yaxis_title="Rain amount",
    legend_title="Legend",
)
fig.update_yaxes(title_text="Rain amount", secondary_y=True)
fig.update_yaxes(range=[y_min,y_max], secondary_y=False)
fig.update_yaxes(range=[y_min,y_max], secondary_y=True)
print("Preds-actuals-errors fig created")
path='images/preds_actuals_errors.png'
# plt.savefig(path)
import plotly.io as pio
pio.write_image(fig, path,width=1600, height=900)

print("fig saved to: ", path)
plt.close('all')

##heat map
#CONFUSION MATRIX TRAIN
df['next_day_rain_occurrence']=pd.Series(np.where((df.next_day_rain_mm<=0.1),0,1))
preds=df['rain_occurrence_prediction'].copy()
actuals=df['next_day_rain_occurrence'].copy()

cf_matrix = confusion_matrix(actuals, preds)
tmp = pd.DataFrame(cf_matrix)

x_labs=tmp.columns.to_list()
y_labs=tmp.index.to_list()

fig=sns.heatmap(tmp, cmap='Blues', xticklabels=x_labs, yticklabels=y_labs, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('Confusion matrix for daily predictions')
print("preds_actuals_confusion_matrix created")
path='images/preds_actuals_confusion_matrix.png'
plt.savefig(path)
print("fig saved to: ", path)
plt.close('all')
