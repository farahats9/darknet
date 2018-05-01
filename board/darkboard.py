import csv
import os

import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from flask import Flask, render_template

app = Flask(__name__)
if os.name() == 'nt':
    app.config['TRAINING_LOG_FILE'] = "./build/darknet/x64/backup/yolov2-obj_log.log"
else:
    app.config['TRAINING_LOG_FILE'] = "../backup/yolov2-obj_log.log"

def smooting(data, weight):
    last = data[0]
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + \
            (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val
    ynew = np.array(smoothed)
    return ynew


@app.route("/")
def hello():
    with open(app.config['TRAINING_LOG_FILE'], "r") as logfile:
        logreader = csv.reader(logfile, delimiter=',')
        x = []
        y = []
        for row in logreader:
            x.append(int(str(row[0]).split(":")[0]))
            y.append(float(str(row[1]).split(" ")[1]))
        x_a = np.array(x)
        y_a = np.array(y)
        trace1 = go.Scatter(
            x=x_a,
            y=y_a,
            mode='markers',
            name='average loss',
            line=dict(
                color='deepskyblue'
            ),
            opacity=0.5
        )
        ynew = smooting(y_a, 0.9)
        trace2 = go.Scatter(
            x=x_a,
            y=ynew,
            mode='lines',
            name='smoothed curve',
        )
        data = [trace1, trace2]
        config = {'showLink': False, 'displaylogo': False,
                  "modeBarButtonsToRemove": ['sendDataToCloud']}
        layout = go.Layout(
            title='Training Progress',
            autosize=True,
            margin=go.Margin(
                l=50,
                r=50,
                b=50,
                t=80,
                # pad=4
            ), xaxis=dict(
                title='Iterations',
                gridcolor="#6c757d"
            ),
            yaxis=dict(
                title='Avg. loss',
                gridcolor="#6c757d"
            ),
            plot_bgcolor="#343a40",
            paper_bgcolor="#343a40",
            font=dict(
                color="#fff"
            )
        )
        fig = go.Figure(data=data, layout=layout)
        div = py.plot(fig, output_type='div',
                      include_plotlyjs=False, config=config)
    return render_template('index.html', graph=div)


app.run(debug=True, host="0.0.0.0", port=6007)
