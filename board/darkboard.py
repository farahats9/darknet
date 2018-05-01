import csv

import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from flask import Flask, render_template

app = Flask(__name__)

app.config['TRAINING_LOG_FILE'] = "../backup/yolov2-obj_log.log"


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
            mode='lines',
            name='average loss',
            line=dict(
                color='deepskyblue'
            )
        )
        data = [trace1]
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
