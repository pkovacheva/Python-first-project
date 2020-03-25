import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import plotly.graph_objs as go
import numpy as np

app = dash.Dash()

training_data = joblib.load("./training_data.pkl")


app.layout = html.Div(children=[
    html.H1(children='Check house prices in Kings County area', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Enter area of the house: '),
        dcc.Input(id='area', placeholder='area', type='number'),
        html.Div(id='result')
    ], style={'textAlign': 'center'}),

    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=training_data['sqm_living'],
                    y=training_data['price'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'},
                        'colorscale': 'Viridis'
                    },
                    marker_color=training_data['price']
                )
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'Area'},
                yaxis={'title': 'Price'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest'
            )
        }
    )
])


@app.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='area', component_property='value')])
def update_area_input(area):
    if area is not None and area is not '':
        try:
            X_pred = np.array([[area]])
            price = model.predict(X_pred)[0]
            return 'With {} square meters you should buy a house for ${:,.2f}'.\
                format(area, price, 2)
        except ValueError:
            return 'Unable to give price of the house'


if __name__ == '__main__':
    model = joblib.load("./LR_ser.pkl")
    app.run_server(debug=True)