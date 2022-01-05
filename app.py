import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import os
import json
import datetime
import dash_auth
from layouts import layout
from callbacks import callbacks

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,assets_folder="static")


server = app.server
app.config.suppress_callback_exceptions = True

VALID_USERNAME_PASSWORD_PAIRS = {"admin": "admin"}

auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

app.layout = layout()
app.title = "Analyzing sports commentary in order to automatically recognize events and extract insights"

callbacks(app)

if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port="8050", debug=True)
