import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import dash_table
import base64

colors = {"background": "#000000", "text": "#ffffff"}

test_png = './assets/hist_classifiers.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

def layout():
    return html.Div(
        [
            html.Div([html.Center([html.H1("Analyzing sports commentary to recognize events and extract insights"),
            html.Div(html.H5("Yanis Miraoui"),style={"color":"blue"}),
            html.Div(html.H5("21-908-504 / ETH Zürich"),style={"color":"blue"}),
            html.Div(html.H5("ymiraoui@student.ethz.ch"),style={"color":"blue"}),
            ])]),
            html.Div([dbc.Tabs([dbc.Tab(home_tab(),label="Home")])]
        )])

def home_tab():
    return html.Div([html.Div([
                html.Div([
                            html.Div(
                                [dcc.Store(data=[], id="results_data"), dcc.Store(data=[], id="model_stats")]
                            ),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        placeholder="Classification models",
                                        id="model_choices",
                                        multi=False,
                                    )
                                ],className="spaced_div"
                            ),
                            html.Div(
                                [
                                    html.Br(),
                                    html.Br(),
                                    html.I("Type your personalized live sports commentary below :\n"),
                                    html.Div([
                                    dcc.Input(
                                        placeholder="Type your live commentary here",
                                        type="text",
                                        value="",
                                        id="perso_commentary",
                                    )], style=dict(display='flex')),
                                    html.Div(id="output")
                                ],className="spaced_div"
                            ),
                            html.Div(
                                [   
                                    html.Br(),
                                    html.Br(),
                                    html.Div(html.I("Click this button if you want to use an existing live sports commentary at random :\n")),
                                    html.Div([
                                    dcc.Input(
                                        placeholder="Random commentary will appear here",
                                        type="text",
                                        value="",
                                        id="random_commentary",
                                    )], style=dict(display='flex')),
                                    html.Div(html.Button('Generate sports commentary', id='submit_val_random', n_clicks=0)),
                                    html.Div(id="output_random")
                                ],className="spaced_div"
                            ),
                        ],
                        className="pretty_container",
                    ),
                html.Div([
                            html.Div(
                                [
                                     html.Div([ html.Div(html.H3("Predicted class:"),style={"font-size":"5.0rem"}), 
                                                html.Div(html.Center([],id="result_text",style={"font-size":"5.0rem"})), 
                                                html.Br(),
                                                html.Div([],id="result_conf",style={"color":"blue", "font-size":"2.0rem"})]),
                                                html.Div(html.Center([html.Audio(id='audio-out', preload='auto', autoPlay=True)]))
                                ],className="spaced_div pretty_container" 
                            ),
                            html.Div(
                                [
                                    html.Div(html.H4("Confusion matrix: "),style={"font-size":"3.0rem"}),
                                    html.Div([html.Img(src='data:image/png;base64,{}'.format(test_base64))]),
                                ],className="spaced_div pretty_container" 
                            )
                        ],
                        className="four columns",
                    ),
                html.Div([
                            html.Div(
                                [
                                    html.Div(html.H3("Guidelines: "),style={"font-size":"5.0rem"}),
                                    html.Div(html.H5("1.")),
                                    html.Div(html.H5("2.")),
                                    html.Div(html.H5("3.")),
                                    html.Div(html.H5("4.")),
                                    html.Div(html.H5("5.")),
                                    html.Div(html.H5("6.")),
                                ],className="spaced_div pretty_container" 
                            ),
                            html.Div(
                                [
                                    html.Div(html.H4("General performance of the models: "),style={"font-size":"3.0rem"}),
                                    html.Div([html.Img(src='data:image/png;base64,{}'.format(test_base64))]),
                                ],className="spaced_div pretty_container" 
                            )
                        ],
                        className="five columns",
                    )
                ],style={"display":"flex"},id="main_frame_div"
            )])

