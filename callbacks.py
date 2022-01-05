import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from utils.helpers import *
import datetime
import numpy as np
import pickle
import sklearn
import xgboost


def callbacks(app):

    @app.callback(
    [   
        Output("output", "children"),
        Output("perso_commentary", "value")
    ],
    Input("perso_commentary", "value"),
    )
    def update_output(input):
        return [u'Personalized live commentary:\n {}'.format(input), input]
    
    @app.callback(
    [   
        Output("output_random", "children"),
        Output("random_commentary", "value")
    ],
    Input("results_data", "data"),
    )
    def update_output_random(data):
        df = pd.DataFrame(data)
        row = df.sample(n=1)
        text = str(row["text"].item())
        return [u'Random live commentary:\n {}'.format(text), text] 
    
    @app.callback(
        [
            Output("results_data", "data"),
            Output("model_stats", "data"),
            Output("model_choices", "options"),
        ],
        Input("main_frame_div", "id"),
    )
    def load_mainframe(id):
        results_data, model_stats = load_contents()
        model_options = ["AdaBoost", "DecisionTree", "KNeighbors", "MLP", "RandomForest", "SVM", "XGBoost"]
        model_options = [{"label": val, "value": val} for val in model_options]

        return [results_data.to_dict("records"),
                model_stats.to_dict("records"), 
                model_options]

    @app.callback(
        [
            Output("result_text","children"),
            Output("result_conf","children"),
        ],
        [
            Input("perso_commentary", "value"),
            Input("random_commentary", "value"),
            Input("results_data", "data"),
            Input("model_stats", "data"),
            Input("model_choices", "value"),
            
        ],
    )
    def display_results_summary(perso_commentary, random_commentary, results_data, model_stats, model_name):
        if not results_data or not model_stats:
            return dash.no_update, dash.no_update, dash.no_update
        results_data = pd.DataFrame(results_data)
        model_stats = pd.DataFrame(model_stats)
        if model_name:
            model = pickle.load(open(f'./models/model_{model_name}.pickle', 'rb'))
        else:
            return ["", ""]
        if perso_commentary:
            prediction = model.predict(finalpreprocess(perso_commentary))
            proba = model.predict_proba(finalpreprocess(perso_commentary))
        else :
            prediction = model.predict(finalpreprocess(random_commentary))
            proba =  model.predict_proba(finalpreprocess(random_commentary))
        
        num_to_cat = {  1:"Attempt", 
                        2:"Corner",
                        3:"Foul", 
                        4:"Yellow card", 
                        5:"Second yellow card", 
                        6:"Red card",
                        7:"Substitution", 
                        8:"Free kick won", 
                        9:"Offside", 
                        10:"Hand ball", 
                        11:"Penalty conceded"
                    }
        event_type = str(num_to_cat[prediction[-1]])
        if prediction[-1] <5:
            confidence = proba[-1][prediction[-1]-1]*100
        else:
            confidence = proba[-1][prediction[-1]-2]*100

        return [str(event_type), 
                f"Confidence of the prediction: {round(confidence,2)}%",
        ]
