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
from scipy.io.wavfile import write, read
import time


def callbacks(app):

    @app.callback(
    [   
        Output("output", "children"),
        Output("perso_commentary", "value"),
        Output("output_random", "children"),
        Output("random_commentary", "value")
    ],
    Input("perso_commentary", "value"),
    Input("results_data", "data"),
    Input("submit_val_random", "n_clicks"),
    )
    def update_output(input, data, button1):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if "submit_val_random" in changed_id:
            input = ""
            df = pd.DataFrame(data)
            row = df.sample(n=1)
            text = str(row["text"].item())
            return [u'Personalized live commentary:\n {}'.format(input), input, u'Random live commentary:\n {}'.format(text), text]
        else :
            return [u'Personalized live commentary:\n {}'.format(input), input, "Random live commentary:\n", ""]
    
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
            Output("audio-out", "src"),
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
        audio = ""
        if not results_data or not model_stats:
            return dash.no_update, dash.no_update, dash.no_update
        results_data = pd.DataFrame(results_data)
        model_stats = pd.DataFrame(model_stats)
        if model_name:
            model = pickle.load(open(f'./models/model_{model_name}.pickle', 'rb'))
        else:
            return ["No model specified", "", ""]
        if len(perso_commentary) > 15:
            time.sleep(3)
            text_to_wav(perso_commentary)
            rate = 22050
            buffer = io.BytesIO()
            rate, audio_numpy = read("./assets/en-GB.wav")
            write(buffer, rate, audio_numpy)
            b64 = base64.b64encode(buffer.getvalue())
            audio = "data:audio/x-wav;base64," + b64.decode("ascii")

            prediction = model.predict(finalpreprocess(perso_commentary))
            proba = model.predict_proba(finalpreprocess(perso_commentary))
        elif len(random_commentary) > 10:
            text_to_wav(random_commentary)
            buffer = io.BytesIO()
            rate, audio_numpy = read("./assets/en-GB.wav")
            write(buffer, rate, audio_numpy)
            b64 = base64.b64encode(buffer.getvalue())
            audio = "data:audio/x-wav;base64," + b64.decode("ascii")

            prediction = model.predict(finalpreprocess(random_commentary))
            proba =  model.predict_proba(finalpreprocess(random_commentary))
        else:
            return ["No text specified", "", ""]
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
                audio
        ]
