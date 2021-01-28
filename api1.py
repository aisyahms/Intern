""" flask server for model prediction """

import pandas as pd
import flask
from flask import Flask, request
from svd import convert_rating, svd_recommendation, recommend

app = flask.Flask(__name__)
app.config["DEBUG"] = True

df = pd.read_csv('NewProfile.csv')

@app.route("/")
def welcome():
    return "Hello, welcome!"

@app.route("/<uid>") # GET is default
def user(uid):
    reco = recommend(df, str(uid), 1, 5)
    return reco
                  
if __name__ == "__main__":
    app.run()
    