""" flask server for model prediction """

import pandas as pd
import flask
from flask import Flask, request, render_template
from svd import convert_rating, svd_recommendation, recommend

app = Flask(__name__)
app.config["DEBUG"] = True

df = pd.read_csv('NewProfile.csv')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("input.html")

@app.route("/recommendations", methods=["POST"])
def recommendations():
    uid = request.form.get("user_id")
    reco = recommend(df, str(uid), 1, 5)
    return reco

        
if __name__ == "__main__":
    app.run()
    # app.run(debug=True)
    