# we import flask

from flask import Flask, jsonify, redirect, render_template, request, session
import pickle
import pymongo
from pymongo import MongoClient
import numpy as np


client = pymongo.MongoClient("mongodb+srv://deen360<password>@cluster0.zyfi4dh.mongodb.net/?retryWrites=true&w=majority")


# name of database
db = client["car_cunsumption"]

#name of collection
collection = db["car"]


app = Flask(__name__)


with open('model-lin.b', 'rb') as f_in:
    (sc, model) = pickle.load(f_in)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        #: takes the user entry
        if not request.form.get("acceleration"):
            return render_template("index.html")


        else:
            Cylinders = request.form.get("Cylinders")
            horsepower = request.form.get("horsepower")
            weight = request.form.get("weight")
            acceleration = request.form.get("acceleration")




            Cylinders = int(Cylinders)
            horsepower = int(horsepower)
            weight = int(weight)
            acceleration = int(acceleration)




            car = {
                "Cylinders": Cylinders,
                "horsepower": horsepower,
                "weight": weight,
                "acceleration": acceleration
                }


            features = []
            features.append(car["Cylinders"])
            features.append(car["horsepower"])
            features.append(car["weight"])
            features.append(car["acceleration"])
            features = np.array([features])


            X = sc.transform(features)
            preds = model.predict(X)

            result = {'consumption':preds[0]}

            Fuel = result['consumption']

            data = { "Cylinders": Cylinders,"horsepower": horsepower,"weight": weight,"acceleration": acceleration,'Consumption': int(Fuel)}
            database = collection.insert_one(data)
            return render_template("index.html", Fuel=Fuel)

    else:

        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


