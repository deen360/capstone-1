import pickle
from sklearn.feature_extraction import DictVectorizer
from flask import Flask, request, jsonify
import numpy as np

with open('model-lin.b', 'rb') as f_in:
    (sc, model) = pickle.load(f_in)

dv = DictVectorizer(sparse=False)

def prepare_features(car):
    features = []
    features.append(car["Cylinders"])
    features.append(car["horsepower"])
    features.append(car["weight"])
    features.append(car["acceleration"])
    features = np.array([features])
    return features




def predict(features):

    X = sc.transform(features)
    preds = model.predict(X)
    return f'{round(preds[0])} Litres'


app = Flask('car-consumption-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'Car': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
