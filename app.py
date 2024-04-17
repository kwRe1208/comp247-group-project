# Import the flask module. An object of Flask class is our WSGI application.
import os.path

import joblib
from flask import Flask, request, jsonify
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_model_path = "models"
_model_extesnion = "pkl"
_model_dict = {
    'rf': 'rf_best_model',
    'svm': 'SVM_best_model',
    'naive': 'naivebayes_trained_model',
    'logistic': 'best_logistic_regression_model',
    'voting': 'voting_classifier_model'
}

# Flask constructor takes the name of current module (__name__) as argument.
app = Flask(__name__)


@app.route("/prediction", methods=["POST"])
def process_model():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    request_data = request.get_json()
    print("request_data:", request_data)

    #store data into a dataframe
    data = request_data.get("data")
    df = pd.DataFrame(data, index=[0])

    #use joblib to load the pipline for preprocessing
    pipeline = joblib.load("models/preprocessor.pkl")
    features = pipeline.transform(df)

    model_name = request_data.get("model")
    data = request_data.get("data")
    model_file_path = f"{_model_path}/{_model_dict[model_name]}.{_model_extesnion}"
    print("model_file_path:", model_file_path)
    if not os.path.exists(model_file_path):
        return jsonify({"error": "Model not found"}), 404

    result = prediction(features, model_file_path)
    
    return jsonify({
        'model': model_name,
        'result': labeling_result(result[0]),
    }), 200


def prediction(data, model_file_path):
    model = joblib.load(model_file_path)
    result = model.predict(data)
    return result

def labeling_result(result):
    if int(result) == 0:
        return "Non-Fatal"
    else:    
        return "Fatal"

@app.route('/')
# ‘/’ URL is bound with welcome() function.
def welcome():
    return "Welcome to the Flask Web Application!"


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run(port=8000, debug=True, use_reloader=False)

