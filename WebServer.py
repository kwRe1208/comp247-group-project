# Import the flask module. An object of Flask class is our WSGI application.
import os.path

import joblib
from flask import Flask, request, jsonify


_model_path = "models"
_model_extesnion = "pkl"
_model_dict = {
    'rf': 'rf_best__model',
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
    model_name = request_data.get("model")
    data = request_data.get("data")
    model_file_path = f"{_model_path}/{_model_dict[model_name]}.{_model_extesnion}"
    print("model_file_path:", model_file_path)
    if not os.path.exists(model_file_path):
        return jsonify({"error": "Model not found"}), 404

    result = prediction(data, model_file_path)
    return jsonify(result), 200


def prediction(data, model_file_path):
    model = joblib.load(model_file_path)
    result = model.predict(data)
    return result


@app.route('/')
# ‘/’ URL is bound with welcome() function.
def welcome():
    return "Welcome to the Flask Web Application!"


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)

