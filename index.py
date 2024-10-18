from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.ensemble import StackingRegressor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


ridge_model = joblib.load(r'E:\MachineLearning\BE\ridge_regression_model.pkl')
nn_model = joblib.load(r'E:\MachineLearning\BE\neural_network_model.pkl')
lin_model = joblib.load(r'E:\MachineLearning\BE\linear_regression_model.pkl')
scaler = joblib.load(r'E:\MachineLearning\BE\scaler.pkl')
stacking_model = joblib.load(r'E:\MachineLearning\BE\stacking_model.pkl')

@app.route('/')
def hello_world():
    return 'Tuấn Anh đã ở đây'

def extract_features(data):
    return np.array([
        data.get('cylinders', 0),
        data.get('displacement', 0),
        data.get('horsepower', 0),
        data.get('weight', 0),
        data.get('acceleration', 0),
        data.get('model_year', 0),
        data.get('origin', 0)
    ]).reshape(1, -1)

@app.route('/predict/ridge', methods=['POST'])
def predict_ridge():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid input'}), 400

    features = extract_features(data)
    features = scaler.transform(features)
    ridge_pred = ridge_model.predict(features)[0]

    return jsonify({'result': ridge_pred})

@app.route('/predict/nn', methods=['POST'])
def predict_nn():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid input'}), 400

    features = extract_features(data)
    features = scaler.transform(features)
    nn_pred = nn_model.predict(features)[0]

    return jsonify({'result': nn_pred})

@app.route('/predict/linear', methods=['POST'])
def predict_linear():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid input'}), 400

    features = extract_features(data)
    features = scaler.transform(features)
    lin_pred = lin_model.predict(features)[0]

    return jsonify({'result': lin_pred})

@app.route('/predict/stacking', methods=['POST'])
def predict_stacking():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid input'}), 400

    features = extract_features(data)
    features = scaler.transform(features)
    stacking_pred = stacking_model.predict(features)[0]

    return jsonify({'result': stacking_pred})

if __name__ == '__main__':
    app.run(debug=True)
