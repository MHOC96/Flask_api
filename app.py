from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

models = {
    'Ferritin': pickle.load(open(os.path.join(MODEL_DIR, 'ferritin_model.pkl'), 'rb')),
    'B12': pickle.load(open(os.path.join(MODEL_DIR, 'b12_model.pkl'), 'rb')),
    'CRP': pickle.load(open(os.path.join(MODEL_DIR, 'crp_model.pkl'), 'rb')),
    'Cystatin_C': pickle.load(open(os.path.join(MODEL_DIR, 'cystatin_c_model.pkl'), 'rb')),
    'HBA1C': pickle.load(open(os.path.join(MODEL_DIR, 'hba1c_model.pkl'), 'rb')),
    'AFP': pickle.load(open(os.path.join(MODEL_DIR, 'afp_model.pkl'), 'rb'))
}


features = {
    'Ferritin': ['age', 'sex', 'hb', 'hct', 'rbc', 'mcv', 'mch', 'mchc', 'wbc', 'alt', 'ast'],
    'B12': ['age', 'hb', 'hct', 'rbc', 'mcv', 'mch'],
    'CRP': ['wbc', 'neutrophils', 'lymphocytes', 'alt', 'ast', 'ggt', 'albumin'],
    'Cystatin_C': ['age', 'sex', 'urea', 'creatinine', 'egfr', 'albumin'],
    'HBA1C': ['age', 'fpg', 'triglycerides', 'cholesterol_total', 'hdl', 'ldl', 'alt', 'ast', 'hb', 'hct', 'rbc', 'mcv', 'mch', 'mchc'],
    'AFP': ['age', 'sex', 'alt', 'ast', 'alp', 'ggt', 'bilirubin_total', 'bilirubin_direct', 'albumin']
}


ranges = {
    'Ferritin': {
        'male': {'low': 30, 'high': 400},
        'female': {'low': 15, 'high': 150}
    },
    'B12': {'low': 200, 'high': 900},
    'CRP': {'low': 0, 'high': 5},
    'Cystatin_C': {'low': 0.6, 'high': 1.0},
    'HBA1C': {'normal': 5.7, 'diabetes': 6.5},
    'AFP': {'low': 0, 'mild_high': 10, 'clinically_high': 20}
}


def categorize_result(test, value, sex=None):
    try:
        if test == 'Ferritin':
            limits = ranges['Ferritin']['male' if str(
                sex).upper().startswith('M') else 'female']
            if value < limits['low']:
                return 'Low Abnormal'
            elif value > limits['high']:
                return 'High Abnormal'
            else:
                return 'Normal'

        elif test == 'B12':
            if value < ranges['B12']['low']:
                return 'Low Abnormal'
            elif value > ranges['B12']['high']:
                return 'High Abnormal'
            else:
                return 'Normal'

        elif test == 'CRP':
            if value > ranges['CRP']['high']:
                return 'High Abnormal'
            else:
                return 'Normal'

        elif test == 'Cystatin_C':
            if value > ranges['Cystatin_C']['high']:
                return 'High Abnormal'
            else:
                return 'Normal'

        elif test == 'HBA1C':
            if value < ranges['HBA1C']['normal']:
                return 'Normal'
            elif value < ranges['HBA1C']['diabetes']:
                return 'Pre-diabetes'
            else:
                return 'Diabetes'

        elif test == 'AFP':
            if value <= ranges['AFP']['mild_high']:
                return 'Normal'
            elif value <= ranges['AFP']['clinically_high']:
                return 'Mildly High'
            else:
                return 'Clinically Significant'

        return 'Unknown'

    except Exception:
        return 'Error in classification'


@app.route('/home')
def home():
    return "Hello World! Testing on the server MHOC"


@app.route('/predict', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        input_df = pd.DataFrame([data])
        results = {}

        for name, model in models.items():
            req_features = features[name]
            missing = [f for f in req_features if f not in input_df.columns]
            if missing:
                results[name] = f"Missing features: {missing}"
                continue

            X = input_df[req_features]
            prediction = float(model.predict(X)[0])
            sex = input_df.get('sex', [None])[0]
            category = categorize_result(name, prediction, sex)

            results[name] = {
                'value': prediction,
                'status': category
            }

        return jsonify({
            'predictions': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
