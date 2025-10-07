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
    'HBA1C': ['age','fpg','triglycerides','cholesterol_total','hdl','ldl','alt','ast','hb','hct','rbc','mcv','mch','mchc'],
    'AFP': ['age','sex','alt','ast','alp','ggt','bilirubin_total','bilirubin_direct','albumin']
}

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
            prediction = model.predict(X)[0]
            results[name] = float(prediction)

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
