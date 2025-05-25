from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained models
lr_hardness_model = joblib.load('lr_hardness_model.pkl')
rf_oxidation_model = joblib.load('rf_oxidation_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    material = request.form['material']
    current = float(request.form['current'])
    heat_input = float(request.form['heat_input'])
    carbon = float(request.form['carbon'])
    manganese = float(request.form['manganese'])
    soaking_time = float(request.form['soaking_time'])

    # Encode material
    material_encoded = 0 if material == 'EN-8' else 1

    # Prepare input data for hardness prediction
    hardness_input = pd.DataFrame({
        'Material': [material_encoded],
        'Current': [current],
        'Heat_Input': [heat_input],
        'Carbon': [carbon],
        'Manganese': [manganese]
    })

    # Prepare input data for oxidation prediction
    oxidation_input = pd.DataFrame({
        'Material': [material_encoded],
        'Current': [current],
        'Heat_Input': [heat_input],
        'Soaking_Time': [soaking_time],
        'Carbon': [carbon],
        'Manganese': [manganese]
    })

    # Make predictions
    hardness_pred = lr_hardness_model.predict(hardness_input)[0]
    oxidation_pred = rf_oxidation_model.predict(oxidation_input)[0]

    return render_template('result.html', hardness=hardness_pred, oxidation=oxidation_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
