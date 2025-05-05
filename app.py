from flask import Flask, request, jsonify, render_template, render_template_string
from predict_output import prediction_ouput
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

country_data = pd.read_csv('./data/country.csv')
indicator_data = pd.read_csv('./data/indicator.csv')

app = Flask(__name__)

country_codes = country_data['code'].unique()
country_names = country_data['country'].unique()
tlas = indicator_data['TLA'].unique()
indicators = indicator_data['indicator'].unique()
issue_tlas = indicator_data['issue_tla'].tolist()  # List of issue_tla values

# Route to render HTML with data for dropdowns
@app.route('/')
def home():
    return render_template('index.html', countries=zip(country_codes, country_names), indicators=zip(zip(tlas,issue_tlas),indicators),)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data['year'] = int(data['year'])
    data['code'] = int(data['code'])
    # Perform prediction
    response = prediction_ouput(data)
    if response['status']==200:
        return jsonify({'predicted_epi_value': response['prediction'][0],'year': response['data']['year'].tolist(),'Epi_value':response['data']['Epi_value'].tolist()})
    else:
        return jsonify({'predicted_epi_value': 'Year must be between 1997 and 2100','year': response['data']['year'].tolist(),'Epi_value':response['data']['Epi_value'].tolist()})
if __name__ == '__main__':
    app.run(debug=True)
