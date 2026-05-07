from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

def preprocess_input(data):
    """Preprocess input data to match model requirements"""
    # Convert to numeric
    data['age'] = float(data['age'])
    data['bmi'] = float(data['bmi'])
    data['children'] = int(data['children'])
    
    # Encode categorical variables
    data['sex'] = 1 if data['sex'] == 'female' else 0
    data['smoker'] = 1 if data['smoker'] == 'yes' else 0
    
    # Create region dummies
    region_dummies = {
        'region_northwest': 1 if data['region'] == 'northwest' else 0,
        'region_southeast': 1 if data['region'] == 'southeast' else 0,
        'region_southwest': 1 if data['region'] == 'southwest' else 0
    }
    data.update(region_dummies)
    del data['region']
    
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Convert to DataFrame with correct column order
        columns = ['age', 'sex', 'bmi', 'children', 'smoker', 
                  'region_northwest', 'region_southeast', 'region_southwest']
        input_df = pd.DataFrame([processed_data])[columns]
        
        # Load model
        import joblib
        model = joblib.load('models/random_forest_model.pkl')
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
