from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form input
        features = [
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['sqft_living']),
            float(request.form['sqft_lot']),
            float(request.form['floors']),
            float(request.form['waterfront']),
            float(request.form['view']),
            float(request.form['condition']),
            float(request.form['grade']),
            float(request.form['sqft_above']),
            float(request.form['sqft_basement']),
            float(request.form['yr_built']),
            float(request.form['yr_renovated']),
            float(request.form['zipcode']),
            float(request.form['lat']),
            float(request.form['long']),
            float(request.form['sqft_living15']),
            float(request.form['sqft_lot15'])
        ]

        # Scale and predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        return render_template('index.html', prediction_text=f'🏠 Predicted House Price: ${prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
