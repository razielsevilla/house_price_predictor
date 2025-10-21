from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load data for reference averages
df = pd.read_csv("kc_house_data.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from form
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
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]

    # Get ZIP code for visualization
    zipcode = int(request.form['zipcode'])
    avg_zip_price = df[df['zipcode'] == zipcode]['price'].mean()

    # --- Create a small bar chart ---
    plt.figure(figsize=(4, 3))
    bars = ['Predicted Price', f'Average in {zipcode}']
    values = [prediction, avg_zip_price if not np.isnan(avg_zip_price) else 0]
    colors = ['#007bff', '#00c851']
    plt.bar(bars, values, color=colors)
    plt.title('Price Comparison')
    plt.ylabel('Price ($)')
    plt.tight_layout()

    # Save plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    prediction_text = f"🏠 Predicted House Price: ${prediction:,.2f}"
    return render_template('index.html', prediction_text=prediction_text, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
