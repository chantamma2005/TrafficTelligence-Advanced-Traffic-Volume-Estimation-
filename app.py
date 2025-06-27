import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and the scaler
model = pickle.load(open('G:/AIML/ML projects/Traffic_volume/model.pkl', 'rb'))
scale = pickle.load(open('C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/scale.pkl', 'rb'))

@app.route('/')  # Home page
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST", "GET"])  # Prediction endpoint
def predict():
    try:
        # 1. Read inputs from form and convert to float
        input_feature = [float(x) for x in request.form.values()]
        features_values = np.array(input_feature)

        # 2. Column names (MUST match training order)
        names = ['holiday', 'temp', 'rain', 'snow', 'weather', 
                 'year', 'month', 'day', 'hours', 'minutes', 'seconds']

        # 3. Create DataFrame from inputs
        y = pd.DataFrame([features_values], columns=names)

        # 4. Apply the scaler (do NOT fit again)
        data_scaled = scale.transform(y)

        # 5. Predict using loaded model
        prediction = model.predict(data_scaled)

        # 6. Return result
        text = "Estimated Traffic Volume is: "
        return render_template("index.html", prediction_text=text + str(round(prediction[0], 2)))

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)