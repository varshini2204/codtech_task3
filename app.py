from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HTML template
html = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        input[type=text], select { width: 100%; padding: 10px; margin: 6px 0 12px; }
        input[type=submit] { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: #e3ffe3; border: 1px solid #b2ffb2; }
    </style>
</head>
<body>
    <h2>🏡 House Price Prediction</h2>
    <form method="POST">
        Area (sq ft): <input type="text" name="area" required><br>
        Bedrooms: <input type="text" name="bedrooms" required><br>
        Bathrooms: <input type="text" name="bathrooms" required><br>
        Stories: <input type="text" name="stories" required><br>
        Main Road (yes=1 / no=0): <input type="text" name="mainroad" required><br>
        Guest Room (yes=1 / no=0): <input type="text" name="guestroom" required><br>
        Basement (yes=1 / no=0): <input type="text" name="basement" required><br>
        Hot Water Heating (yes=1 / no=0): <input type="text" name="hotwaterheating" required><br>
        Air Conditioning (yes=1 / no=0): <input type="text" name="airconditioning" required><br>
        Parking Spaces: <input type="text" name="parking" required><br>
        Preferred Area (yes=1 / no=0): <input type="text" name="prefarea" required><br>
        Furnishing Status (0=Unfurnished, 1=Semi-Furnished, 2=Furnished): <input type="text" name="furnishingstatus" required><br>
        <input type="submit" value="Predict Price">
    </form>

    {% if prediction is not none %}
    <div class="result">
        <h3>💰 Predicted Price: ₹ {{ prediction }} </h3>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            data = [
                float(request.form['area']),
                int(request.form['bedrooms']),
                int(request.form['bathrooms']),
                int(request.form['stories']),
                int(request.form['mainroad']),
                int(request.form['guestroom']),
                int(request.form['basement']),
                int(request.form['hotwaterheating']),
                int(request.form['airconditioning']),
                int(request.form['parking']),
                int(request.form['prefarea']),
                int(request.form['furnishingstatus']),
            ]
            data = np.array(data).reshape(1, -1)
            scaled_data = scaler.transform(data)
            price = model.predict(scaled_data)[0]
            prediction = f"{int(price):,}"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(html, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
