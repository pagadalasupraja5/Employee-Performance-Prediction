from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/performance_model.pkl", "rb") as f:
    model, le = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["attendance"]),
        float(request.form["years"]),
        float(request.form["training"]),
        float(request.form["rating"]),
        float(request.form["overtime"])
    ]

    prediction = model.predict([features])
    result = le.inverse_transform(prediction)[0]

    return render_template("index.html", prediction_text=f"Performance: {result}")

if __name__ == "__main__":
    app.run(debug=True)