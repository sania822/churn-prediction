from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask import render_template

# Create Flask app
app = Flask(__name__)

# Load trained model once when server starts
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(" Error loading model:", e)


# Home route (check API running)
@app.route("/")
def home():
    return jsonify({
        "message": "Customer Churn Prediction API is running"
    })

@app.route("/ui")
def ui():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "Send data as {'features': [...]}"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "churn_probability": float(probability) if probability is not None else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run()
