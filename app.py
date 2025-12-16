from flask import Flask, render_template, request, jsonify
import fasttext
import joblib
import numpy as np

app = Flask(__name__)

# ------------------------
# Load Models and Encoder
# ------------------------
fasttext_model = fasttext.load_model("models/roman_urdu_fasttext_model.bin")
label_encoder = joblib.load("models/label_encoder.pkl")

logistic_model = joblib.load("models/logistic_regression_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")

# ------------------------
# Text to Vector Function
# ------------------------
def get_sentence_vector(text):
    """
    Convert text to a FastText averaged word vector.
    """
    text = str(text).lower().strip()
    words = text.split()
    vectors = []

    for w in words:
        try:
            vectors.append(fasttext_model.get_word_vector(w))
        except Exception:
            continue

    if len(vectors) == 0:
        return np.zeros(fasttext_model.get_dimension())

    return np.mean(vectors, axis=0)

# ------------------------
# Routes
# ------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Incoming JSON:", data)

        text = data.get("text", "")
        model_choice = data.get("model", "logistic_regression")

        if not text.strip():
            return jsonify({"error": "Empty text"}), 400

        # Convert to vector
        vec = get_sentence_vector(text).reshape(1, -1)

        # Select model
        if model_choice == "svm":
            model = svm_model
        elif model_choice == "random_forest":
            model = rf_model
        else:
            model = logistic_model

        # Predict
        pred = model.predict(vec)[0]

        # Convert numeric label -> original label
        label = label_encoder.inverse_transform([pred])[0]

        # Normalize label output
        label_lower = str(label).lower().strip()
        if label_lower in ["1", "abusive", "true", "abuse", "offensive"]:
            final_label = "Abusive Comment"
        else:
            final_label = "Non-Abusive Comment"

        print(f"Predicted label: {label} → {final_label}")
        return jsonify({"prediction": final_label})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import traceback
    try:
        print("🚀 Starting Flask server...")
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print("❌ Server crashed:")
        traceback.print_exc()
