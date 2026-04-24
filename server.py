# # """
# # ML SERVER PLACEHOLDER
# # /ml/server.py

# # This is a Python Flask server that will load your trained model.pkl
# # and serve predictions to the Next.js API route at /api/predict.

# # HOW TO USE:
# # 1. Train your ML model and save it as /ml/model.pkl
# # 2. Install dependencies: pip install flask scikit-learn joblib numpy
# # 3. Run this server: python ml/server.py
# # 4. Set ML_API_URL=http://localhost:5000 in your .env file
# # 5. The Next.js backend (/api/predict/route.js) will call this server automatically

# # EXPECTED INPUT:
# # POST /predict
# # Content-Type: application/json
# # {
# #   "symptoms": [0, 1, 0, 1, 0, 0, 1, ...]  <- 132 binary values, one per symptom
# # }

# # EXPECTED OUTPUT:
# # {
# #   "top_predictions": [
# #     { "disease": "Flu", "confidence": 0.82 },
# #     { "disease": "Cold", "confidence": 0.12 },
# #     { "disease": "Typhoid", "confidence": 0.06 }
# #   ]
# # }
# # """

# # from flask import Flask, request, jsonify
# # from flask_cors import CORS  # pip install flask-cors
# # import joblib                # pip install joblib
# # import numpy as np
# # import os

# # app = Flask(__name__)
# # CORS(app)  # Allow requests from Next.js dev server

# # # ─── MODEL LOADING ────────────────────────────────────────────────────────────
# # # ↓↓↓ YOUR MODEL WILL BE LOADED HERE ↓↓↓
# # # Place your trained model.pkl in the /ml/ directory

# # MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
# # model = None

# # def load_model():
# #     global model
# #     if os.path.exists(MODEL_PATH):
# #         print(f"[ML] Loading model from {MODEL_PATH}")
# #         model = joblib.load(MODEL_PATH)
# #         print(f"[ML] Model loaded successfully: {type(model).__name__}")
# #     else:
# #         print(f"[ML] WARNING: model.pkl not found at {MODEL_PATH}")
# #         print("[ML] Server running in demo mode — place model.pkl here to enable predictions")


# # # ─── PREDICTION ENDPOINT ──────────────────────────────────────────────────────
# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         data = request.get_json()
# #         symptoms_vector = data.get("symptoms", [])

# #         if len(symptoms_vector) != 132:
# #             return jsonify({"error": f"Expected 132 symptoms, got {len(symptoms_vector)}"}), 400

# #         if model is None:
# #             # ↓ DEMO MODE: return mock predictions when model.pkl is not present
# #             return jsonify({
# #                 "top_predictions": [
# #                     {"disease": "Common Cold", "confidence": 0.76},
# #                     {"disease": "Influenza", "confidence": 0.14},
# #                     {"disease": "Allergic Rhinitis", "confidence": 0.10},
# #                 ],
# #                 "note": "Demo mode — place model.pkl in /ml/ directory"
# #             })

# #         # ↓↓↓ REAL PREDICTION LOGIC ↓↓↓
# #         # Convert input to numpy array with correct shape
# #         X = np.array(symptoms_vector).reshape(1, -1)

# #         # Get class probabilities (assumes model supports predict_proba)
# #         # If your model uses predict() instead, adjust accordingly
# #         if hasattr(model, "predict_proba"):
# #             proba = model.predict_proba(X)[0]          # shape: (n_classes,)
# #             class_names = model.classes_               # list of disease names

# #             # Sort by confidence descending, take top 3
# #             sorted_indices = np.argsort(proba)[::-1][:3]
# #             top_predictions = [
# #                 {
# #                     "disease": str(class_names[i]),
# #                     "confidence": round(float(proba[i]), 4),
# #                 }
# #                 for i in sorted_indices
# #                 if proba[i] > 0.01  # Only include predictions with >1% confidence
# #             ]
# #         else:
# #             # Fallback for models without predict_proba
# #             predicted_class = model.predict(X)[0]
# #             top_predictions = [
# #                 {"disease": str(predicted_class), "confidence": 1.0}
# #             ]

# #         return jsonify({"top_predictions": top_predictions})

# #     except Exception as e:
# #         print(f"[ML] Prediction error: {e}")
# #         return jsonify({"error": str(e)}), 500


# # @app.route("/health", methods=["GET"])
# # def health():
# #     return jsonify({
# #         "status": "ok",
# #         "model_loaded": model is not None,
# #         "model_type": type(model).__name__ if model else None,
# #     })


# # if __name__ == "__main__":
# #     load_model()
# #     port = int(os.environ.get("ML_PORT", 5000))
# #     print(f"[ML] Server starting on http://localhost:{port}")
# #     app.run(host="0.0.0.0", port=port, debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import numpy as np
# import json
# import os

# app = Flask(__name__)
# CORS(app)

# BASE = os.path.dirname(os.path.abspath(__file__))

# # ── Load everything on startup ─────────────────────────
# print("[ML] Loading model files...")

# model = joblib.load(os.path.join(BASE, "model.pkl"))
# le    = joblib.load(os.path.join(BASE, "label_encoder.pkl"))

# with open(os.path.join(BASE, "symptom_columns.json")) as f:
#     SYMPTOM_COLS = json.load(f)

# print(f"[ML] ✅ Model loaded   : {type(model).__name__}")
# print(f"[ML] ✅ Diseases       : {len(le.classes_)}")
# print(f"[ML] ✅ Symptoms       : {len(SYMPTOM_COLS)}")
# print(f"[ML] ✅ Server ready!")


# # ── Prediction endpoint ────────────────────────────────
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()

#         if "symptoms" not in data:
#             return jsonify({"error": "Missing symptoms field"}), 400

#         symptoms_vector = data["symptoms"]

#         if len(symptoms_vector) != len(SYMPTOM_COLS):
#             return jsonify({
#                 "error": f"Expected {len(SYMPTOM_COLS)} symptoms, got {len(symptoms_vector)}"
#             }), 400

#         # Convert to numpy and predict
#         X     = np.array(symptoms_vector).reshape(1, -1)
#         proba = model.predict_proba(X)[0]

#         # Get top 3 predictions
#         top3_idx = np.argsort(proba)[::-1][:3]

#         top_predictions = [
#             {
#                 "disease"   : str(le.classes_[i]),
#                 "confidence": round(float(proba[i]), 4),
#             }
#             for i in top3_idx
#             if proba[i] > 0.01
#         ]

#         return jsonify({"top_predictions": top_predictions})

#     except Exception as e:
#         print(f"[ML] Prediction error: {e}")
#         return jsonify({"error": str(e)}), 500


# # ── Health check ───────────────────────────────────────
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status"  : "ok",
#         "model"   : type(model).__name__,
#         "diseases": len(le.classes_),
#         "symptoms": len(SYMPTOM_COLS),
#     })


# if __name__ == "__main__":
#     port = int(os.environ.get("ML_PORT", 5000))
#     print(f"[ML] Running on http://localhost:{port}")
#     app.run(host="0.0.0.0", port=port, debug=False)




# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import numpy as np
# import json
# import os

# app = Flask(__name__)
# CORS(app)

# BASE = os.path.dirname(os.path.abspath(__file__))

# print("[ML] Loading model files...")

# try:
#     model = joblib.load(os.path.join(BASE, "model.pkl"))
#     le    = joblib.load(os.path.join(BASE, "label_encoder.pkl"))

#     with open(os.path.join(BASE, "symptom_columns.json")) as f:
#         SYMPTOM_COLS = json.load(f)

#     print(f"[ML] ✅ Model    : {type(model).__name__}")
#     print(f"[ML] ✅ Diseases : {len(le.classes_)}")
#     print(f"[ML] ✅ Symptoms : {len(SYMPTOM_COLS)}")

# except FileNotFoundError as e:
#     print(f"[ML] ❌ Missing file: {e}")
#     print("[ML] Make sure model.pkl, label_encoder.pkl, symptom_columns.json are in /ml/")
#     exit(1)


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()

#         if "symptoms" not in data:
#             return jsonify({"error": "Missing symptoms"}), 400

#         symptoms_vector = data["symptoms"]

#         if len(symptoms_vector) != len(SYMPTOM_COLS):
#             return jsonify({
#                 "error": f"Expected {len(SYMPTOM_COLS)} values, got {len(symptoms_vector)}"
#             }), 400

#         X     = np.array(symptoms_vector).reshape(1, -1)
#         proba = model.predict_proba(X)[0]

#         top3 = np.argsort(proba)[::-1][:3]

#         top_predictions = [
#             {
#                 "disease"   : str(le.classes_[i]),
#                 "confidence": round(float(proba[i]), 4),
#             }
#             for i in top3
#             if proba[i] > 0.01
#         ]

#         return jsonify({"top_predictions": top_predictions})

#     except Exception as e:
#         print(f"[ML] Error: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status"  : "ok",
#         "model"   : type(model).__name__,
#         "diseases": len(le.classes_),
#         "symptoms": len(SYMPTOM_COLS),
#     })


# if __name__ == "__main__":
#     port = int(os.environ.get("ML_PORT", 5000))
#     print(f"[ML] Server running → http://localhost:{port}")
#     app.run(host="0.0.0.0", port=port, debug=False)




from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

print("[ML] Loading model files...")

try:
    # ── CHANGED: load 3 models instead of 1 ──────────────────────────
    svm_model = joblib.load(os.path.join(BASE, "svm_model.pkl"))
    rf_model  = joblib.load(os.path.join(BASE, "rf_model.pkl"))
    nb_model  = joblib.load(os.path.join(BASE, "nb_model.pkl"))
    # ─────────────────────────────────────────────────────────────────

    le = joblib.load(os.path.join(BASE, "label_encoder.pkl"))

    with open(os.path.join(BASE, "symptom_columns.json")) as f:
        SYMPTOM_COLS = json.load(f)

    print(f"[ML] ✅ SVM      : loaded")
    print(f"[ML] ✅ RF       : loaded")
    print(f"[ML] ✅ NB       : loaded")
    print(f"[ML] ✅ Diseases : {len(le.classes_)}")
    print(f"[ML] ✅ Symptoms : {len(SYMPTOM_COLS)}")

except FileNotFoundError as e:
    print(f"[ML] ❌ Missing file: {e}")
    print("[ML] Make sure svm_model.pkl, rf_model.pkl, nb_model.pkl,")
    print("[ML] label_encoder.pkl, symptom_columns.json are in /ml/")
    exit(1)


# ── CHANGED: ensemble helper (replaces single model.predict_proba) ───
def ensemble_predict_proba(X):
    """Weighted average: SVM 50% + RF 30% + NB 20%"""
    return (
        0.5 * svm_model.predict_proba(X) +
        0.3 * rf_model.predict_proba(X)  +
        0.2 * nb_model.predict_proba(X)
    )
# ─────────────────────────────────────────────────────────────────────


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "symptoms" not in data:
            return jsonify({"error": "Missing symptoms"}), 400

        symptoms_vector = data["symptoms"]

        if len(symptoms_vector) != len(SYMPTOM_COLS):
            return jsonify({
                "error": f"Expected {len(SYMPTOM_COLS)} values, got {len(symptoms_vector)}"
            }), 400

        X = np.array(symptoms_vector).reshape(1, -1)

        # ── CHANGED: use ensemble instead of single model ─────────────
        proba = ensemble_predict_proba(X)[0]
        # ─────────────────────────────────────────────────────────────

        top3 = np.argsort(proba)[::-1][:3]

        top_predictions = [
            {
                "disease"   : str(le.classes_[i]),
                "confidence": round(float(proba[i]), 4),
            }
            for i in top3
            if proba[i] > 0.01
        ]

        return jsonify({"top_predictions": top_predictions})

    except Exception as e:
        print(f"[ML] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "ok",
        "model"   : "Ensemble (SVM + RF + NB)",   # ── CHANGED
        "diseases": len(le.classes_),
        "symptoms": len(SYMPTOM_COLS),
    })


if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 5000))
    print(f"[ML] Server running → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
