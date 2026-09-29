# api/predictor.py

import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
from bson import ObjectId

from api.database import assessments_collection


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Single artifact: Pipeline(StandardScaler + RandomForestClassifier)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")


# ============================================================
# FEATURE ORDER (must match training)
# ============================================================

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# ============================================================
# LABEL CONVENTION NOTE
# ============================================================
#
# The training CSV (../data/raw/heart.csv) uses a NON-STANDARD
# convention where:
#
#     target = 0  ->  DISEASE present
#     target = 1  ->  NO disease
#
# The standard UCI Cleveland convention is the opposite.
#
# Therefore, at inference we invert the model's output so the
# rest of the system can use the standard convention:
#
#     prediction = 1  ->  DISEASE
#     prediction = 0  ->  NO disease
#
# If you retrain on a correctly-labeled CSV, remove the
# `1 -` inversion below and switch `probabilities[0]` back to
# `probabilities[1]`.
# ============================================================


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("CARDIOLENS - LOADING MODEL")
print("=" * 70)

model = joblib.load(MODEL_PATH)

print(f"Model loaded: {MODEL_PATH}")
print(f"Model type: {type(model).__name__}")

if hasattr(model, "steps"):
    print("Pipeline steps:")
    for name, step in model.steps:
        print(f"  - {name}: {type(step).__name__}")

# The RF step is what we explain with SHAP
rf_model = model.named_steps["model"] if hasattr(model, "steps") else model

print("=" * 70)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(rf_model)


def get_shap_explanation(input_scaled, input_raw):
    """
    Generate SHAP explanations for the DISEASE class.

    Because the training labels are inverted (target=0 means
    disease), SHAP values for the model's class 0 correspond to
    "disease" contributions. We flip them here so a positive
    shap_value always means "pushes toward disease", matching
    the convention used in the prediction.
    """
    shap_values = explainer.shap_values(input_scaled)

    # SHAP output format varies by version:
    #   - list  [class_0_values, class_1_values]  (older)
    #   - ndarray [n_samples, n_features, n_classes]  (newer)
    if isinstance(shap_values, list):
        # Flipped: class 0 is disease in our training data
        values = shap_values[0][0]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        values = shap_values[0, :, 0]
    else:
        values = shap_values[0]

    explanation = []
    for feature, raw_value, shap_value in zip(
        FEATURES, input_raw.iloc[0].values, values
    ):
        explanation.append({
            "feature": feature,
            "value": float(raw_value),
            "shap_value": float(shap_value),
        })

    explanation.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    return explanation


# ============================================================
# PREDICTION
# ============================================================

def predict_heart_disease(data, user_id: str):
    """
    Run a heart-disease prediction, log it, and persist the result
    to MongoDB.

    Parameters
    ----------
    data : pydantic model
        Must expose: age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    user_id : str
        MongoDB ObjectId string of the user.
    """

    print("\n\n")
    print("=" * 70)
    print("CARDIOLENS - NEW PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. RAW INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal,
        ]],
        columns=FEATURES,
    )

    print("\n[1] RAW INPUT")
    print("-" * 70)
    print(input_data.to_string(index=False))
    print("-" * 70)

    # --------------------------------------------------------
    # 2. PREDICT (Pipeline scales internally)
    # --------------------------------------------------------
    #
    # NOTE: model was trained on a CSV where target=0 = disease,
    # target=1 = healthy. We invert here so the returned
    # `prediction` follows the standard convention:
    #     prediction = 1  ->  disease
    #     prediction = 0  ->  no disease
    #
    # In this flipped scheme:
    #     probabilities[0] = P(disease)      <- what we want
    #     probabilities[1] = P(no disease)
    # --------------------------------------------------------

    raw_pred = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    prediction = 1 - raw_pred
    probability = float(probabilities[0])   # P(disease) in flipped data

    print("\n[2] MODEL PREDICTION")
    print("-" * 70)
    print(f"Raw model class         : {raw_pred}  (0=disease, 1=healthy in training data)")
    print(f"Flipped prediction       : {prediction}  (1=disease, 0=no disease)")
    print(f"P(disease) [flipped]     : {probability:.6f}")
    print(f"P(no disease) [flipped]  : {probabilities[1]:.6f}")
    print(f"P(disease) percentage    : {probability * 100:.2f}%")

    if prediction == 1:
        result = "Higher likelihood of heart disease"
    else:
        result = "Lower likelihood of heart disease"

    print(f"Result                   : {result}")
    print("-" * 70)

    # --------------------------------------------------------
    # 3. SHAP (scale input manually — SHAP needs scaled features)
    # --------------------------------------------------------

    scaler = model.named_steps["scaler"]
    input_scaled = pd.DataFrame(
        scaler.transform(input_data),
        columns=FEATURES,
    )

    print("\n[3] SCALED INPUT (for SHAP)")
    print("-" * 70)
    print(input_scaled.to_string(index=False))
    print("-" * 70)

    explanation = get_shap_explanation(input_scaled, input_data)

    print("\n[4] SHAP - TOP 5 CONTRIBUTORS (positive = pushes toward disease)")
    print("-" * 70)
    for item in explanation[:5]:
        sign = "+" if item["shap_value"] >= 0 else "-"
        print(
            f"  {item['feature']:10s} "
            f"value={item['value']:>8.2f}  "
            f"SHAP={sign}{abs(item['shap_value']):.4f}"
        )
    print("-" * 70)

    # --------------------------------------------------------
    # 4. SAVE TO MONGODB
    # --------------------------------------------------------

    print("\n[5] SAVING ASSESSMENT TO MONGODB")
    print("-" * 70)

    assessment = {
        "user_id": ObjectId(user_id),
        "patient_data": {
            "age": data.age,
            "sex": data.sex,
            "cp": data.cp,
            "trestbps": data.trestbps,
            "chol": data.chol,
            "fbs": data.fbs,
            "restecg": data.restecg,
            "thalach": data.thalach,
            "exang": data.exang,
            "oldpeak": data.oldpeak,
            "slope": data.slope,
            "ca": data.ca,
            "thal": data.thal,
        },
        "prediction": prediction,
        "probability": probability,
        "result": result,
        "model": "Random Forest Tuned (Pipeline)",
        "explanation": explanation,
        "created_at": datetime.now(timezone.utc),
    }

    mongo_result = assessments_collection.insert_one(assessment)
    assessment_id = str(mongo_result.inserted_id)

    print(f"Assessment saved. MongoDB ID: {assessment_id}")
    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)
    print("\n")

    # --------------------------------------------------------
    # 5. RETURN TO API LAYER
    # --------------------------------------------------------

    return {
        "id": assessment_id,
        "prediction": prediction,
        "probability": probability,
        "result": result,
        "model": "Random Forest Tuned (Pipeline)",
        "explanation": explanation,
    }