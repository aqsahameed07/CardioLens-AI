import os
from datetime import datetime, timezone

import joblib
import pandas as pd
import shap
from bson import ObjectId

from api.database import assessments_collection


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_tuned.pkl",
)

model = joblib.load(MODEL_PATH)

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

# SHAP explainer for the trained Random Forest
explainer = shap.TreeExplainer(model)


def get_shap_explanation(input_data: pd.DataFrame):
    """
    Generate SHAP values for the submitted patient.

    For binary classification, we use the SHAP
    contribution for Class 1.
    """

    shap_values = explainer.shap_values(input_data)

    # Different SHAP versions return different structures.
    if isinstance(shap_values, list):
        # Older SHAP versions:
        # [class_0_values, class_1_values]
        values = shap_values[1][0]

    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # Newer SHAP versions:
        # (samples, features, classes)
        values = shap_values[0, :, 1]

    else:
        # Fallback for a 2D output
        values = shap_values[0]

    explanation = []

    for feature, value, shap_value in zip(
        FEATURES,
        input_data.iloc[0].values,
        values,
    ):
        explanation.append(
            {
                "feature": feature,
                "value": float(value),
                "shap_value": float(shap_value),
            }
        )

    # Show the features with the largest
    # absolute contribution first.
    explanation.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return explanation


def predict_heart_disease(data, user_id: str):

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

    # -----------------------------
    # Model Prediction
    # -----------------------------

    prediction = int(
        model.predict(input_data)[0]
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    result = (
        "Higher likelihood of heart disease"
        if prediction == 1
        else "Lower likelihood of heart disease"
    )

    # -----------------------------
    # SHAP Explanation
    # -----------------------------

    explanation = get_shap_explanation(
        input_data
    )

    # -----------------------------
    # Save Assessment
    # -----------------------------

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

        "model": "Random Forest Tuned",

        "explanation": explanation,

        "created_at": datetime.now(timezone.utc),
    }

    mongo_result = assessments_collection.insert_one(
        assessment
    )

    return {
        "id": str(mongo_result.inserted_id),
        "prediction": prediction,
        "probability": probability,
        "result": result,
        "explanation": explanation,
    }