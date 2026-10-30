"""
Step 2: FastAPI backend.

Run from the project root:
    uvicorn backend.main:app --reload
"""
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# <project>/models  (this file is in <project>/backend, so go up one level)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Load both models ONCE when the server starts (not on every request)
regression_model = joblib.load(MODELS_DIR / "regression_model.joblib")
classification_model = joblib.load(MODELS_DIR / "classification_model.joblib")

# The classifier returns 0, 1 or 2; this list turns that number into a name
SPECIES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="ML Prediction API")

# Allow the HTML page (a different "origin") to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Input shapes: FastAPI checks incoming JSON against these classes ---
class RegressionInput(BaseModel):
    med_inc: float      # median income (in $10,000s)
    house_age: float    # median house age (years)
    ave_rooms: float    # average rooms per household
    ave_occup: float    # average occupants per household


class ClassificationInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    # The model expects a 2D list: one row per sample, columns in training order
    row = [[data.med_inc, data.house_age, data.ave_rooms, data.ave_occup]]
    value = regression_model.predict(row)[0]          # result is in $100,000s
    return {"predicted_value_usd": round(float(value) * 100_000, 2)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    row = [[data.sepal_length, data.sepal_width,
            data.petal_length, data.petal_width]]
    class_index = int(classification_model.predict(row)[0])
    confidence = float(classification_model.predict_proba(row)[0][class_index])
    return {"species": SPECIES[class_index], "confidence": round(confidence, 3)}
