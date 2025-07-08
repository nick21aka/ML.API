from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# Inicializar app
app = FastAPI()

# Cargar modelos
clf_model = joblib.load("modelo/clf_survived_AUC>80.joblib")
reg_model = joblib.load("modelo/reg_team_equip_R2>80.joblib")

# Esquema de entrada para los modelos
class InputData(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "✅ API de Machine Learning activa"}

@app.post("/predict/clasificacion")
def predict_classification(data: InputData):
    pred = clf_model.predict([data.features])
    return {"prediction": int(pred[0])}

@app.post("/predict/regresion")
def predict_regression(data: InputData):
    pred = reg_model.predict([data.features])
    return {"prediction": float(pred[0])}
