# api.py  –  FastAPI con dos endpoints: clasificación y regresión
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib

# ====================  CARGAR MODELOS  ====================
try:
    clf = joblib.load("clf_survived_AUC_80.joblib")         # clasificador (AUC > 0.80)
    reg = joblib.load("reg_team_equip_R2_80.joblib")        # regresor (R² > 0.80)
except FileNotFoundError as e:
    # Render mostrará el traceback si alguno falta
    raise RuntimeError(f"Modelo faltante: {e}")

# ====================  APP FASTAPI  ========================
app = FastAPI(
    title="CS:GO ML API",
    description="Predice supervivencia de ronda (clasificación) y valor de equipo (regresión)",
    version="1.0"
)

# ====================  ESQUEMAS DE ENTRADA  ================

class ClfInput(BaseModel):
    # 17 números EXACTOS en el mismo orden que espera tu pipeline de clasificación
    features: List[float] = Field(..., min_length=17, max_length=17)

class RegInput(BaseModel):
    # 22 números EXACTOS en el mismo orden que espera tu pipeline de regresión
    features: List[float] = Field(..., min_length=22, max_length=22)

# ====================  ENDPOINTS  ==========================

@app.post("/predict/survival")
def predict_survival(data: ClfInput):
    """Devuelve probabilidad y etiqueta (0/1) de sobrevivir la ronda."""
    try:
        proba = clf.predict_proba([data.features])[0, 1]
        label = int(proba >= 0.5)
        return {"probability": round(float(proba), 3), "label": label}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))

@app.post("/predict/equipment_value")
def predict_equipment_value(data: RegInput):
    """Devuelve el valor inicial del equipo (dinero)."""
    try:
        value = reg.predict([data.features])[0]
        return {"equipment_value": round(float(value), 2)}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
