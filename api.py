from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib

# ==== Carga de modelos ====
# Asegúrate de que los .joblib estén en la misma carpeta
try:
    clf = joblib.load("clf_survived_AUC>80.joblib")  # modelo de clasificación (30 features)
    reg = joblib.load("reg_team_equip_R2>80.joblib")  # modelo de regresión (26 features)
except FileNotFoundError as err:
    # Si falta un archivo, la API no arrancará y Render mostrará el traceback
    raise RuntimeError(f"Modelo faltante: {err}")


app = FastAPI(
    title="CS:GO ML API",
    description="Clasificación de supervivencia (30 features) y regresión de valor de equipo (26 features)",
    version="1.0.0"
)


# ============ Esquemas de entrada ============
# Clasificación → 30 valores numéricos (orden EXACTO del ColumnTransformer)
class ClfInput(BaseModel):
    features: List[float] = Field(..., min_length=30, max_length=30)

# Regresión → 26 valores numéricos (orden EXACTO del ColumnTransformer)
class RegInput(BaseModel):
    features: List[float] = Field(..., min_length=26, max_length=26)


# ============ Endpoints ============
@app.post("/predecir/supervivencia")
def predecir_supervivencia(data: ClfInput):
    """Devuelve probabilidad y etiqueta (0/1) de supervivencia."""
    try:
        proba = clf.predict_proba([data.features])[0, 1]
        label = int(proba >= 0.5)
        return {
            "probability": round(float(proba), 3),
            "label": label
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predecir/valor_del_equipo")
def predecir_valor_equipo(data: RegInput):
    """Devuelve el valor inicial del equipo (dinero)."""
    try:
        value = reg.predict([data.features])[0]
        return {
            "equipment_value": round(float(value), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
