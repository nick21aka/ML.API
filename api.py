%%writefile api.py
# 👇 Aquí va el contenido completo del archivo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib

clf = joblib.load("clf_survived_AUC_80.joblib")
reg = joblib.load("reg_team_equip_R2_80.joblib")

app = FastAPI()

class ClfInput(BaseModel):
    features: conlist(float, min_items=17, max_items=17)

class RegInput(BaseModel):
    features: conlist(float, min_items=22, max_items=22)

@app.post("/predict/survival")
def predict_survival(data: ClfInput):
    try:
        proba = clf.predict_proba([data.features])[0, 1]
        return {"probability": float(round(proba, 3)), "label": int(proba >= 0.5)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/equipment_value")
def predict_equipment(data: RegInput):
    try:
        value = reg.predict([data.features])[0]
        return {"equipment_value": float(round(value, 2))}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
