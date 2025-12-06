from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# -------------------------
# Load models and data
# -------------------------
dv = joblib.load("models/dict_vectorizer.bin")
rain_clf = joblib.load("models/rain_classifier.bin")
rain_reg = joblib.load("models/rain_regressor.bin")
ecocrop = joblib.load("models/ecocrop_df.bin")

features = list(dv.feature_names_)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Weather & Crop API")

# -------------------------
# Request models
# -------------------------
class WeatherInput(BaseModel):
    T2M: float
    RH2M: float
    ALLSKY_SFC_SW_DWN: float
    month: int
    AEZ: str

class SiteInput(BaseModel):
    site_temp: float
    site_rain: float
    tolerance: float = 0.0

# -------------------------
# Crop recommendation
# -------------------------
def recommend_crops(site_temp, site_rain, tol=0.0):
    candidates = []
    for _, row in ecocrop.iterrows():
        tmin, tmax = row.get('tmin', np.nan), row.get('tmax', np.nan)
        rmin, rmax = row.get('rmin', np.nan), row.get('rmax', np.nan)

        if np.isnan([tmin, tmax, rmin, rmax]).any():
            continue

        if tmin - tol <= site_temp <= tmax + tol and rmin - tol <= site_rain <= rmax + tol:
            candidates.append(row.get('scientificname', row.get('comname', None)))

    return list(filter(None, candidates))

# -------------------------
# Endpoints
# -------------------------
@app.post("/predict_weather")
def predict_weather(data: WeatherInput):
    X = dv.transform([data.dict()])
    prob = float(rain_clf.predict_proba(X)[0][1])
    amt = float(rain_reg.predict(X)[0])
    return {"rain_probability": prob, "predicted_precipitation": amt}

@app.post("/recommend_crops")
def recommend(data: SiteInput):
    crops = recommend_crops(data.site_temp, data.site_rain, data.tolerance)
    return {"recommended_crops": crops}
