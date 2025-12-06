from fastapi import FastAPI, Request
import pickle
import pandas as pd
app = FastAPI()

# Use pipeline_v1.bin instead
with open('pipeline_v1.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.post("/predict")
async def predict(request: Request):
    test = await request.json()
    X_test = dv.transform([test])
    y_pred = model.predict_proba(X_test)[0, 1]
    churn = y_pred >= 0.5
    return {"churn_probability": float(y_pred), "churn": bool(churn)}
