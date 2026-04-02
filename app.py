from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

modelo = joblib.load("modelo/modelo_regresion.pkl")

@app.get("/")
def home():
    return {"mensaje": "API funcionando"}

@app.post("/predict")
def predict(data: dict):
    valores = np.array(data["input"]).reshape(1, -1)
    pred = modelo.predict(valores)
    return {"prediccion": pred.tolist()}