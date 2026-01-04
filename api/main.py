from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict

app = FastAPI(title="Heart Disease Prediction API")

class Patient(BaseModel):
    age:int
    sex:int
    cp:int
    trestbps:int
    chol:int
    fbs:int
    restecg:int
    thalach:int
    exang:int
    oldpeak:float
    slope:int
    ca:int
    thal:int

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API Running"}

@app.post("/predict")
def make_prediction(data: Patient):
    return predict(data.dict())
