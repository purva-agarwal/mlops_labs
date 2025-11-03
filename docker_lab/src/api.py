from fastapi import FastAPI
from pydantic import BaseModel
from src.train import train_model

app = FastAPI(title="Iris Model API")

# Train model when container starts
model = train_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Prediction API is running!"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}
