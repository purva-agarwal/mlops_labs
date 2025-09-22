import logging

from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, field_validator
from predict import predict_data
from typing import Dict

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

    @classmethod
    def validate_range(cls, value: float, name: str) -> float:
        if not (0 <= value <= 2000): 
            raise ValueError(f"{name} must be between 0 and 2000")
        return value

    @field_validator(*__annotations__.keys())
    @classmethod
    def check_feature_range(cls, v, info):
        return cls.validate_range(v, info.field_name)

class WineResponse(BaseModel):
    response:str
    confidence: str
    probabilities: Dict[str, str]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.get("/version", status_code=status.HTTP_200_OK)
async def get_version():
    return {"model_version": "1.0.0"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315_of_diluted_wines,
            wine_features.proline
        ]]

        label, confidence, probabilities  = predict_data(features)
        return WineResponse(response=label, confidence=confidence, probabilities=probabilities)
    
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


    