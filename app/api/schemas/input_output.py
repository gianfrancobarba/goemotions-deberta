from pydantic import BaseModel
from typing import Dict

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    emotions: Dict[str, float]
    sentiment: str
