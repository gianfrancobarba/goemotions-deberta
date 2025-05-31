from fastapi import APIRouter, HTTPException
from app.api.schemas.input_output import PredictionRequest, PredictionResponse
from api.services.predict_service import predict_emotions

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")
    return predict_emotions(request.text)
