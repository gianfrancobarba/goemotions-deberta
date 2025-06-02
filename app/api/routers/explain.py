from fastapi import APIRouter, HTTPException
from api.schemas.input_output import PredictionRequest
from api.services.explain_service import explain_emotions

router = APIRouter()

@router.post("/explain")
def explain_text(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")
    return explain_emotions(request.text)
