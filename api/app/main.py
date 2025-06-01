# File: main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import del modulo di predizione
from model.src.predict import predict_emotions
# Import del wrapper che utilizza explain_pipeline (che a sua volta usa il nuovo verbalizer)
from model.explainability.explain_pipeline import explain as explain_pipeline

app = FastAPI()

# Configurazione CORS (in sviluppo permettiamo da qualsiasi origine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # in produzione limita a ["http://localhost:3000"] o domini specifici
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    """
    Endpoint /predict:
      Input: JSON { "text": "..." }
      Output: { emotions: {...}, sentiment: "..." }
    """
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")

    result = predict_emotions(input_data.text)
    return {
        "emotions": result.get("emotions", {}),
        "sentiment": result.get("sentiment", "neutral")
    }

@app.post("/explain")
def explain_text(input_data: TextInput):
    """
    Endpoint /explain:
      Input: JSON { "text": "..." }
      Output: l'intero JSON restituito da explain_pipeline, che include:
        - original_text
        - predictions
        - features
        - surrogates
        - explanations
    """
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")

    # Invoca l’intera pipeline di explain (explain_with_surrogates + verbalizer)
    result = explain_pipeline(input_data.text)

    # Restituisci tutto quanto generato (incluso explanations.simple_explanation e i dettagli tecnici)
    return result

@app.get("/")
def root():
    return {"message": "GoEmotions API is running"}
