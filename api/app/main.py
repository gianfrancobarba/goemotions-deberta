# api/app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.src.predict import predict_emotions  # <-- usa la tua funzione

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")
    result = predict_emotions(input_data.text)
    return {"emotions": result}

@app.get("/")
def root():
    return {"message": "GoEmotions API is running"}
