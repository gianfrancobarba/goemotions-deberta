from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.src.predict import predict_emotions

app = FastAPI()

# ✅ CORS: permetti richieste da qualsiasi origine (anche da altro container)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # oppure specifica "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")
    result = predict_emotions(input_data.text)
    return {
        "emotions": result.get("emotions", {}),
        "sentiment": result.get("sentiment", "neutral")
    }

@app.get("/")
def root():
    return {"message": "GoEmotions API is running"}
