from app.model.inference.predict import predict_emotions as model_predict

def predict_emotions(text: str) -> dict:
    result = model_predict(text)
    return {
        "emotions": result.get("emotions", {}),
        "sentiment": result.get("sentiment", "neutral")
    }
