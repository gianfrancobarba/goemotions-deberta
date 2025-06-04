# File: app/api/services/predict_service.py

from model.inference.predict import predict_emotions as model_predict
from api.logger import log_request
import time

def predict_emotions(text: str) -> dict:
    start_time = time.perf_counter()  # Inizio cronometro

    result = model_predict(text)  # Qui avviene la vera inferenza

    end_time = time.perf_counter()
    duration = end_time - start_time 

    log_request(
        route="/predict",
        input_text=text,
        output_data={
            "result": result,
            "duration_seconds": round(duration, 4)
        }
    )
    
    return {
        "emotions": result.get("emotions", {}),
        "sentiment": result.get("sentiment", "neutral")
    }
