# predictor.py

"""
predictor.py

Fornisce la funzione `predict_all` che restituisce, per una lista di testi,
le probabilità di ciascuna emozione del modello DeBERTa.
"""

from typing import List, Dict
import torch

from model.inference.predict import tokenizer, model, device, GOEMOTIONS_LABELS
from model.explainability.config_surrogate import MAX_LEN


def predict_all(texts: List[str]) -> List[Dict[str, float]]:
    """
    Prende una lista di testi e restituisce, per ciascuno, un dict
    con le probabilità (0.0–1.0) di tutte le emozioni.

    Args:
      texts: lista di stringhe.

    Returns:
      Lista di dizionari emozione->probabilità.
    """
    results: List[Dict[str, float]] = []
    model.eval()

    for text in texts:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        prediction = {
            label: float(round(prob, 3))
            for label, prob in zip(GOEMOTIONS_LABELS, probs)
        }
        results.append(prediction)

    return results