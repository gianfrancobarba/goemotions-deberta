# model/src/predict.py

import os
import torch
import logging
from typing import Dict
from transformers import AutoTokenizer, AutoConfig
from model.src.train_utils import CustomMultiLabelModel
from config.config import CFG
from collections import defaultdict



def infer_sentiment(emotions: Dict[str, float]) -> str:
    scores = defaultdict(float)
    for emotion, score in emotions.items():
        if emotion in POSITIVE:
            scores["positive"] += score
        elif emotion in NEGATIVE:
            scores["negative"] += score
        else:
            scores["neutral"] += score

    # Prende il sentiment col punteggio totale più alto
    if not scores:
        return "neutral"
    return max(scores.items(), key=lambda x: x[1])[0]


# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Etichette (GoEmotions v1.0, senza 'neutral' se non usata)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

POSITIVE = {"joy", "love", "gratitude", "relief", "amusement", "optimism", "admiration", "pride", "approval", "caring", "excitement", "contentment"}
NEGATIVE = {"anger", "fear", "sadness", "disappointment", "disgust", "remorse", "grief", "annoyance", "disapproval", "embarrassment", "nervousness"}
NEUTRAL  = {"realization", "curiosity", "confusion", "surprise"}

# Se il numero di etichette è 27, rimuovi 'neutral' da GOEMOTIONS_LABELS
if CFG.num_labels == 27 and len(GOEMOTIONS_LABELS) == 28:
    GOEMOTIONS_LABELS.remove("neutral")

# Caricamento modello/tokenizer una volta sola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_dir)
config = AutoConfig.from_pretrained(CFG.model_dir)

model = CustomMultiLabelModel(
    model_name=CFG.model_name,
    num_labels=CFG.num_labels,
    pos_weight=torch.ones(CFG.num_labels)  # usato solo nel training
)
state_dict = torch.load(os.path.join(CFG.model_dir, "pytorch_model.bin"), map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if 'loss_fn' not in k}
model.load_state_dict(filtered_state_dict, strict=False)
model.to(device)
model.eval()

# Definizione delle emozioni positive e negative
def infer_sentiment(emotions: Dict[str, float]) -> str:
    scores = defaultdict(float)
    for emotion, score in emotions.items():
        if emotion in POSITIVE:
            scores["positive"] += score
        elif emotion in NEGATIVE:
            scores["negative"] += score
        else:
            scores["neutral"] += score

    # Prende il sentiment col punteggio totale più alto
    if not scores:
        return "neutral"
    return max(scores.items(), key=lambda x: x[1])[0]


# 🔧 Funzione chiamabile da FastAPI
def predict_emotions(text: str) -> Dict[str, float]:
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=CFG.max_length
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Restituisce solo etichette con probabilità ≥ threshold, convertendo prob a float
    result = {
        label: round(float(prob), 3)  # 🔁 cast esplicito a float puro
        for label, prob in zip(GOEMOTIONS_LABELS, probs)
        if prob >= CFG.threshold
    }
    return {
        "emotions": result,
        "sentiment": infer_sentiment(result)
    }


if __name__ == "__main__":
    while True:
    # Input dell'utente
        text = input("Inserisci una frase per l'analisi delle emozioni (exit per uscire): ")
        result = predict_emotions(text)
        if text.lower() == "exit":
            break
        print("Emozioni rilevate:")
        for emotion, score in result.items():
            print(f"  {emotion}: {score}")
