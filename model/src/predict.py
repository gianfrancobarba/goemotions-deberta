# model/src/predict.py

import os
import torch
import logging
from typing import Dict
from transformers import AutoTokenizer, AutoConfig
from model.src.train_utils import CustomMultiLabelModel
from config.config import CFG

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
model.load_state_dict(torch.load(os.path.join(CFG.model_dir, "deberta_model.pt"), map_location=device))
model.to(device)
model.eval()

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
    return result

if __name__ == "__main__":
    text = input("Inserisci una frase per l'analisi delle emozioni: ")
    result = predict_emotions(text)
    print("Emozioni rilevate:")
    for emotion, score in result.items():
        print(f"  {emotion}: {score}")
