import os
import json
import torch
import logging
from typing import Dict
from collections import defaultdict

from transformers import AutoTokenizer, AutoConfig

from app.config.loader import CFG
from app.model.training.train_utils import CustomMultiLabelModel

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === Etichette GoEmotions v1.0 ===
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

POSITIVE = {
    "joy", "love", "gratitude", "relief", "amusement", "optimism",
    "admiration", "pride", "approval", "caring", "excitement", "contentment"
}
NEGATIVE = {
    "anger", "fear", "sadness", "disappointment", "disgust", "remorse",
    "grief", "annoyance", "disapproval", "embarrassment", "nervousness"
}
NEUTRAL = {"realization", "curiosity", "confusion", "surprise"}

# Rimozione "neutral" se esclusa
if CFG.model.num_labels == 27 and len(GOEMOTIONS_LABELS) == 28:
    GOEMOTIONS_LABELS.remove("neutral")

# === INFERENZA ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = CFG.model.name

logger.info("Caricamento tokenizer e config...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

logger.info("Inizializzazione modello CustomMultiLabelModel...")
model = CustomMultiLabelModel(
    model_name=model_name,
    num_labels=CFG.model.num_labels,
    pos_weight=torch.ones(CFG.model.num_labels)  # placeholder per l'inferenza
)

state_dict_path = os.path.join(CFG.model.dir, "pytorch_model.bin")
logger.info(f"Caricamento pesi del modello da: {state_dict_path}")
state_dict = torch.load(state_dict_path, map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if 'loss_fn' not in k}

model.load_state_dict(filtered_state_dict, strict=False)
model.to(device)
model.eval()

# === Thresholds dinamici ===
thresholds_path = CFG.paths.thresholds
logger.info(f"Caricamento thresholds da: {thresholds_path}")
with open(thresholds_path, "r") as f:
    THRESHOLDS = json.load(f)


def infer_sentiment(emotions: Dict[str, float]) -> str:
    scores = defaultdict(float)
    for emotion, score in emotions.items():
        if emotion in POSITIVE:
            scores["positive"] += score
        elif emotion in NEGATIVE:
            scores["negative"] += score
        else:
            scores["neutral"] += score
    return max(scores.items(), key=lambda x: x[1])[0] if scores else "neutral"


def predict_emotions(text: str) -> Dict[str, Dict[str, float] | str]:
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=CFG.model.max_length
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    result = {}
    for i, (label, prob) in enumerate(zip(GOEMOTIONS_LABELS, probs)):
        threshold = THRESHOLDS.get(str(i), 0.5)
        if prob >= threshold:
            result[label] = round(float(prob), 3)

    return {
        "emotions": result,
        "sentiment": infer_sentiment(result)
    }


def main():
    while True:
        text = input("Inserisci una frase per l'analisi delle emozioni (exit per uscire): ")
        if text.lower() == "exit":
            break
        result = predict_emotions(text)
        print("Emozioni rilevate:")
        for key, value in result.items():
            print(f"  {key}: {value}")


# Esegui come: python -m app.model.inference.predict
if __name__ == "__main__":
    main()
