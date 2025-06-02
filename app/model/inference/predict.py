import os
import torch
import logging
from typing import Dict
from transformers import AutoTokenizer, AutoConfig
from model.training.train_utils import CustomMultiLabelModel
from config.loader import CFG
from collections import defaultdict

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Etichette (caricate da config)
GOEMOTIONS_LABELS = list(CFG.labels.list)
if not CFG.labels.include_neutral and "neutral" in GOEMOTIONS_LABELS:
    GOEMOTIONS_LABELS.remove("neutral")

POSITIVE = set(CFG.sentiment.positive)
NEGATIVE = set(CFG.sentiment.negative)
NEUTRAL = set(CFG.sentiment.neutral)

# Caricamento modello/tokenizer una volta sola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(CFG.model.dir)
config = AutoConfig.from_pretrained(CFG.model.dir)

model = CustomMultiLabelModel(
    model_name=CFG.model.name,
    num_labels=CFG.model.num_labels,
    pos_weight=torch.ones(CFG.model.num_labels)  # usato solo nel training
)
model_path = os.path.join(CFG.model.dir, CFG.model.model_file)
state_dict = torch.load(model_path, map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if 'loss_fn' not in k}
model.load_state_dict(filtered_state_dict, strict=False)
model.to(device)
model.eval()

# Funzione per inferire il sentiment
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

# Funzione chiamabile da FastAPI o CLI
def predict_emotions(text: str) -> Dict[str, float]:
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

    threshold = CFG.thresholding.default_threshold
    result = {
        label: round(float(prob), 3)
        for label, prob in zip(GOEMOTIONS_LABELS, probs)
        if prob >= threshold
    }
    return {
        "emotions": result,
        "sentiment": infer_sentiment(result)
    }

# Esecuzione da terminale
if __name__ == "__main__":
    while True:
        text = input("Inserisci una frase per l'analisi delle emozioni (exit per uscire): ")
        if text.lower() == "exit":
            break
        result = predict_emotions(text)
        print("Emozioni rilevate:")
        for key, value in result.items():
            print(f"  {key}: {value}")
