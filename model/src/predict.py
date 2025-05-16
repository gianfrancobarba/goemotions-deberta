import os
import torch
import logging
from transformers import AutoTokenizer, AutoConfig
from model.src.train_utils import CustomMultiLabelModel
from config.config import CFG

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Etichette (puoi cambiarle se usi una lista personalizzata)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Rimuovi 'neutral' se non è stata usata durante il training
if CFG.num_labels == 27 and len(GOEMOTIONS_LABELS) == 28:
    GOEMOTIONS_LABELS.remove("neutral")

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricamento tokenizer e modello
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_dir)
    config = AutoConfig.from_pretrained(CFG.model_dir)

    model = CustomMultiLabelModel(
        model_name=CFG.model_name,
        num_labels=CFG.num_labels,
        pos_weight=torch.ones(CFG.num_labels)  # non usato in eval
    )
    model.load_state_dict(torch.load(os.path.join(CFG.model_dir, "deberta_model.pt"), map_location=device))
    model.to(device)
    model.eval()

    logger.info("✳️ Inserisci una frase per la predizione (digita 'exit' per terminare)")

    while True:
        text = input("📝 > ").strip()
        if text.lower() == "exit":
            logger.info("🛑 Uscita.")
            break
        if not text:
            continue

        # Preprocessing
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=CFG.max_length)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Stampa emozioni predette con probabilità
        print("\n🎯 Emozioni rilevate:")
        for i, p in enumerate(probs):
            if p >= 0.01:  # soglia di visualizzazione
                print(f"  - {GOEMOTIONS_LABELS[i]:<15}: {p:.2f}")
        print()


if __name__ == "__main__":
    predict()
