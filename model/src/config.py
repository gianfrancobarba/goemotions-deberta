# model/src/config.py

"""
Configurazione centralizzata per il progetto GoEmotions + DeBERTa.

Contiene:
- Nomi di modelli e tokenizer
- Parametri per la tokenizzazione e preprocessing
- Mapping delle label (nome ↔ indice)
- Hyperparametri di training (se noti)
"""

from typing import List

# Nome del modello base da HuggingFace (tokenizer e weights)
MODEL_NAME = "microsoft/deberta-v3-base"

# Numero totale di etichette (GoEmotions = 28)
NUM_LABELS = 28

# Lunghezza massima dei token (padding + truncation)
MAX_LENGTH = 128

# Etichette in ordine, secondo HuggingFace
LABEL_LIST: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Mapping inverso: indice → etichetta
IDX2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Mapping diretto: etichetta → indice
LABEL2IDX = {label: i for i, label in enumerate(LABEL_LIST)}

# [Opzionale] Hyperparametri iniziali (da sovrascrivere in fase di tuning)
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01

# [Opzionale] Device di default (può essere override)
DEVICE = "cuda"