import os
from typing import List

class CFG:
    # === MODELLO ===
    model_name = "microsoft/deberta-v3-base"

    # === LABELS === (neutral esclusa)
    label_list: List[str] = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise"
    ]  # 27 etichette reali, senza 'neutral'

    num_labels = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # === TOKENIZZAZIONE ===
    max_length = 128

    # === HYPERPARAMETRI ===
    learning_rate = 2e-5
    batch_size = 16
    num_epochs = 5
    weight_decay = 0.01

    # === RANDOM SEED ===
    seed = 42

    # === DEVICE ===
    device = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

    # === PATH (relativi alla root /app) ===
    base_path = "/app"
    data_dir = os.path.join(base_path, "model", "data")
    model_dir = os.path.join(base_path, "model", "models")
    logs_dir = os.path.join(base_path, "model", "logs")
    outputs_dir = os.path.join(base_path, "model", "outputs")

    # === NOMI FILE STANDARD ===
    tokenizer_file = os.path.join(model_dir, "tokenizer")
    model_file = os.path.join(model_dir, "deberta_model.pt")
    metrics_file = os.path.join(outputs_dir, "metrics.json")
    thresholds_path = os.path.join(logs_dir, "thresholds.json")
