# model/src/model_utils.py

import os
import logging
import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from config.config import CFG

# === Logging ===
log_path = os.path.join(CFG.logs_dir, "model_utils.log")
os.makedirs(CFG.logs_dir, exist_ok=True)
logging.basicConfig(
    filename=log_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class CustomMultiLabelModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, pos_weight: torch.Tensor):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # [CLS] token
        loss = self.loss_fn(logits, labels.float()) if labels is not None else None
        return {"loss": loss, "logits": logits}


def build_model(model_name: str, num_labels: int, pos_weight: torch.Tensor):
    """
    Inizializza il modello personalizzato e il tokenizer.

    Returns:
        model (nn.Module), tokenizer
    """
    logger.info(f"🧠 Costruzione modello {model_name} con {num_labels} classi.")
    model = CustomMultiLabelModel(model_name, num_labels, pos_weight)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def compute_pos_weights(dataset) -> torch.Tensor:
    """
    Calcola i pesi per BCEWithLogitsLoss in base alla distribuzione multilabel.

    Args:
        dataset (datasets.Dataset): dataset di training

    Returns:
        torch.Tensor: vettore dei pesi (dim = num_labels)
    """
    logger.info("📊 Calcolo pos_weight per BCEWithLogitsLoss...")
    labels = np.vstack([example["labels"] for example in dataset])
    label_counts = labels.sum(axis=0)
    total_samples = labels.shape[0]
    pos_weight = (total_samples - label_counts + 1) / (label_counts + 1)
    pos_weight = torch.clamp(torch.tensor(pos_weight, dtype=torch.float32), min=0.5, max=3.0)
    logger.info(f"✅ pos_weight shape: {pos_weight.shape}")
    return pos_weight


def compute_metrics(eval_preds):
    """
    Funzione per calcolare le metriche multilabel.

    Args:
        eval_preds: predizioni (logits), labels reali

    Returns:
        dict con f1, precision, recall (media micro)
    """
    logger.info("📈 Calcolo delle metriche di valutazione...")
    logits, labels = eval_preds
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    return {
        "f1": f1_score(labels, preds, average="micro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0)
    }
