# model/src/evaluate.py

import os
import torch
import json
import logging
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoConfig

from config.config import CFG
from model.src.preprocess import load_and_preprocess_dataset
from model.src.train_utils import CustomMultiLabelModel

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate():
    logger.info("🔍 Avvio della valutazione sul test set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_dir)
    config = AutoConfig.from_pretrained(CFG.model_dir)

    model = CustomMultiLabelModel(
        model_name=CFG.model_name,
        num_labels=CFG.num_labels,
        pos_weight=torch.ones(CFG.num_labels)
    )
    model.load_state_dict(torch.load(os.path.join(CFG.model_dir, "deberta_model.pt"), map_location=device))
    model.to(device)
    model.eval()

    dataset = load_and_preprocess_dataset()
    test_dataset = dataset["test"]

    all_preds, all_labels = [], []
    batch_size = CFG.batch_size

    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset.select(range(i, min(i + batch_size, len(test_dataset))))
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs["logits"])
            preds = (probs > 0.5).long()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    # Micro
    f1_micro = f1_score(y_true, y_pred, average="micro")
    precision_micro = precision_score(y_true, y_pred, average="micro")
    recall_micro = recall_score(y_true, y_pred, average="micro")

    # Macro
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    logger.info(f"✅ Valutazione completata:")
    logger.info(f"   F1-score (micro):   {f1_micro:.4f}")
    logger.info(f"   Precision (micro):  {precision_micro:.4f}")
    logger.info(f"   Recall (micro):     {recall_micro:.4f}")
    logger.info(f"   F1-score (macro):   {f1_macro:.4f}")
    logger.info(f"   Precision (macro):  {precision_macro:.4f}")
    logger.info(f"   Recall (macro):     {recall_macro:.4f}")

    # 🔽 Salva le metriche su file
    os.makedirs(CFG.logs_dir, exist_ok=True)
    output_path = os.path.join(CFG.logs_dir, "test_metrics.json")
    with open(output_path, "w") as f:
        json.dump({
            "f1_micro": round(f1_micro, 4),
            "precision_micro": round(precision_micro, 4),
            "recall_micro": round(recall_micro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
        }, f, indent=4)

    logger.info(f"📁 Metriche test salvate in: {output_path}")


if __name__ == "__main__":
    evaluate()
