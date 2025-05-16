import os
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoConfig, default_data_collator

from config.config import CFG
from model.src.preprocess import load_and_preprocess_dataset
from model.src.train_utils import CustomMultiLabelModel

def compute_optimal_thresholds(model, dataset, batch_size=32, thresholds=np.arange(0.1, 0.91, 0.05)):
    """
    Calcola i migliori threshold per ciascuna etichetta in base all'F1 score.
    Restituisce:
      - best_thresholds: dict {indice_label: threshold}
      - metrics_df: dataframe con precision, recall, f1 per ogni label
    """
    model.eval()
    model.to(CFG.device)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)
            labels = batch["labels"].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu().numpy()

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    num_labels = all_labels.shape[1]

    best_thresholds = {}
    rows = []

    for i in range(num_labels):
        best_f1 = 0.0
        best_t = 0.5
        best_metrics = (0.0, 0.0, 0.0)

        for t in thresholds:
            preds = (all_logits[:, i] > t).astype(int)
            precision = precision_score(all_labels[:, i], preds, zero_division=0)
            recall = recall_score(all_labels[:, i], preds, zero_division=0)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_metrics = (precision, recall, f1)

        best_thresholds[i] = round(best_t, 2)
        rows.append({
            "label_index": i,
            "best_threshold": round(best_t, 2),
            "precision": round(best_metrics[0], 4),
            "recall": round(best_metrics[1], 4),
            "f1_score": round(best_metrics[2], 4),
        })

    metrics_df = pd.DataFrame(rows)
    return best_thresholds, metrics_df


def main():
    print("Caricamento modello e dataset di validazione...")

    # Config e tokenizer
    config = AutoConfig.from_pretrained(CFG.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_dir)

    # Modello
    model = CustomMultiLabelModel(
        model_name=CFG.model_name,
        num_labels=CFG.num_labels,
        pos_weight=None
    )
    model.load_state_dict(torch.load(os.path.join(CFG.model_dir, "pytorch_model.bin")))

    # Dataset
    dataset = load_and_preprocess_dataset(remove_neutral=True)
    dev_dataset = dataset["validation"]

    print("Calcolo thresholds ottimali per ciascuna etichetta...")
    best_thresholds, metrics_df = compute_optimal_thresholds(model, dev_dataset)

    # Salvataggio risultati
    os.makedirs(os.path.dirname(CFG.thresholds_path), exist_ok=True)

    with open(CFG.thresholds_path, "w") as f:
        json.dump(best_thresholds, f, indent=4)
    print(f"Thresholds salvati in {CFG.thresholds_path}")

    metrics_csv_path = CFG.thresholds_path.replace(".json", "_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metriche salvate in {metrics_csv_path}")


if __name__ == "__main__":
    main()
