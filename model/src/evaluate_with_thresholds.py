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


def apply_thresholds(logits, thresholds_dict):
    thresholds = np.array([thresholds_dict[str(i)] for i in range(logits.shape[1])])
    return (logits > thresholds).astype(int)


def main():
    print("Valutazione del modello con thresholds ottimali...")

    # === Caricamento thresholds ===
    with open(CFG.thresholds_path, "r") as f:
        best_thresholds = json.load(f)

    # === Caricamento modello ===
    config = AutoConfig.from_pretrained(CFG.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_dir)
    model = CustomMultiLabelModel(CFG.model_name, CFG.num_labels, pos_weight=None)
    model.load_state_dict(torch.load(os.path.join(CFG.model_dir, "pytorch_model.bin")))
    model.to(CFG.device)
    model.eval()

    # === Dataset ===
    dataset = load_and_preprocess_dataset(remove_neutral=True)
    eval_dataset = dataset["validation"]  # oppure "test"
    dataloader = DataLoader(eval_dataset, batch_size=32, collate_fn=default_data_collator)

    all_logits = []
    all_labels = []

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

    # === Applica thresholds e calcola metriche ===
    preds = apply_thresholds(all_logits, best_thresholds)

    f1_micro = f1_score(all_labels, preds, average="micro")
    f1_macro = f1_score(all_labels, preds, average="macro")
    precision = precision_score(all_labels, preds, average="micro")
    recall = recall_score(all_labels, preds, average="micro")

    print("=== Risultati ===")
    print(f"F1 (micro):  {f1_micro:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")

    # === Salva risultati in logs/ ===
    os.makedirs("logs", exist_ok=True)

    # JSON summary
    results_dict = {
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_micro": round(precision, 4),
        "recall_micro": round(recall, 4),
    }
    with open("logs/eval_with_thresholds.json", "w") as f:
        json.dump(results_dict, f, indent=4)

    # CSV dettagliato per etichetta
    label_metrics = []
    for i in range(CFG.num_labels):
        p = precision_score(all_labels[:, i], preds[:, i], zero_division=0)
        r = recall_score(all_labels[:, i], preds[:, i], zero_division=0)
        f1 = f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        label_metrics.append({
            "label_index": i,
            "threshold": best_thresholds[str(i)],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
        })

    df = pd.DataFrame(label_metrics)
    df.to_csv("logs/eval_with_thresholds_metrics.csv", index=False)
    print("Metriche salvate in logs/eval_with_thresholds.json e .csv")


if __name__ == "__main__":
    main()
