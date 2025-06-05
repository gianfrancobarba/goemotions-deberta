import os
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoConfig, default_data_collator

import mlflow
from config.loader import CFG
from utils.preprocess import load_and_preprocess_dataset
from utils.mlflow_utils import start_or_continue_run
from model.training.train_utils import CustomMultiLabelModel

# === Setup dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Applica soglie ai logits ===
def apply_thresholds(logits: np.ndarray, thresholds_dict: dict) -> np.ndarray:
    thresholds = np.array([thresholds_dict[str(i)] for i in range(logits.shape[1])])
    return (logits > thresholds).astype(int)

# === Funzione principale di valutazione ===
def evaluate_with_thresholds():
    print("🔍 Valutazione del modello con thresholds ottimali...")

    # Carica soglie ottimali da file
    with open(CFG.paths.thresholds, "r") as f:
        best_thresholds = json.load(f)

    # Carica modello e tokenizer
    config = AutoConfig.from_pretrained(CFG.model.name)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.name)
    model = CustomMultiLabelModel(
        model_name=CFG.model.name,
        num_labels=CFG.model.num_labels,
        pos_weight=None
    )
    model_path = os.path.join(CFG.model.dir, CFG.model.model_file)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dataset
    dataset = load_and_preprocess_dataset(remove_neutral=(CFG.model.num_labels == 27))
    eval_dataset = dataset["validation"]
    dataloader = DataLoader(
        eval_dataset,
        batch_size=CFG.thresholding.batch_size,
        collate_fn=default_data_collator
    )

    # Inference
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = apply_thresholds(all_logits, best_thresholds)

    # Metriche globali
    f1_micro = f1_score(all_labels, preds, average="micro")
    f1_macro = f1_score(all_labels, preds, average="macro")
    precision = precision_score(all_labels, preds, average="micro")
    recall = recall_score(all_labels, preds, average="micro")

    print("=== Risultati ===")
    print(f"F1 (micro):  {f1_micro:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")

    # Salvataggio risultati
    os.makedirs(CFG.paths.model_logs, exist_ok=True)
    output_json = os.path.join(CFG.paths.model_logs, "eval_with_thresholds.json")
    output_csv = os.path.join(CFG.paths.model_logs, "eval_with_thresholds_metrics.csv")

    results_dict = {
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_micro": round(precision, 4),
        "recall_micro": round(recall, 4),
    }
    with open(output_json, "w") as f:
        json.dump(results_dict, f, indent=4)

    label_metrics = []
    for i in range(CFG.model.num_labels):
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

    pd.DataFrame(label_metrics).to_csv(output_csv, index=False)
    print(f"📁 Metriche salvate in: {output_json} e {output_csv}")

    # Logging MLflow
    mlflow.log_param("thresholding_method", "apply_precomputed")
    mlflow.log_artifact(output_json)
    mlflow.log_artifact(output_csv)
    mlflow.log_metric("f1_micro", f1_micro)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("precision_micro", precision)
    mlflow.log_metric("recall_micro", recall)

    for row in label_metrics:
        mlflow.log_metric(f"f1_label_{int(row['label_index'])}", row["f1_score"])

# === Entry point ===
if __name__ == "__main__":
    mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
    mlflow.set_experiment(CFG.mlflow.experiment_name)

    with mlflow.start_run(run_name="evaluate_with_thresholds"):
        evaluate_with_thresholds()

