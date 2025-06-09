import os
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoConfig, default_data_collator

import mlflow
mlflow.autolog()
from config.loader import CFG
from utils.mlflow_utils import start_or_continue_run
from utils.preprocess import load_and_preprocess_dataset
from model.training.train_utils import CustomMultiLabelModel

# === Setup dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Calcolo delle soglie ottimali ===
def compute_optimal_thresholds(model, dataset):
    batch_size = CFG.thresholding.batch_size
    t_cfg = CFG.thresholding.threshold_range
    thresholds = np.arange(t_cfg.start, t_cfg.stop + t_cfg.step, t_cfg.step)

    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
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

        best_thresholds[str(i)] = round(best_t, 2)
        rows.append({
            "label_index": i,
            "best_threshold": round(best_t, 2),
            "precision": round(best_metrics[0], 4),
            "recall": round(best_metrics[1], 4),
            "f1_score": round(best_metrics[2], 4),
        })

    metrics_df = pd.DataFrame(rows)
    return best_thresholds, metrics_df

# === Funzione principale ===
def tune_thresholds():
    print("🎯 Inizio tuning delle soglie...")

    model_dir = CFG.model.dir or CFG.model.name
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = CustomMultiLabelModel(
        model_name=CFG.model.name,
        num_labels=CFG.model.num_labels,
        pos_weight=None
    )
    model_path = os.path.join(CFG.model.dir, CFG.model.model_file)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    dataset = load_and_preprocess_dataset(remove_neutral=(CFG.model.num_labels == 27))
    dev_dataset = dataset["validation"]

    print("🧪 Calcolo thresholds ottimali per ciascuna etichetta...")
    best_thresholds, metrics_df = compute_optimal_thresholds(model, dev_dataset)

    os.makedirs(os.path.dirname(CFG.paths.thresholds), exist_ok=True)

    with open(CFG.paths.thresholds, "w") as f:
        json.dump(best_thresholds, f, indent=4)
    print(f"✅ Thresholds salvati in {CFG.paths.thresholds}")

    metrics_csv_path = CFG.paths.thresholds.replace(".json", "_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"📊 Metriche salvate in {metrics_csv_path}")

    # Log su MLflow
    mlflow.log_artifact(CFG.paths.thresholds)
    mlflow.log_artifact(metrics_csv_path)
    mlflow.log_param("thresholding_method", "per-label search")
    tr = CFG.thresholding.threshold_range
    mlflow.log_param("threshold_range", f"{tr.start}-{tr.stop}-{tr.step}")

    for _, row in metrics_df.iterrows():
        mlflow.log_metric(f"f1_label_{int(row['label_index'])}", row["f1_score"])

# === Entry point ===
if __name__ == "__main__":
    mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
    mlflow.set_experiment(CFG.mlflow.experiment_name)

    with start_or_continue_run(run_name="tune_thresholds"):
        tune_thresholds()
