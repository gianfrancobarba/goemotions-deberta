import os
import torch
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow

from transformers import AutoTokenizer, AutoConfig

from config.loader import CFG
from utils.preprocess import load_and_preprocess_dataset
from model.training.train_utils import CustomMultiLabelModel
from utils.mlflow_utils import start_or_continue_run

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate():
    logger.info("🚀 Avvio valutazione finale sul test set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricamento tokenizer e config
    tokenizer = AutoTokenizer.from_pretrained(CFG.paths.model_dir)
    config = AutoConfig.from_pretrained(CFG.paths.model_dir)

    # Costruzione modello e caricamento pesi
    model = CustomMultiLabelModel(
        model_name=CFG.model.name,
        num_labels=CFG.model.num_labels,
        pos_weight=torch.ones(CFG.model.num_labels)
    )
    model_path = CFG.paths.model_file
    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'loss_fn' not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()

    # Caricamento test set preprocessato
    dataset = load_and_preprocess_dataset()
    test_dataset = dataset["test"]
    batch_size = CFG.training.per_device_eval_batch_size

    all_preds, all_probs, all_labels = [], [], []

    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset.select(range(i, min(i + batch_size, len(test_dataset))))
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs["logits"])
            preds = (probs > 0.5).long()  # threshold fisso

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    # Calcolo metriche
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision_micro = precision_score(y_true, y_pred, average="micro")
    recall_micro = recall_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    metrics = {
        "f1_micro": round(f1_micro, 4),
        "precision_micro": round(precision_micro, 4),
        "recall_micro": round(recall_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
    }

    # Logging MLflow come run figlia
    mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
    mlflow.set_experiment(CFG.mlflow.experiment_name)

    with start_or_continue_run(run_name="evaluate_test_final"):
        mlflow.log_metrics(metrics)

        os.makedirs(CFG.paths.model_logs, exist_ok=True)

        # Salvataggio metriche locali
        metrics_path = CFG.paths.metrics_file
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)

        # Salvataggio previsioni raw
        preds_df = pd.DataFrame({
            "true_labels": y_true.tolist(),
            "predicted_labels": y_pred.tolist(),
            "probabilities": y_prob.tolist(),
        })
        preds_path = os.path.join(CFG.paths.model_logs, "test_predictions.json")
        preds_df.to_json(preds_path, orient="records", indent=2)
        mlflow.log_artifact(preds_path)

        logger.info("✅ Valutazione completata. Metriche e artefatti loggati con successo.")


if __name__ == "__main__":
    evaluate()
