# model/src/train_utils.py

import os
import json
import torch
import numpy as np
from typing import Dict
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from config.config import CFG
from model.src.preprocess import load_and_preprocess_dataset


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
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        loss = self.loss_fn(logits, labels.float()) if labels is not None else None
        return {"loss": loss, "logits": logits}


def compute_pos_weights(dataset) -> torch.Tensor:
    """
    Calcola i pesi per la BCEWithLogitsLoss per tenere conto dello sbilanciamento.
    """
    labels = np.vstack([example["labels"] for example in dataset])
    label_counts = labels.sum(axis=0)
    total_samples = labels.shape[0]
    pos_weight = (total_samples - label_counts + 1) / (label_counts + 1)
    return torch.clamp(torch.tensor(pos_weight, dtype=torch.float32), min=0.5, max=3.0)


def compute_metrics(eval_preds):
    """
    Calcola F1, precision e recall micro-aggregati.
    """
    logits, labels = eval_preds
    predictions = (logits > 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, predictions, average="micro"),
        "precision_micro": precision_score(labels, predictions, average="micro"),
        "recall_micro": recall_score(labels, predictions, average="micro"),
    }


def save_model(trainer, tokenizer, model):
    """
    Salva il modello in formato compatibile HuggingFace.
    Include: modello, tokenizer, configurazione, pesi (pytorch_model.bin), metriche.
    """
    output_dir = trainer.args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 🔧 Patch per assicurarsi che il config sia salvato correttamente
    model.config.num_labels = CFG.num_labels
    model.config.problem_type = "multi_label_classification"

    # 🔧 Patch per assicurarsi che il config sia salvato correttamente
    model.config.num_labels = CFG.num_labels
    model.config.problem_type = "multi_label_classification"
    model.config.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Salva metriche (opzionale)
    metrics = trainer.evaluate()
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Modello salvato in {output_dir}")
    print(f"Metriche salvate in {metrics_path}")

def get_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        learning_rate=1e-5,
        warmup_steps=500,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        dataloader_num_workers=4,
        report_to="none"
    )


def train_and_evaluate() -> float:
    """
    Esegue il training con i parametri attuali in CFG e ritorna l'F1 micro sul validation set.
    Utile per Optuna.
    """
    dataset = load_and_preprocess_dataset()
    pos_weight = compute_pos_weights(dataset["train"])

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = CustomMultiLabelModel(CFG.model_name, CFG.num_labels, pos_weight)

    # Args per tuning
    args = get_training_args(CFG.model_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics.get("eval_f1", 0.0)
