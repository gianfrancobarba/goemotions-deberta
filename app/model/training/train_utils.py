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
    ProgressCallback,
)

from config.loader import CFG
from utils.preprocess import load_and_preprocess_dataset


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
    labels = np.vstack([example["labels"] for example in dataset])
    label_counts = labels.sum(axis=0)
    total_samples = labels.shape[0]
    pos_weight = (total_samples - label_counts + 1) / (label_counts + 1)
    return torch.clamp(torch.tensor(pos_weight, dtype=torch.float32), min=0.5, max=3.0)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = (logits > CFG.thresholding.default_threshold).astype(int)
    return {
        "f1_micro": f1_score(labels, predictions, average="micro"),
        "precision_micro": precision_score(labels, predictions, average="micro"),
        "recall_micro": recall_score(labels, predictions, average="micro"),
    }


def save_model(trainer, tokenizer, model):
    output_dir = trainer.args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Salva configurazione aggiornata
    model.config.num_labels = CFG.model.num_labels
    model.config.problem_type = "multi_label_classification"
    model.config.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Salva metriche
    metrics = trainer.evaluate()
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Modello salvato in {output_dir}")
    print(f"Metriche salvate in {metrics_path}")


from transformers import TrainingArguments

def get_training_args(cfg) -> TrainingArguments:
    cfg = cfg or CFG
    return TrainingArguments(
        output_dir=cfg.model.dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=float(cfg.training.learning_rate),
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        logging_steps=cfg.training.logging_steps,
        fp16=cfg.training.fp16,
        max_steps=cfg.training.max_steps,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        report_to=cfg.training.report_to,
    )


def train_and_evaluate(cfg) -> float:
    cfg = cfg or CFG
    dataset = load_and_preprocess_dataset()
    pos_weight = compute_pos_weights(dataset["train"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = CustomMultiLabelModel(cfg.model.name, cfg.model.num_labels, pos_weight)

    training_args = get_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
            ProgressCallback()
        ]
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics.get("eval_f1_micro", 0.0)

