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

from app.config.loader import CFG
from app.utils.preprocess import load_and_preprocess_dataset


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


def get_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=CFG.training.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.training.per_device_eval_batch_size,
        num_train_epochs=CFG.training.num_train_epochs,
        learning_rate=float(CFG.training.learning_rate),
        warmup_steps=CFG.training.warmup_steps,
        weight_decay=CFG.training.weight_decay,
        lr_scheduler_type=CFG.training.lr_scheduler_type,
        evaluation_strategy=CFG.training.evaluation_strategy,
        save_strategy=CFG.training.save_strategy,
        save_total_limit=CFG.training.save_total_limit,
        logging_steps=CFG.training.logging_steps,
        fp16=CFG.training.fp16,
        load_best_model_at_end=CFG.training.load_best_model_at_end,
        metric_for_best_model=CFG.training.metric_for_best_model,
        greater_is_better=CFG.training.greater_is_better,
        dataloader_num_workers=CFG.training.dataloader_num_workers,
        report_to=CFG.training.report_to
    )


def train_and_evaluate() -> float:
    dataset = load_and_preprocess_dataset()
    pos_weight = compute_pos_weights(dataset["train"])

    tokenizer = AutoTokenizer.from_pretrained(CFG.model.name)
    model = CustomMultiLabelModel(CFG.model.name, CFG.model.num_labels, pos_weight)

    training_args = get_training_args(CFG.model.dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics.get("eval_f1_micro", 0.0)
