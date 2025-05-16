# model/src/train.py

import os
import json
import logging
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from config.config import CFG
from model.src.preprocess import load_and_preprocess_dataset
from model.src.train_utils import (
    compute_pos_weights,
    compute_metrics,
    CustomMultiLabelModel,
    save_model,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train():
    logger.info("🚀 Inizio training...")

    # Imposta il seed per la riproducibilità
    set_seed(CFG.seed)

    logger.info("🔄 Caricamento del dataset GoEmotions (config: 'simplified')...")
    dataset = load_and_preprocess_dataset()
    logger.info("✅ Preprocessing completato.")

    logger.info("📊 Calcolo pos_weight per BCEWithLogitsLoss...")
    pos_weight = compute_pos_weights(dataset["train"])
    logger.info(f"✅ pos_weight shape: {pos_weight.shape}")

    logger.info(f"🧠 Costruzione modello {CFG.model_name} con {CFG.num_labels} classi.")
    model = CustomMultiLabelModel(
        model_name=CFG.model_name,
        num_labels=CFG.num_labels,
        pos_weight=pos_weight
    )

    logger.info("🔤 Inizializzazione del tokenizer DeBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    logger.info("⚙️  Configurazione TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=CFG.model_dir,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        num_train_epochs=CFG.num_epochs,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_steps=0,
        logging_dir=CFG.logs_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=2,
        report_to="none"
    )

    logger.info("🧪 Inizializzazione Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("🏁 Avvio training...")
    trainer.train()

    logger.info("📈 Calcolo delle metriche di valutazione finale...")
    metrics = trainer.evaluate()

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Salva le metriche finali su file JSON
    os.makedirs(CFG.logs_dir, exist_ok=True)
    metrics_path = os.path.join(CFG.logs_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"✅ Metriche salvate in {metrics_path}")

    logger.info("💾 Salvataggio modello e configurazione finale...")
    save_model(trainer, tokenizer, model)


if __name__ == "__main__":
    train()
