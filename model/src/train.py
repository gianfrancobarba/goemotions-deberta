# model/src/train.py

import os
import json
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
    get_training_args
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train():
    logger.info("Inizio training...")

    # Imposta il seed per la riproducibilità
    set_seed(CFG.seed)

    logger.info("Caricamento del dataset GoEmotions (config: 'simplified')...")
    dataset = load_and_preprocess_dataset()

    # Calcolo pos_weight per la BCEWithLogitsLoss
    # Commenta questa parte se non vuoi usare una loss pesata
    # logger.info("Calcolo pos_weight per BCEWithLogitsLoss...")
    # pos_weight = compute_pos_weights(dataset["train"])
    # logger.info(f"pos_weight shape: {pos_weight.shape}")

    logger.info(f"Costruzione modello {CFG.model_name} con {CFG.num_labels} classi...")
    model = CustomMultiLabelModel(
        model_name=CFG.model_name,
        num_labels=CFG.num_labels,
        pos_weight=None  #disattivo il bilanciamento
    )

    logger.info("Inizializzazione del tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    logger.info("Configurazione TrainingArguments...")
    training_args = get_training_args(CFG.model_dir)

    logger.info("Inizializzazione Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    logger.info("Avvio training...")
    trainer.train()

    logger.info("Salvataggio modello e configurazione...")
    save_model(trainer, tokenizer, model)


if __name__ == "__main__":
    train()
