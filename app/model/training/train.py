import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Importare numpy prima di tutto per evitare incompatibilità MKL
import numpy as np  # noqa: F401
import pandas as pd

# Per filtrare warning di sklearn su metriche ill-defined
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Logging
import logging
import torch
import mlflow
mlflow.autolog()
import mlflow.pytorch
from transformers import (
    AutoTokenizer,
    Trainer,
    EarlyStoppingCallback,
    ProgressCallback,
    set_seed
)

from utils.mlflow_utils import start_or_continue_run
from config.loader import CFG
from utils.preprocess import load_and_preprocess_dataset
from model.training.train_utils import (
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

    # Setup MLflow
    mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
    mlflow.set_experiment(CFG.mlflow.experiment_name)

    # Seed
    set_seed(CFG.model.seed)

    # Dataset
    logger.info("Caricamento del dataset GoEmotions (config: 'simplified')...")
    dataset = load_and_preprocess_dataset()

    # Modello
    logger.info(f"Costruzione modello {CFG.model.name} con {CFG.model.num_labels} classi...")
    model = CustomMultiLabelModel(
        model_name=CFG.model.name,
        num_labels=CFG.model.num_labels,
        pos_weight=None
    )

    # Tokenizer
    logger.info("Inizializzazione del tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.name)

    # Argomenti di training
    logger.info("Configurazione TrainingArguments...")
    training_args = get_training_args(cfg = None)

    # Trainer
    logger.info("Inizializzazione Trainer...")
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

    # Avvio run MLflow (nested se già attiva)
    with start_or_continue_run(run_name="train"):
        try:
            logger.info("Logging dei parametri in MLflow...")
            mlflow.log_param("model_name", CFG.model.name)
            mlflow.log_param("num_labels", CFG.model.num_labels)
            for k, v in vars(CFG.training).items():
                mlflow.log_param(k, v)

            # Training
            logger.info("Avvio training...")
            trainer.train()

            # Valutazione
            logger.info("Valutazione del modello...")
            metrics = trainer.evaluate()
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Salvataggio locale
            logger.info("Salvataggio su disco (tokenizer e config)...")
            save_model(trainer, tokenizer, model)

            # Logging del modello in MLflow come artifact grezzo
            logger.info("Logging del modello su MLflow...")

            # Salva l'intera directory di output del training (contiene model + tokenizer)
            mlflow.log_artifacts(training_args.output_dir, artifact_path="model")

            logger.info("✅ Modello loggato con successo in MLflow.")

        except Exception as e:
            logger.exception("Errore durante la run MLflow:")
            raise

if __name__ == "__main__":
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    train()
