import os
import json
import optuna
import logging
import pandas as pd
from copy import deepcopy
from transformers import set_seed
import mlflow

from config.loader import CFG
from model.training.train_utils import train_and_evaluate

# === Setup logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === Optuna Objective ===
def objective(trial):
    # Clona l'intera configurazione per non alterare CFG globale
    trial_cfg = deepcopy(CFG)

    space = CFG.tuning.search_space

    # Cast espliciti e sicuri dei bounds
    lr_min = float(space.learning_rate.min)
    lr_max = float(space.learning_rate.max)
    wd_min = float(space.weight_decay.min)
    wd_max = float(space.weight_decay.max)
    ep_min = int(space.num_epochs.min)
    ep_max = int(space.num_epochs.max)
    batch_choices = list(space.batch_size)

    # Sampling dallo spazio di ricerca
    trial_cfg.training.learning_rate = trial.suggest_float(
        "learning_rate", lr_min, lr_max, log=True
    )
    trial_cfg.training.weight_decay = trial.suggest_float(
        "weight_decay", wd_min, wd_max
    )
    trial_cfg.training.num_train_epochs = trial.suggest_int(
        "num_epochs", ep_min, ep_max
    )
    trial_cfg.training.per_device_train_batch_size = trial.suggest_categorical(
        "batch_size", batch_choices
    )

    # Imposta il seed per riproducibilità
    set_seed(int(trial_cfg.model.seed))

    with mlflow.start_run(nested=True, run_name=f"optuna_trial_{trial.number}"):
        # Log dei parametri del trial
        mlflow.log_params({
            "learning_rate": trial_cfg.training.learning_rate,
            "batch_size": trial_cfg.training.per_device_train_batch_size,
            "weight_decay": trial_cfg.training.weight_decay,
            "num_train_epochs": trial_cfg.training.num_train_epochs,
        })

        # Addestramento + valutazione
        f1_micro = train_and_evaluate(cfg=trial_cfg)

        # Log della metrica obiettivo
        mlflow.log_metric("f1_micro", f1_micro)
        trial.set_user_attr("f1_score", f1_micro)

    return f1_micro


# === Run tuning ===
def run_tuning():
    logger.info("🚀 Avvio tuning Optuna...")

    mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
    mlflow.set_experiment(CFG.tuning.experiment_name)

    study = optuna.create_study(
        study_name=CFG.tuning.study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=CFG.tuning.n_trials)

    logger.info(f"✅ Tuning completato. Best trial: {study.best_trial.number}")
    logger.info(f"🎯 Best f1_micro: {study.best_value}")
    logger.info(f"📊 Best params: {study.best_params}")

    os.makedirs(CFG.paths.outputs, exist_ok=True)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(CFG.paths.outputs, "tuning_trials.csv"), index=False)

    with open(os.path.join(CFG.paths.outputs, "best_hyperparams.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    logger.info(f"📁 Risultati salvati in {CFG.paths.outputs}")


if __name__ == "__main__":
    run_tuning()
