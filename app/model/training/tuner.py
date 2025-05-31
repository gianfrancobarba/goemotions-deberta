# model/src/tuner.py

import os
import optuna
import logging
import pandas as pd
from transformers import set_seed

from config.config import CFG
from model.src.train_utils import train_and_evaluate

# === Setup logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === Obiettivo per Optuna ===
def objective(trial):
    # Suggerisci iperparametri
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    num_epochs = trial.suggest_int("num_epochs", 2, 5)

    # Imposta temporaneamente la configurazione
    CFG.learning_rate = learning_rate
    CFG.batch_size = batch_size
    CFG.weight_decay = weight_decay
    CFG.num_epochs = num_epochs

    # Assicura riproducibilità
    set_seed(CFG.seed)

    logger.info(f"🔍 Trial {trial.number} - lr: {learning_rate}, bs: {batch_size}, wd: {weight_decay}, ep: {num_epochs}")

    # Allenamento + valutazione (ritorna F1 score su validation)
    f1 = train_and_evaluate()
    trial.set_user_attr("f1_score", f1)

    return f1


def run_tuning(n_trials: int = 5, study_name: str = "deberta_tuning"):
    logger.info("🚀 Avvio della ricerca Optuna...")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info("✅ Tuning completato.")
    logger.info(f"🥇 Miglior trial: {study.best_trial.number}")
    logger.info(f"🎯 Miglior F1 score: {study.best_value}")
    logger.info(f"📊 Parametri: {study.best_params}")

    # Salvataggio dei risultati
    os.makedirs(CFG.outputs_dir, exist_ok=True)

    # CSV di tutti i trials
    df = study.trials_dataframe()
    df.to_csv(os.path.join(CFG.outputs_dir, "tuning_trials.csv"), index=False)

    # JSON dei migliori parametri
    best_cfg_path = os.path.join(CFG.outputs_dir, "best_hyperparams.json")
    with open(best_cfg_path, "w") as f:
        import json
        json.dump(study.best_params, f, indent=4)

    logger.info(f"💾 Risultati salvati in {CFG.outputs_dir}")


if __name__ == "__main__":
    run_tuning()
