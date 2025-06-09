# run_experiment.py

import mlflow
mlflow.autolog()
import logging
from config.loader import CFG
from model.training.train import train
from model.thresholds.tune_thresholds import tune_thresholds
from model.training.evaluate_with_thresholds import evaluate_with_thresholds
from model.training.evaluate import evaluate

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imposta MLflow
mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
mlflow.set_experiment(CFG.mlflow.experiment_name)

def main():
    with mlflow.start_run(run_name=CFG.mlflow.run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        logger.info(f"🔁 Run principale avviata: {parent_run_id}")

        # Step 1: Training
        try:
            logger.info("🚀 Step 1: Training...")
            train()
        except Exception as e:
            logger.exception("❌ Errore durante il training:")
            mlflow.set_tag("train_status", "failed")
            raise

        # Step 2: Tuning thresholds
        try:
            logger.info("🎯 Step 2: Tuning thresholds...")
            with mlflow.start_run(run_name="tune_thresholds", nested=True):
                tune_thresholds()
        except Exception as e:
            logger.exception("❌ Errore durante la regolazione delle soglie:")
            mlflow.set_tag("tune_thresholds_status", "failed")
            raise

        # Step 3: Final evaluation with thresholds
        try:
            logger.info("📊 Step 3: Validazione con thresholds finale...")
            with mlflow.start_run(run_name="evaluate_with_thresholds", nested=True):
                evaluate_with_thresholds()
        except Exception as e:
            logger.exception("❌ Errore durante la valutazione finale:")
            mlflow.set_tag("evaluate_status", "failed")
            raise

        # ✅ Step 4: Raw evaluation (soglia fissa 0.5)
        try:
            logger.info("📉 Step 4: Valutazione su test (threshold = 0.5)...")
            with mlflow.start_run(run_name="evaluate_raw_threshold_0.5", nested=True):
                evaluate()
        except Exception as e:
            logger.exception("❌ Errore durante la valutazione grezza:")
            mlflow.set_tag("evaluate_status", "failed")
            raise

        logger.info(f"✅ Esperimento completato. Run principale: {parent_run_id}")

if __name__ == "__main__":
    main()
