# run_experiment.py

import mlflow
import os
from config.loader import CFG

# Imposta MLflow
mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
mlflow.set_experiment(CFG.mlflow.experiment_name)

def main():
    with mlflow.start_run(run_name=CFG.mlflow.run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        print("\n🚀 Step 1: Training...")
        with mlflow.start_run(run_name="train", nested=True):
            os.system("python -m model.training.train")

        print("\n🎯 Step 2: Tuning thresholds...")
        with mlflow.start_run(run_name="tune_thresholds", nested=True):
            os.system("python -m model.thresholds.tune_thresholds")

        print("\n📊 Step 3: Final evaluation...")
        with mlflow.start_run(run_name="evaluate", nested=True):
            os.system("python -m model.training.evaluate_with_thresholds")

        print(f"\n✅ Esperimento completo. Run principale: {parent_run_id}")

if __name__ == "__main__":
    main()
