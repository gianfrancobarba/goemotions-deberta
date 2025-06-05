from contextlib import contextmanager
import mlflow

@contextmanager
def start_or_continue_run(run_name: str):
    """
    Se esiste una run attiva, avvia run nested.
    Altrimenti, avvia una nuova run principale.
    """
    if mlflow.active_run() is None:
        with mlflow.start_run(run_name=run_name):
            yield
    else:
        with mlflow.start_run(run_name=run_name, nested=True):
            yield
