import logging
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

from app.config.loader import CFG
from app.utils.preprocess import load_and_preprocess_dataset
from app.model.training.train_utils import (
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

    # Seed
    set_seed(CFG.model.seed)

    logger.info("Caricamento del dataset GoEmotions (config: 'simplified')...")
    dataset = load_and_preprocess_dataset()

    logger.info(f"Costruzione modello {CFG.model.name} con {CFG.model.num_labels} classi...")
    model = CustomMultiLabelModel(
        model_name=CFG.model.name,
        num_labels=CFG.model.num_labels,
        pos_weight=None
    )

    logger.info("Inizializzazione del tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.name)

    logger.info("Configurazione TrainingArguments...")
    training_args = get_training_args(CFG.model.dir)

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
    from transformers import logging
    logging.set_verbosity_error()
    
    train()

