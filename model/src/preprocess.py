# model/src/preprocess.py

import os
import logging
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from config.config import CFG

# Setup logging
log_path = os.path.join(CFG.logs_dir, "preprocess.log")
os.makedirs(CFG.logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_and_preprocess_dataset() -> DatasetDict:
    """
    Carica e preprocessa il dataset GoEmotions (semplified), tokenizza e binarizza le etichette.

    Returns:
        DatasetDict: contenente i dati tokenizzati e le etichette multilabel binarie
    """
    logger.info("🔄 Caricamento del dataset GoEmotions (config: 'simplified')...")
    dataset: DatasetDict = load_dataset("go_emotions", "simplified")

    logger.info("🔤 Inizializzazione del tokenizer DeBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    def preprocess(example):
        encoding = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=CFG.max_length
        )
        multilabel_vector = [0] * CFG.num_labels
        for idx in example["labels"]:
            if 0 <= idx < CFG.num_labels:
                multilabel_vector[idx] = 1
        encoding["labels"] = multilabel_vector
        return encoding

    logger.info("⚙️  Applicazione del preprocessing...")
    tokenized_dataset: DatasetDict = dataset.map(preprocess, batched=False)

    logger.info("✅ Preprocessing completato.")
    return tokenized_dataset


def save_tokenized_dataset(dataset: DatasetDict, save_dir: str):
    """
    Salva il dataset tokenizzato in formato HuggingFace Arrow.

    Args:
        dataset (DatasetDict): dataset tokenizzato
        save_dir (str): path in cui salvare i file
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"💾 Salvataggio del dataset preprocessato in {save_dir}...")
    dataset.save_to_disk(save_dir)
    logger.info("✅ Dataset salvato correttamente.")


# Esecuzione standalone
if __name__ == "__main__":
    tokenized_ds = load_and_preprocess_dataset()
    logger.info("📦 Primo esempio dal dataset tokenizzato:")
    logger.info(tokenized_ds["train"][0])
    save_tokenized_dataset(tokenized_ds, os.path.join(CFG.data_dir, "tokenized_dataset"))
