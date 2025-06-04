import os
import logging
import csv
from typing import Dict, List, Any
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from config.loader import CFG

# Setup logging
os.makedirs(CFG.paths.model_logs, exist_ok=True)
log_path = os.path.join(CFG.paths.model_logs, "preprocess.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_raw_dataset(remove_neutral: bool = True) -> DatasetDict:
    """
    Carica i file .tsv locali di GoEmotions (train/dev/test) come DatasetDict.
    """
    logger.info("Caricamento file TSV da directory locale...")

    def read_tsv(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            examples = []
            for row in reader:
                if len(row) != 3:
                    continue
                text, labels_str, _ = row
                label_ids = list(map(int, labels_str.split(",")))
                if remove_neutral and 27 in label_ids:
                    continue
                examples.append({"text": text, "labels": label_ids})
            return examples

    base_path = CFG.paths.raw_data_dir
    data_dict = {
        "train": read_tsv(os.path.join(base_path, "train.tsv")),
        "validation": read_tsv(os.path.join(base_path, "dev.tsv")),
        "test": read_tsv(os.path.join(base_path, "test.tsv"))
    }

    logger.info("Conversione in HuggingFace DatasetDict...")
    return DatasetDict({k: Dataset.from_list(v) for k, v in data_dict.items()})


def initialize_tokenizer():
    logger.info("Inizializzazione del tokenizer...")
    return AutoTokenizer.from_pretrained(CFG.model.name)


def preprocess_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=CFG.model.max_length
    )
    multilabel_vector = [0] * CFG.model.num_labels
    for idx in example["labels"]:
        if 0 <= idx < CFG.model.num_labels:
            multilabel_vector[idx] = 1
    encoding["labels"] = multilabel_vector
    return encoding


def preprocess_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    logger.info("Applicazione tokenizzazione e binarizzazione etichette...")
    return dataset.map(lambda x: preprocess_example(x, tokenizer), batched=False)


def save_tokenized_dataset(dataset: DatasetDict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Salvataggio dataset preprocessato in {save_dir}...")
    dataset.save_to_disk(save_dir)
    logger.info("Dataset salvato correttamente.")


def load_and_preprocess_dataset(remove_neutral: bool = True) -> DatasetDict:
    dataset = load_raw_dataset(remove_neutral=remove_neutral)
    tokenizer = initialize_tokenizer()
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)
    logger.info("Preprocessing completato.")
    return tokenized_dataset


def main():
    tokenized_ds = load_and_preprocess_dataset(remove_neutral=True)
    logger.info("Visualizzazione di un esempio tokenizzato:")
    logger.info(tokenized_ds["train"][0])
    save_tokenized_dataset(tokenized_ds, CFG.paths.tokenized_data_dir)


# Compatibile con: python -m app.utils.preprocess
if __name__ == "__main__":
    main()
