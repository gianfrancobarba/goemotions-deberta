import os
import logging
import csv
from typing import Dict, List, Any
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from config.loader import CFG

# Setup logging
log_path = os.path.join(CFG.paths["logs"], "preprocess.log")
os.makedirs(CFG.paths["logs"], exist_ok=True)
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

    Args:
        remove_neutral (bool): se True, rimuove gli esempi con etichetta 'neutral' (id = 27)

    Returns:
        DatasetDict: dataset completo
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

    base_path = CFG.paths["raw_data_dir"]
    data_dict = {
        "train": read_tsv(os.path.join(base_path, "train.tsv")),
        "validation": read_tsv(os.path.join(base_path, "dev.tsv")),
        "test": read_tsv(os.path.join(base_path, "test.tsv"))
    }

    logger.info("Conversione in HuggingFace DatasetDict...")
    dataset_dict = DatasetDict({
        k: Dataset.from_list(v) for k, v in data_dict.items()
    })

    return dataset_dict


def initialize_tokenizer():
    logger.info("Inizializzazione del tokenizer...")
    return AutoTokenizer.from_pretrained(CFG.model["name"])


def preprocess_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=CFG.model["max_length"]
    )
    multilabel_vector = [0] * CFG.model["num_labels"]
    for idx in example["labels"]:
        if 0 <= idx < CFG.model["num_labels"]:
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


if __name__ == "__main__":
    tokenized_ds = load_and_preprocess_dataset(remove_neutral=True)
    logger.info("Visualizzazione di un esempio tokenizzato:")
    logger.info(tokenized_ds["train"][0])
    save_tokenized_dataset(tokenized_ds, CFG.paths["tokenized_data_dir"])
