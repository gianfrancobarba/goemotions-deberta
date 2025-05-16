# model/src/preprocess.py

import os
import logging
from typing import Dict, List, Any
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


def load_raw_dataset(remove_neutral: bool = True) -> DatasetDict:
    """
    Carica il dataset GoEmotions (configurazione 'simplified') dalla libreria HuggingFace.

    Args:
        remove_neutral (bool): se True, rimuove gli esempi che includono l'etichetta 'neutral' (27)

    Returns:
        DatasetDict: dataset con (eventuale) filtro sugli esempi contenenti 'neutral'
    """
    logger.info("Caricamento del dataset GoEmotions ('simplified')...")
    dataset: DatasetDict = load_dataset("go_emotions", "simplified")

    if remove_neutral:
        logger.info("Rimozione degli esempi contenenti l'etichetta 'neutral' (27)...")

        def filter_neutral(example: Dict[str, Any]) -> bool:
            return 27 not in example["labels"]

        dataset = dataset.filter(filter_neutral)

    logger.info("Dataset caricato con successo.")
    return dataset


def initialize_tokenizer():
    """
    Inizializza il tokenizer basato sul modello specificato nella configurazione.

    Returns:
        AutoTokenizer: tokenizer HuggingFace
    """
    logger.info("Inizializzazione del tokenizer DeBERTa...")
    return AutoTokenizer.from_pretrained(CFG.model_name)


def preprocess_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Tokenizza un esempio testuale e binarizza le etichette multilabel.

    Args:
        example (dict): esempio contenente il testo e le etichette originali
        tokenizer: tokenizer HuggingFace

    Returns:
        dict: esempio arricchito con input_ids, attention_mask e labels binarizzati
    """
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=CFG.max_length
    )

    # Inizializza un vettore multilabel binario
    multilabel_vector = [0] * CFG.num_labels
    for idx in example["labels"]:
        if 0 <= idx < CFG.num_labels:
            multilabel_vector[idx] = 1

    encoding["labels"] = multilabel_vector
    return encoding


def preprocess_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """
    Applica la funzione di preprocessamento (tokenizzazione + etichette binarie) al dataset.

    Args:
        dataset (DatasetDict): dataset originale
        tokenizer: tokenizer HuggingFace

    Returns:
        DatasetDict: dataset tokenizzato e con labels binarie
    """
    logger.info("Applicazione della tokenizzazione e binarizzazione etichette...")
    return dataset.map(lambda x: preprocess_example(x, tokenizer), batched=False)


def save_tokenized_dataset(dataset: DatasetDict, save_dir: str):
    """
    Salva il dataset preprocessato in formato HuggingFace Arrow.

    Args:
        dataset (DatasetDict): dataset tokenizzato
        save_dir (str): directory di destinazione
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Salvataggio del dataset preprocessato in {save_dir}...")
    dataset.save_to_disk(save_dir)
    logger.info("Dataset salvato correttamente.")


def load_and_preprocess_dataset(remove_neutral: bool = True) -> DatasetDict:
    """
    Pipeline completa: carica, filtra, tokenizza e restituisce il dataset preprocessato.

    Args:
        remove_neutral (bool): se True, rimuove gli esempi con etichetta 'neutral'

    Returns:
        DatasetDict: dataset pronto per il training
    """
    dataset = load_raw_dataset(remove_neutral=remove_neutral)
    tokenizer = initialize_tokenizer()
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)
    logger.info("Preprocessing completato.")
    return tokenized_dataset


# Esecuzione standalone
if __name__ == "__main__":
    tokenized_ds = load_and_preprocess_dataset(remove_neutral=True)
    logger.info("Visualizzazione di un esempio dal dataset tokenizzato:")
    logger.info(tokenized_ds["train"][0])
    save_tokenized_dataset(tokenized_ds, os.path.join(CFG.data_dir, "tokenized_dataset"))
