# model/src/preprocess.py

"""
Modulo di preprocessing per il dataset GoEmotions (semplified).

Funzionalità:
- Caricamento del dataset dalla libreria HuggingFace
- Tokenizzazione del campo 'text' con il tokenizer DeBERTa v3 base
- Conversione della colonna 'labels' da lista di indici a vettore multilabel binario
- Restituzione di un oggetto DatasetDict pronto per la fase di training

Dataset: go_emotions (config: "simplified")
Modello: microsoft/deberta-v3-base
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Nome del modello HuggingFace
MODEL_NAME = "microsoft/deberta-v3-base"

# Numero totale di etichette nel dataset (GoEmotions = 28 classi)
NUM_LABELS = 28

# Lunghezza massima del padding per la tokenizzazione
MAX_LENGTH = 128


def load_and_preprocess_dataset() -> DatasetDict:
    """
    Carica e preprocessa il dataset GoEmotions (semplified).

    Returns:
        DatasetDict: contenente i dati tokenizzati e le etichette multilabel binarie
    """
    print("Caricamento del dataset GoEmotions (config: 'simplified')...")
    dataset = load_dataset("go_emotions", "simplified")

    print("Inizializzazione del tokenizer DeBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(example):
        # Tokenizzazione del testo
        encoding = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        # Inizializza un vettore binario di 28 zeri
        multilabel_vector = [0] * NUM_LABELS
        for idx in example["labels"]:
            if 0 <= idx < NUM_LABELS:
                multilabel_vector[idx] = 1
        encoding["labels"] = multilabel_vector
        return encoding

    print("Applicazione del preprocessing...")
    tokenized_dataset = dataset.map(preprocess, batched=False)

    print("Preprocessing completato.")
    return tokenized_dataset


# Esecuzione standalone (sviluppo e debug)
if __name__ == "__main__":
    ds = load_and_preprocess_dataset()
    print("Esempio dal dataset tokenizzato:")
    print(ds["train"][0])
