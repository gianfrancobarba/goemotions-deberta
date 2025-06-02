# File: model/explainability/config_surrogate.py

import os
from typing import List


class CFG:
    # === SURROGATE MODEL PARAMETERS ===
    # Soglia per considerare un’emozione “attiva”
    THRESHOLD: float = 0.5

    # Numero di perturbazioni generate per ogni frase
    # (aumentato per migliorare la copertura del contesto e la fedeltà del surrogate)
    N_PERTURBATIONS: int = 100

    # Profondità massima degli alberi surrogate
    # (aumentata per permettere albero più complesso e quindi potenzialmente più accurato)
    MAX_DEPTH: int = 5

    # Semi da usare per la creazione dell’ensemble di surrogate
    # (seeds diversi per ogni Decision Tree)
    SEEDS: List[int] = [42, 43, 44]

    # === MODEL PREDICTOR PARAMETERS ===
    # Lunghezza massima dei token per il modello DeBERTa (non cambiato)
    MAX_LEN: int = 128

    # === DEVICE ===
    DEVICE: str = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"


# Esportiamo le costanti per comodità
THRESHOLD = CFG.THRESHOLD
N_PERTURBATIONS = CFG.N_PERTURBATIONS
MAX_DEPTH = CFG.MAX_DEPTH
SEEDS = CFG.SEEDS
MAX_LEN = CFG.MAX_LEN
DEVICE = CFG.DEVICE