"""
config_surrogate.py

Parametri globali per la pipeline di explainability locale.
Tutte le costanti sono centralizzate qui per facilità di manutenzione.
"""

import os
from typing import List


class CFG:
    # === SURROGATE MODEL PARAMETERS ===
    # Threshold per considerare un'emozione rilevata
    THRESHOLD: float = 0.5

    # Numero di perturbazioni da generare per ogni frase
    N_PERTURBATIONS: int = 40

    # Massima profondità dell'albero surrogato
    MAX_DEPTH: int = 3

    # === MODEL PREDICTOR PARAMETERS ===
    # Lunghezza massima dei token per il modello DeBERTa
    MAX_LEN: int = 128

    # === DEVICE ===
    DEVICE: str = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"


# Per comodità, esportiamo anche le costanti in cima al modulo
THRESHOLD = CFG.THRESHOLD
N_PERTURBATIONS = CFG.N_PERTURBATIONS
MAX_DEPTH = CFG.MAX_DEPTH
MAX_LEN = CFG.MAX_LEN
DEVICE = CFG.DEVICE
