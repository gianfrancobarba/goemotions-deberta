# File: perturber.py

"""
perturber.py

Genera perturbazioni testuali di un input:
- dropout (rimozione casuale di parole)
- masking (sostituzione di parole con <mask>)
- shuffle (scambio casuale di alcune parole)
- synonym replacement (sostituzione con sinonimi WordNet più accurati)
"""

import random
import re
from typing import List, Optional
from nltk.corpus import wordnet

# Strategie disponibili
STRATEGIES = ["dropout", "masking", "shuffle", "synonym"]


def dropout(text: str, p: float = 0.1) -> str:
    """
    Rimuove casualmente ogni token con probabilità p.
    """
    tokens = text.split()
    # Mantiene solo i token che non vengono “bucati”
    return " ".join(t for t in tokens if random.random() > p)


def masking(text: str, p: float = 0.15) -> str:
    """
    Con probabilità p sostituisce ciascun token con il token <mask>,
    che DeBERTa (come BERT/Roberta) utilizza per il masked‐language modeling.
    """
    tokens = text.split()
    masked = []
    for tok in tokens:
        if random.random() < p:
            masked.append("<mask>")
        else:
            masked.append(tok)
    return " ".join(masked)


def shuffle(text: str, p: float = 0.1) -> str:
    """
    Scambia casualmente circa il 10% dei token:
    calcola n_swaps = floor(len(tokens) * p) e spruzza n_swaps scambi tra posizioni a caso.
    """
    tokens = text.split()
    n_swaps = max(1, int(len(tokens) * p))
    idx = list(range(len(tokens)))
    for _ in range(n_swaps):
        i, j = random.sample(idx, 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def synonym_replacement(text: str, p: float = 0.1) -> str:
    """
    Con prob. p prova a sostituire ciascun token con un sinonimo ricavato da WordNet,
    mantenendo la punteggiatura e rispettando la parte del discorso.
    """
    def get_pos_tag(word: str) -> Optional[str]:
        """
        Ritorna la parte del discorso (n, v, a, r) del primo synset trovato,
        oppure None se non si trova nulla.
        """
        synsets = wordnet.synsets(word)
        if not synsets:
            return None
        pos = synsets[0].pos()
        if pos.startswith("n"):
            return "n"
        if pos.startswith("v"):
            return "v"
        if pos.startswith("a") or pos.startswith("s"):
            return "a"
        if pos.startswith("r"):
            return "r"
        return None

    # Manteniamo la punteggiatura: split “intelligente” (non perdere punti o virgole)
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    new_tokens = []

    for tok in tokens:
        # Applichiamo solo ai token alfanumerici (no punteggiatura)
        if re.match(r"^\w+$", tok) and random.random() < p:
            lower = tok.lower()
            pos = get_pos_tag(lower)
            if pos:
                synsets = wordnet.synsets(lower, pos=pos)
                # Raccolgo tutti i lemma-names (senza underscore) dello stesso POS
                lemmas = [
                    l.name().replace("_", " ")
                    for s in synsets
                    for l in s.lemmas()
                    if "_" not in l.name()
                ]
                # Rimuovo duplicati e la parola stessa
                lemmas = list({l for l in lemmas if l.lower() != lower})
                if lemmas:
                    choice = random.choice(lemmas)
                    # Se il token originale era in maiuscolo, capitalizzo anche il sinonimo
                    if tok[0].isupper():
                        choice = choice.capitalize()
                    new_tokens.append(choice)
                    continue
        # Altrimenti lascio il token invariato
        new_tokens.append(tok)

    # Ricostruisco la frase, rimettendo spazio dopo ogni parola
    rebuilt = []
    for tok in new_tokens:
        if re.match(r"[^\w\s]", tok):
            # se è solo punteggiatura, la attacco dopo la parola precedente
            if rebuilt:
                rebuilt[-1] = rebuilt[-1] + tok
            else:
                rebuilt.append(tok)
        else:
            rebuilt.append(tok + " ")
    return "".join(rebuilt).strip()


def apply_random_strategy(text: str, strategies: List[str]) -> str:
    """
    Sceglie a caso una strategia e la applica.
    """
    strat = random.choice(strategies)
    if strat == "dropout":
        return dropout(text)
    if strat == "masking":
        return masking(text)
    if strat == "shuffle":
        return shuffle(text)
    if strat == "synonym":
        return synonym_replacement(text)
    raise ValueError(f"Unknown strategy: {strat}")


def generate_perturbations(
    text: str,
    n: int,
    strategies: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[str]:
    """
    Genera fino a n perturbazioni uniche di testo.
    Usa un contatore max_tries per evitare loop infiniti.

    Args:
      text: frase originale
      n: numero di perturbazioni desiderate
      strategies: lista di strategie (default STRATEGIES)
      seed: seed opzionale per riproducibilità

    Returns:
      Lista di perturbazioni uniche (escluse eventuali ripetizioni e l’originale).
    """
    if seed is not None:
        random.seed(seed)

    if strategies is None:
        strategies = STRATEGIES

    perturbations = set()
    tries = 0
    max_tries = n * len(strategies) * 3

    while len(perturbations) < n and tries < max_tries:
        new_text = apply_random_strategy(text, strategies)
        if new_text != text:
            perturbations.add(new_text)
        tries += 1

    return list(perturbations)
