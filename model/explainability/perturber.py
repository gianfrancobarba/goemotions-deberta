# perturber.py

"""
perturber.py

Genera perturbazioni testuali di un input:
- dropout
- masking
- shuffle
- synonym replacement
"""

import random
import re
from typing import List
from nltk.corpus import wordnet

# Strategie disponibili
STRATEGIES = ["dropout", "masking", "shuffle", "synonym"]


def dropout(text: str, p: float = 0.1) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if random.random() > p)


def masking(text: str, p: float = 0.15) -> str:
    tokens = text.split()
    return " ".join("[MASK]" if random.random() < p else t for t in tokens)


def shuffle(text: str, p: float = 0.1) -> str:
    tokens = text.split()
    n = int(len(tokens) * p)
    idx = list(range(len(tokens)))
    for _ in range(n):
        i, j = random.sample(idx, 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def synonym_replacement(text: str, p: float = 0.1) -> str:
    tokens = re.findall(r"\w+", text)
    new_tokens = []
    for t in tokens:
        if random.random() < p:
            syns = wordnet.synsets(t)
            if syns:
                lemmas = [l.name().replace("_", " ") for s in syns for l in s.lemmas()]
                if lemmas:
                    new_tokens.append(random.choice(lemmas))
                    continue
        new_tokens.append(t)
    return " ".join(new_tokens)


def apply_random_strategy(text: str, strategies: List[str]) -> str:
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
    strategies: List[str] = None
) -> List[str]:
    """
    Genera fino a n perturbazioni uniche di testo.
    Usa un contatore max_tries per evitare loop infiniti.

    Args:
      text: frase originale
      n: numero di perturbazioni desiderate
      strategies: lista di nomi di strategie (default tutte)

    Returns:
      Lista di perturbazioni (esclusa l’originale).
    """
    if strategies is None:
        strategies = STRATEGIES

    perturbations = set()
    tries = 0
    max_tries = n * len(strategies) * 3

    while len(perturbations) < n and tries < max_tries:
        new = apply_random_strategy(text, strategies)
        if new != text:
            perturbations.add(new)
        tries += 1

    return list(perturbations)
