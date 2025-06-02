# metrics.py

"""
metrics.py

Metriche di valutazione per i surrogate models:
- Fidelity: accuratezza del surrogato rispetto al modello originale
- Sparsity: semplicità dell’albero (nodi di split / nodi totali)
- Stability: coerenza delle predizioni su perturbazioni
"""

from typing import List
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def fidelity_score(
    surrogate_model: DecisionTreeClassifier,
    X: np.ndarray,
    y_true: List[int]
) -> float:
    """
    Calcola la fedeltà: percentuale di predizioni corrette
    del surrogato rispetto al vettore target y_true.
    """
    y_pred = surrogate_model.predict(X)
    return accuracy_score(y_true, y_pred)


def sparsity_score(
    surrogate_model: DecisionTreeClassifier
) -> float:
    """
    Calcola la sparsità come rapporto tra nodi di split e nodi totali:
    (node_count - n_leaves) / node_count.
    Più il valore è basso, più l’albero è sparso (interpretabile).
    """
    tree = surrogate_model.tree_
    n_nodes = tree.node_count
    # children_left == -1 identifica foglie
    n_leaves = np.count_nonzero(tree.children_left == -1)
    return (n_nodes - n_leaves) / n_nodes if n_nodes > 0 else 0.0


def stability_score(
    y_true: List[int],
    texts: List[str]
) -> float:
    """
    Calcola la stabilità come percentuale di predizioni del surrogato
    (= y_true) che rimangono identiche alla prima (base) su tutte le perturbazioni.

    Args:
      y_true: lista di label binarie (prima la base, poi perturbazioni)
      texts: lista delle frasi (non usato nella metrica, ma mantenuto per compatibilità)

    Returns:
      Valore compreso tra 0 e 1.
    """
    if len(y_true) <= 1:
        return 1.0
    base = y_true[0]
    rest = y_true[1:]
    stable = sum(1 for y in rest if y == base)
    return stable / len(rest)