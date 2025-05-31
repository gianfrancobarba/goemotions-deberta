# metrics.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from typing import List

def fidelity_score(surrogate_model: DecisionTreeClassifier, X, y_true) -> float:
    """
    Calcola la fedeltà del modello surrogato rispetto alle etichette binarie derivate
    dal modello originale. È l'accuratezza delle predizioni del surrogato.
    """
    y_pred = surrogate_model.predict(X)
    return accuracy_score(y_true, y_pred)

def sparsity_score(surrogate_model: DecisionTreeClassifier) -> float:
    """
    Calcola una misura di sparsità: proporzione di nodi di split rispetto al numero massimo possibile.
    Più basso è il valore, più l'albero è sparso (più interpretabile).
    """
    n_nodes = surrogate_model.tree_.node_count
    max_possible = 2 ** surrogate_model.get_depth() - 1
    return n_nodes / max(max_possible, 1)

def stability_score(y_true: List[int], texts: List[str]) -> float:
    """
    Calcola una proxy della stabilità: quante volte l’etichetta della frase originale
    si mantiene uguale anche nelle perturbazioni. Assumiamo che y_true[0] sia la label
    della frase originale, il resto sono le perturbazioni.
    """
    if len(y_true) <= 1:
        return 1.0

    base = y_true[0]
    perturbed = y_true[1:]
    same = sum(1 for y in perturbed if y == base)
    return same / len(perturbed)
