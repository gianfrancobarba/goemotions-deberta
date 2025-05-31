# visualizer.py

from typing import Dict
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from termcolor import colored


def print_rules_and_metrics(surrogate_outputs: Dict[str, Dict]):
    """
    Stampa le regole e le metriche per ogni emozione spiegata.
    """
    for emotion, info in surrogate_outputs.items():
        print(colored(f"\nSurrogato per '{emotion}':", "green"))
        print(info["tree_rules"])

        metrics = info.get("metrics", {})
        if metrics:
            print(colored(f"  Fidelity: {metrics.get('fidelity', 0):.2f}", "cyan"))
            print(colored(f"  Sparsity: {metrics.get('sparsity', 0):.2f}", "cyan"))
            print(colored(f"  Stability: {metrics.get('stability', 0):.2f}", "cyan"))


def plot_surrogates(surrogate_outputs: Dict[str, Dict], classifiers: Dict[str, DecisionTreeClassifier], feature_names: list):
    """
    Plotta graficamente ogni albero surrogato con i nomi delle feature.
    """
    for emotion, clf in classifiers.items():
        plt.figure(figsize=(10, 5))
        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=["0", "1"],
            filled=True,
            rounded=True,
            fontsize=9
        )
        plt.title(f"Surrogate Tree for '{emotion}'")
        plt.tight_layout()
        plt.show()
