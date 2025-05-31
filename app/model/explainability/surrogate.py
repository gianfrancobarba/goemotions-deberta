import pandas as pd
from typing import Dict

from sklearn.tree import DecisionTreeClassifier, export_text

from perturber import generate_perturbations
from features import extract_features
from predictor import predict_all
from config_surrogate import THRESHOLD, N_PERTURBATIONS, MAX_DEPTH
from metrics import fidelity_score, sparsity_score, stability_score


def explain_with_surrogates(text: str) -> Dict[str, Dict]:
    """
    Addestra un albero surrogato locale per ciascuna emozione rilevata ≥ soglia.
    Ritorna un dizionario con le regole e i target binari per ogni emozione.
    """
    # 1. Genera perturbazioni (inclusa la frase originale)
    perturbed_texts = [text] + generate_perturbations(text, n=N_PERTURBATIONS)

    # 2. Estrai feature interpretabili
    df_features = extract_features(perturbed_texts)

    # 3. Ottieni predizioni multilabel del modello
    predictions = predict_all(perturbed_texts)

    # 4. Emozioni da spiegare = quelle con score ≥ soglia nella frase originale
    base_probs = predictions[0]
    emotions_above_threshold = [emo for emo, score in base_probs.items() if score >= THRESHOLD]

    if not emotions_above_threshold:
        return {}

    # 5. Costruisci target binari per ciascuna emozione da spiegare
    emotion_targets = {
        emotion: [int(p[emotion] >= THRESHOLD) for p in predictions]
        for emotion in emotions_above_threshold
    }

    # 6. Addestra un albero per ciascuna emozione
    result = {}
    X = df_features.values
    feature_names = list(df_features.columns)

    for emotion in emotions_above_threshold:
        y = emotion_targets[emotion]

        clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=0)
        clf.fit(X, y)

        rules = export_text(clf, feature_names=feature_names)

        # === METRICHE specifiche per questo surrogato ===
        fidelity = fidelity_score(clf, X, y)
        sparsity = sparsity_score(clf)
        stability = stability_score(y, perturbed_texts)

        print(f"\nSurrogato per '{emotion}':")
        print(rules)
        print(f"  Fidelity: {round(fidelity, 2)}")
        print(f"  Sparsity: {round(sparsity, 2)}")
        print(f"  Stability: {round(stability, 2)}")

        result[emotion] = {
            "target_vector": y,
            "tree_rules": rules,
            "metrics": {
                "fidelity": fidelity,
                "sparsity": sparsity,
                "stability": stability
            }
        }

    return result


# ========================
# TEST MANUALE
# ========================
if __name__ == "__main__":
    test_input = "I finally did it, I'm so proud of myself!"
    output = explain_with_surrogates(test_input)

    print(f"\n[✓] Emozioni spiegate: {list(output.keys())}\n")
    for emotion, info in output.items():
        print(f"Surrogato per '{emotion}':")
        print(info["tree_rules"])
