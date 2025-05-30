# surrogate.py

"""
surrogate.py

Costruisce surrogate models (Decision Tree) per spiegare
le emozioni predette da DeBERTa. Ritorna un dict strutturato
con regole, target e metriche.
"""

from typing import Any, Dict, List
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from perturber import generate_perturbations
from features import extract_features
from predictor import predict_all
from config_surrogate import THRESHOLD, N_PERTURBATIONS, MAX_DEPTH
from metrics import fidelity_score, sparsity_score, stability_score


def explain_with_surrogates(text: str) -> Dict[str, Any]:
    """
    Genera una spiegazione locale usando surrogate models.

    Args:
      text: frase da spiegare.

    Returns:
      {
        "original_text": str,
        "predictions": {emo: prob, ...},
        "features": [ {feat: val, ...}, ... ],
        "surrogates": {
            emo: {
               "rules": str,
               "target_vector": [0,1,...],
               "metrics": { "fidelity":float, ... }
            }, ...
        }
      }
    """
    # 1) perturbazioni
    perturbed: List[str] = [text] + generate_perturbations(text, n=N_PERTURBATIONS)

    # 2) features
    df: pd.DataFrame = extract_features(perturbed, as_dict=False)
    features_dicts: List[Dict] = extract_features(perturbed, as_dict=True)

    # 3) predizioni complete
    preds: List[Dict[str, float]] = predict_all(perturbed)
    base_preds = preds[0]

    # 4) emozioni da spiegare
    emos: List[str] = [e for e, p in base_preds.items() if p >= THRESHOLD]

    # 5) costruzione target binari
    surrogates: Dict[str, Any] = {}
    X = df.values
    feature_names = list(df.columns)

    for emo in emos:
        y = [int(p[emo] >= THRESHOLD) for p in preds]
        clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=0)
        clf.fit(X, y)
        rules = export_text(clf, feature_names=feature_names)

        # metriche
        fidelity = fidelity_score(clf, X, y)
        sparsity = sparsity_score(clf)
        stability = stability_score(y, perturbed)

        surrogates[emo] = {
            "rules": rules,
            "target_vector": y,
            "metrics": {
                "fidelity": fidelity,
                "sparsity": sparsity,
                "stability": stability
            }
        }

    return {
        "original_text": text,
        "predictions": base_preds,
        "features": features_dicts,
        "surrogates": surrogates
    }


# ========================
# TEST MANUALE
# ========================
if __name__ == "__main__":
    import json
    out = explain_with_surrogates("I am so happy and proud!")
    print(json.dumps(out, indent=2))
