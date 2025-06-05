# File: model/explainability/surrogate.py

"""
surrogate.py

Costruisce surrogate models (Decision Tree) per spiegare
le emozioni predette da DeBERTa. Ritorna un dict strutturato
con regole, target, metriche e importanza delle feature.
"""

from typing import Any, Dict, List
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from model.explainability.perturber import generate_perturbations
from model.explainability.features import extract_features
from model.explainability.predictor import predict_all
from model.explainability.config_surrogate import THRESHOLD, N_PERTURBATIONS, MAX_DEPTH
from model.explainability.metrics import fidelity_score, sparsity_score, stability_score


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
               "metrics": { "fidelity":float, ... },
               "feature_importances": {
                   "bow": { feat: importance, ... },
                   "aggregated": { feat: importance, ... }
               }
            }, ...
        }
      }
    """
    # 1) Genero n perturbazioni + testo originale
    perturbed: List[str] = [text] + generate_perturbations(text, n=N_PERTURBATIONS)

    # 2) Estraggo feature interpretabili per ogni variante
    df: pd.DataFrame = extract_features(perturbed, as_dict=False)
    features_dicts: List[Dict] = extract_features(perturbed, as_dict=True)

    # 3) Calcolo predizioni di DeBERTa per ogni variante
    preds: List[Dict[str, float]] = predict_all(perturbed)
    base_preds = preds[0]  # predizione sul testo originale

    # 4) Seleziono quali emozioni spiegare (quelle con probabilità ≥ soglia)
    emos: List[str] = [e for e, p in base_preds.items() if p >= THRESHOLD]

    # 5) Costruisco i surrogate model (decision tree) per ciascuna emozione
    surrogates: Dict[str, Any] = {}
    X = df.values
    feature_names = list(df.columns)

    for emo in emos:
        # 5.1) Vettore target binario: 1 se la variante ha prob ≥ soglia, altrimenti 0
        y = [int(p[emo] >= THRESHOLD) for p in preds]

        # 5.2) Alleno un DecisionTreeClassifier con profondità massima MAX_DEPTH
        clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=0)
        clf.fit(X, y)

        # 5.3) Estraggo le regole testuali dal decision tree
        rules = export_text(clf, feature_names=feature_names)

        # 5.4) Calcolo le metriche di qualità del surrogate
        fidelity = fidelity_score(clf, X, y)
        sparsity = sparsity_score(clf)
        stability = stability_score(y, perturbed)

        # 5.5) Prendo tutte le importances
        all_importances = dict(zip(feature_names, clf.feature_importances_))

        # 5.6) Recupero le feature del testo originale (non perturbato)
        original_features = features_dicts[0]

        # 5.7) Separo:
        # - le bow_ presenti nel testo originale
        # - le feature aggregate significative
        bow_importances = {
            k: v for k, v in all_importances.items()
            if k.startswith("bow_") and original_features.get(k, 0) > 0 and v > 0
        }
        aggregated_importances = {
            k: v for k, v in all_importances.items()
            if not k.startswith("bow_") and v > 0
        }

        # 5.8) Costruisco dizionario separato per bow vs aggregate
        importances = {
            "bow": bow_importances,
            "aggregated": aggregated_importances
        }

        # 5.9) Salvo risultati del surrogate
        surrogates[emo] = {
            "rules": rules,
            "target_vector": y,
            "metrics": {
                "fidelity": fidelity,
                "sparsity": sparsity,
                "stability": stability
            },
            "feature_importances": importances
        }

    # 6) Ritorno struttura completa
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
    print(json.dumps(out, indent=2, ensure_ascii=False))
