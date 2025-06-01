# File: model/explainability/explain_pipeline.py

from typing import Dict, Any
from model.explainability.surrogate import explain_with_surrogates
from model.explainability.verbalizer import verbalize_explanation

def explain(text: str) -> Dict[str, Any]:
    """
    Esegue l'intero processo di spiegazione per una singola frase.

    Args:
      text: frase da spiegare.

    Returns:
      Dict contenente:
        - original_text: il testo originale
        - predictions: dizionario emotion -> score
        - features: lista di vettori di features
        - surrogates: dizionario con surrogate model data
        - explanations: dizionario emotion -> spiegazione strutturata
    """
    # 1) Ottengo il result grezzo da explain_with_surrogates
    result = explain_with_surrogates(text)

    # 2) Aggiungo il testo originale (utile per i conteggi reali)
    result["original_text"] = text

    # 3) Genero le spiegazioni strutturate tramite verbalizer (compreso il simple_explanation)
    result["explanations"] = verbalize_explanation(result)

    return result


# ========================
# CLI DI ESEMPIO (non usata dall’API, ma utile per testing da linea di comando)
# ========================
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Explain a text via local surrogates")
    parser.add_argument("text", type=str, help="Text to explain")
    args = parser.parse_args()

    result = explain(args.text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
