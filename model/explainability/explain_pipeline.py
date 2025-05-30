# explain_pipeline.py

"""
explain_pipeline.py

Orchestratore principale per l'explainability:
invoca perturber, features, predictor e surrogate, e restituisce
un unico dict con tutti i risultati.
"""

from typing import Dict, Any
from surrogate import explain_with_surrogates


def explain(text: str) -> Dict[str, Any]:
    """
    Esegue l'intero processo di spiegazione per una singola frase.

    Args:
      text: frase da spiegare.

    Returns:
      Dictionary con:
        - original_text
        - predictions
        - features
        - surrogates
    """
    return explain_with_surrogates(text)


# ========================
# CLI DI ESEMPIO (non usata da API)
# ========================
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Explain a text via local surrogates")
    parser.add_argument("text", type=str, help="Text to explain")
    args = parser.parse_args()

    result = explain(args.text)
    print(json.dumps(result, indent=2))
