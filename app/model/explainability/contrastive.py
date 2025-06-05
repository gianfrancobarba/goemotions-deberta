from typing import Dict, List
from model.explainability.predictor import predict_all

def get_contrastive_deltas(
    original_text: str,
    emotion: str,
    top_words: List[str],
    threshold: float = 0.2
) -> Dict[str, float]:
    """
    Calcola per ogni parola quanto diminuisce la probabilità dell’emozione
    se la parola viene rimossa dal testo originale.

    Ritorna solo le parole con delta > soglia.
    """
    deltas: Dict[str, float] = {}
    base_pred = predict_all([original_text])[0].get(emotion, 0.0)

    perturb_texts = []
    valid_words = []

    for word in top_words:
        tokens = original_text.split()
        new_tokens = [tok for tok in tokens if tok.lower() != word.lower()]
        new_text = " ".join(new_tokens)
        if new_text and new_text != original_text:
            perturb_texts.append(new_text)
            valid_words.append(word)

    if not perturb_texts:
        return {}

    preds = predict_all(perturb_texts)

    for word, pred in zip(valid_words, preds):
        p = pred.get(emotion, 0.0)
        delta = base_pred - p
        if delta > threshold:
            deltas[word] = round(delta, 3)

    return deltas
